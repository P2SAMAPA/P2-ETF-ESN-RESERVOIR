"""engine.py — ESN walk-forward engine.

Pipeline per universe
---------------------
1. Build lagged input matrix X (T, features) and target matrix Y (T, n_etf)
2. Split into warm-up / train / validate / OOS periods (see config.py)
3. Initial fit of ESN ensemble on train set
4. Walk-forward over OOS: every REFIT_FREQ days re-fit on rolling REFIT_WINDOW
5. Daily score = ESN_FORECAST_WT * esn_pred_z + MOMENTUM_WT * ewm_mom_z
6. Rank ETFs cross-sectionally; output top-N + full history DataFrames
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from data_manager import build_input_matrix
from reservoir import ESNEnsemble


# ── Helpers ───────────────────────────────────────────────────────────────────

def _zscore_cross(arr: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score of a 1-D array (one row of scores)."""
    mu  = arr.mean()
    std = arr.std() + 1e-8
    return (arr - mu) / std


def _ewm_return(log_returns: np.ndarray, span: int) -> np.ndarray:
    """EWM of a (T, n) array along axis 0. Returns last row annualised."""
    alpha = 2.0 / (span + 1)
    ewm   = np.zeros_like(log_returns)
    ewm[0] = log_returns[0]
    for t in range(1, len(log_returns)):
        ewm[t] = alpha * log_returns[t] + (1 - alpha) * ewm[t - 1]
    return ewm[-1] * 252  # annualise


# ── Main engine ───────────────────────────────────────────────────────────────

def run_engine(
    log_returns: pd.DataFrame,
    macro_df: pd.DataFrame,
    universe_tickers: list[str],
    universe_name: str,
) -> dict:
    """Run the ESN engine for one universe.

    Returns a dict with daily scoring history and the latest snapshot.
    """
    avail = [t for t in universe_tickers if t in log_returns.columns]
    n_etf = len(avail)

    print(
        f"\n{'='*60}\n"
        f"Universe: {universe_name}  ({n_etf} ETFs)\n"
        f"Period: {log_returns.index[0].date()} → {log_returns.index[-1].date()}"
        f"  ({len(log_returns)} days)\n"
        f"{'='*60}"
    )

    # ── Build input / target matrices ─────────────────────────────────────────
    # X : (T', n_features)   input to reservoir
    # Y : (T', n_etf)        1-day ahead log return for each ETF
    # dates: DatetimeIndex aligned to row i  (= date of Y[i])
    X, dates = build_input_matrix(log_returns, macro_df, avail)
    # Target: shift returns 1 day forward relative to X
    ret_arr = log_returns[avail].reindex(dates).values     # (T', n_etf)
    # Y[i] = return on date dates[i]  (the day the prediction is *for*)
    # X[i] = features built from window ending at dates[i]-1
    Y = ret_arr.astype(np.float32)

    # ── Date masks ────────────────────────────────────────────────────────────
    warmup_mask   = dates <= config.WARMUP_END
    train_mask    = (dates > config.WARMUP_END)  & (dates <= config.TRAIN_END)
    val_mask      = (dates > config.TRAIN_END)   & (dates <= config.VALIDATE_END)
    oos_mask      = dates > config.VALIDATE_END

    n_warmup = warmup_mask.sum()
    n_train  = train_mask.sum()
    n_val    = val_mask.sum()
    n_oos    = oos_mask.sum()

    print(
        f"  Warm-up: {n_warmup}d  Train: {n_train}d  "
        f"Val: {n_val}d  OOS: {n_oos}d"
    )

    # ── Initial fit on warm-up + train set ───────────────────────────────────
    init_end  = n_warmup + n_train
    U_init    = X[:init_end]
    Y_init    = Y[:init_end]

    n_features = X.shape[1]
    ensemble   = ESNEnsemble(n_inputs=n_features, n_outputs=n_etf)
    ensemble.fit(U_init, Y_init, warmup_steps=n_warmup)
    print(f"  Initial fit complete ({n_features} input features, "
          f"{config.RESERVOIR_SIZE} reservoir nodes × {config.N_ENSEMBLES} ensembles)")

    # ── Walk-forward scoring over validate + OOS ─────────────────────────────
    score_start = n_warmup + n_train   # first index to score from

    # Warm up reservoir on train data before walk-forward
    ensemble.reset_states()
    for t in range(score_start):
        ensemble.predict_one(X[t])    # drive state forward, discard prediction

    daily_records    : list[dict] = []
    score_records    : list[dict] = []
    ranking_records  : list[dict] = []

    last_refit_at = score_start

    for t in range(score_start, len(X)):
        date = dates[t]

        # ── Periodic refit on rolling window ──────────────────────────────────
        if (t - last_refit_at) >= config.REFIT_FREQ:
            window_start = max(0, t - config.REFIT_WINDOW)
            U_fit = X[window_start:t]
            Y_fit = Y[window_start:t]
            warmup_steps = min(50, len(U_fit) // 10)
            ensemble.fit(U_fit, Y_fit, warmup_steps=warmup_steps)
            last_refit_at = t

        # ── ESN forecast ──────────────────────────────────────────────────────
        esn_pred = ensemble.predict_one(X[t])       # (n_etf,)

        # ── EWM momentum signal ───────────────────────────────────────────────
        ret_window = log_returns[avail].iloc[
            max(0, t - config.EWM_SPAN_RETURN * 3) : t
        ].values
        if len(ret_window) < 5:
            mom = np.zeros(n_etf)
        else:
            mom = _ewm_return(ret_window, config.EWM_SPAN_RETURN)

        # ── Composite score (cross-sectionally z-scored then blended) ─────────
        esn_z = _zscore_cross(esn_pred)
        mom_z = _zscore_cross(mom)
        composite = (config.ESN_FORECAST_WT * esn_z
                     + config.MOMENTUM_WT    * mom_z)

        # ── Rank ETFs ─────────────────────────────────────────────────────────
        ranked_idx = np.argsort(composite)[::-1]
        top_ticker = avail[ranked_idx[0]]
        top_score  = float(composite[ranked_idx[0]])

        # CASH flag
        cash_flag = top_score < config.CASH_THRESHOLD

        # ── Records ───────────────────────────────────────────────────────────
        ds = date.strftime("%Y-%m-%d")

        daily_records.append({
            "date":       ds,
            "top_ticker": "CASH" if cash_flag else top_ticker,
            "top_score":  round(top_score, 6),
            "cash_flag":  cash_flag,
            "refit_day":  (t == last_refit_at),
        })

        score_records.append({
            "date": ds,
            **{avail[i]: round(float(composite[i]), 6) for i in range(n_etf)},
        })

        ranking_records.append({
            "date": ds,
            **{avail[ranked_idx[r]]: r + 1 for r in range(n_etf)},
        })

        if t % 252 == 0 or t == len(X) - 1:
            top5 = [(avail[ranked_idx[r]], round(float(composite[ranked_idx[r]]), 4))
                    for r in range(min(5, n_etf))]
            print(
                f"  {ds} | top5: "
                + "  ".join(f"{tk}({sc:+.3f})" for tk, sc in top5)
                + (" [CASH]" if cash_flag else "")
            )

    # ── Latest snapshot ───────────────────────────────────────────────────────
    latest_scores_row  = score_records[-1]
    latest_ranking_row = ranking_records[-1]
    latest_date        = daily_records[-1]["date"]

    # Build latest full score dict for JSON export
    latest_scores = {}
    for i, tkr in enumerate(avail):
        score_val = latest_scores_row[tkr]
        rank_val  = latest_ranking_row[tkr]
        esn_pred_last = ensemble.predict_one(X[-1])
        latest_scores[tkr] = {
            "composite_score": round(score_val, 6),
            "esn_forecast":    round(float(esn_pred_last[i]), 6),
            "rank":            int(rank_val),
        }

    latest_ranked = sorted(
        latest_scores.items(), key=lambda x: x[1]["composite_score"], reverse=True
    )

    # ── Collect DataFrames ────────────────────────────────────────────────────
    daily_df   = pd.DataFrame(daily_records).set_index("date")
    score_df   = pd.DataFrame(score_records).set_index("date")
    ranking_df = pd.DataFrame(ranking_records).set_index("date")

    # ── In-sample diagnostics (validate period) ───────────────────────────────
    val_idx  = [i for i, d in enumerate(dates) if val_mask[i]]
    val_dates = [dates[i].strftime("%Y-%m-%d") for i in val_idx]
    val_preds_in_score = score_df.reindex(val_dates).dropna()

    print(
        f"\n  Latest ({latest_date}) top-{config.TOP_N}: "
        + "  ".join(
            f"{t}({v['composite_score']:+.3f})"
            for t, v in latest_ranked[: config.TOP_N]
        )
    )

    return {
        "latest_date":   latest_date,
        "latest_scores": latest_scores,
        "latest_ranked": latest_ranked,
        "daily_df":      daily_df,
        "score_df":      score_df,
        "ranking_df":    ranking_df,
        "universe":      universe_name,
        "n_etf":         n_etf,
        "n_features":    n_features,
    }
