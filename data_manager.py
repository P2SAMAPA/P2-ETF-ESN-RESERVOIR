"""data_manager.py — Data loading and feature engineering for ESN engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import config

ETF_TICKERS = sorted(set(config.EQUITY_SECTORS_TICKERS + config.FI_COMMODITIES_TICKERS))


def load_data(token: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download master_data.parquet and return (log_returns, macro_df).

    Returns
    -------
    log_returns : DataFrame  shape (T, n_etfs)  — daily log returns
    macro_df    : DataFrame  shape (T, 4)        — VIX, DXY, T10Y2Y, TBILL_3M
    """
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=token,
        cache_dir="./hf_cache",
    )

    df = pd.read_parquet(file_path)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True).set_index("Date")

    # ── ETF log returns ───────────────────────────────────────────────────────
    available = [t for t in ETF_TICKERS if t in df.columns]
    prices     = df[available].ffill()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # ── Macro features ────────────────────────────────────────────────────────
    macro_cols = [c for c in config.MACRO_COLS if c in df.columns]
    macro_df   = df[macro_cols].reindex(log_returns.index).ffill().fillna(0.0)

    print(
        f"Loaded {len(log_returns)} rows × {len(log_returns.columns)} ETFs"
        f" | Macro: {macro_cols}"
    )
    return log_returns, macro_df


def build_input_matrix(
    log_returns: pd.DataFrame,
    macro_df: pd.DataFrame,
    tickers: list[str],
    window: int = config.INPUT_WINDOW,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Build lagged input matrix for the reservoir.

    Each row = flattened [etf_returns_{t-window+1:t}, macro_{t-window+1:t}]
    Shape: (T - window, n_features) where n_features = (n_etf + n_macro) * window

    Returns
    -------
    X     : ndarray shape (T-window, n_features)  — z-scored
    dates : DatetimeIndex aligned to X rows        — date of the *target* bar
    """
    avail   = [t for t in tickers if t in log_returns.columns]
    ret_arr = log_returns[avail].values          # (T, n_etf)
    mac_arr = macro_df.values                    # (T, n_macro)
    dates   = log_returns.index

    # Z-score features over full history to avoid look-ahead on scale
    def _zscore(arr: np.ndarray) -> np.ndarray:
        mu  = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True) + 1e-8
        return (arr - mu) / std

    ret_z = _zscore(ret_arr)
    mac_z = _zscore(mac_arr)

    combined = np.concatenate([ret_z, mac_z], axis=1)  # (T, n_etf + n_macro)

    rows, out_dates = [], []
    for t in range(window, len(combined)):
        rows.append(combined[t - window : t].ravel())   # flatten window → 1D
        out_dates.append(dates[t])

    X = np.array(rows, dtype=np.float32)                # (T-window, features)
    return X, pd.DatetimeIndex(out_dates)
