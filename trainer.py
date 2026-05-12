"""trainer.py — ESN engine orchestrator: run engine + push results to HuggingFace."""

from __future__ import annotations

import io
import json
import os

from huggingface_hub import HfApi

import config
import data_manager
from engine import run_engine


def push_results(result: dict, universe: str, token: str) -> None:
    """Upload all result artefacts to the HuggingFace output dataset."""
    slug = universe.lower().replace("_", "-")
    api  = HfApi(token=token)

    api.create_repo(
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )

    # ── Summary JSON ──────────────────────────────────────────────────────────
    output = {
        "run_date":      config.TODAY,
        "universe":      universe,
        "latest_date":   result["latest_date"],
        "latest_scores": result["latest_scores"],
        "latest_ranked": [
            {"ticker": t, **v} for t, v in result["latest_ranked"]
        ],
        "config": {
            "reservoir_size":   config.RESERVOIR_SIZE,
            "spectral_radius":  config.SPECTRAL_RADIUS,
            "sparsity":         config.SPARSITY,
            "leak_rate":        config.LEAK_RATE,
            "ridge_alpha":      config.RIDGE_ALPHA,
            "n_ensembles":      config.N_ENSEMBLES,
            "input_window":     config.INPUT_WINDOW,
            "esn_forecast_wt":  config.ESN_FORECAST_WT,
            "momentum_wt":      config.MOMENTUM_WT,
            "refit_freq":       config.REFIT_FREQ,
            "refit_window":     config.REFIT_WINDOW,
            "top_n":            config.TOP_N,
        },
    }

    api.upload_file(
        path_or_fileobj=io.BytesIO(
            json.dumps(output, indent=2, default=str).encode()
        ),
        path_in_repo=f"esn_{config.TODAY}_{slug}.json",
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        commit_message=f"ESN results {config.TODAY} — {slug}",
    )

    # ── Daily summary CSV ─────────────────────────────────────────────────────
    api.upload_file(
        path_or_fileobj=io.BytesIO(result["daily_df"].to_csv().encode()),
        path_in_repo=f"daily_{slug}.csv",
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        commit_message=f"Daily summary {config.TODAY} — {slug}",
    )

    # ── Composite scores CSV (full history) ────────────────────────────────────
    api.upload_file(
        path_or_fileobj=io.BytesIO(result["score_df"].to_csv().encode()),
        path_in_repo=f"scores_{slug}.csv",
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        commit_message=f"Score history {config.TODAY} — {slug}",
    )

    # ── Rankings CSV (full history) ───────────────────────────────────────────
    api.upload_file(
        path_or_fileobj=io.BytesIO(result["ranking_df"].to_csv().encode()),
        path_in_repo=f"rankings_{slug}.csv",
        repo_id=config.HF_OUTPUT_REPO,
        repo_type="dataset",
        commit_message=f"Rankings history {config.TODAY} — {slug}",
    )

    print(f"  ✅ Pushed → {config.HF_OUTPUT_REPO}/esn_{config.TODAY}_{slug}.json")


def main() -> None:
    token = config.HF_TOKEN
    if not token:
        print("HF_TOKEN not set — aborting.")
        return

    target = os.environ.get("ESN_UNIVERSE", "ALL").upper()

    log_returns, macro_df = data_manager.load_data(token=token)

    for universe_name, tickers in config.UNIVERSES.items():
        if target != "ALL" and universe_name != target:
            continue

        result = run_engine(
            log_returns=log_returns,
            macro_df=macro_df,
            universe_tickers=tickers,
            universe_name=universe_name,
        )
        push_results(result, universe_name, token)

    print("\n✅ ESN engine complete.")


if __name__ == "__main__":
    main()
