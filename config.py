"""config.py — Echo State Network (Reservoir Computing) engine configuration."""

import os
from datetime import datetime

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE   = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-esn-results"
HF_TOKEN       = os.environ.get("HF_TOKEN", None)

# ── Universes ─────────────────────────────────────────────────────────────────
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM",
]
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
COMBINED_TICKERS       = sorted(set(EQUITY_SECTORS_TICKERS + FI_COMMODITIES_TICKERS))

UNIVERSES = {
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED":       COMBINED_TICKERS,
}

# ── Macro features fed into reservoir ────────────────────────────────────────
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# ── Data split strategy (2008-2026 YTD) ──────────────────────────────────────
# Warm-up:  2008-01-01 → 2012-12-31  (4 yrs) — reservoir transient wash-out
# Train:    2013-01-01 → 2019-12-31  (7 yrs) — fit readout ridge regression
# Validate: 2020-01-01 → 2021-12-31  (2 yrs) — tune alpha & spectral radius
# Test/OOS: 2022-01-01 → 2026-YTD    (live)  — daily walk-forward scoring
WARMUP_END   = "2012-12-31"
TRAIN_END    = "2019-12-31"
VALIDATE_END = "2021-12-31"
# Everything after VALIDATE_END is treated as live OOS walk-forward

# ── Reservoir hyper-parameters ────────────────────────────────────────────────
RESERVOIR_SIZE    = 1000      # number of recurrent nodes
SPECTRAL_RADIUS   = 0.90      # rho(W) — controls echo property; < 1 = stable
SPARSITY          = 0.05      # fraction of non-zero reservoir connections
INPUT_SCALING     = 0.10      # scales input weight matrix W_in
LEAK_RATE         = 0.30      # leaky-integrator alpha: x = (1-a)*x + a*f(...)
RIDGE_ALPHA       = 1e-4      # L2 regularisation for readout regression
N_ENSEMBLES       = 5         # number of independent reservoirs (different seeds)
RANDOM_SEEDS      = [42, 137, 271, 314, 999]  # one per ensemble member

# ── Input window ─────────────────────────────────────────────────────────────
INPUT_WINDOW      = 21        # days of lagged features concatenated as input
FORECAST_HORIZON  = 1         # predict 1-day ahead log return

# ── Scoring ───────────────────────────────────────────────────────────────────
EWM_SPAN_RETURN   = 21        # EWM span for blending ESN forecast with momentum
ESN_FORECAST_WT   = 0.70      # weight on ESN predicted return
MOMENTUM_WT       = 0.30      # weight on EWM momentum signal
TOP_N             = 6         # top-N ETFs in recommendation output
CASH_THRESHOLD    = -0.02     # if top-1 ESN score < threshold → recommend CASH

# ── Walk-forward ─────────────────────────────────────────────────────────────
REFIT_FREQ        = 63        # re-fit readout weights every N trading days
REFIT_WINDOW      = 504       # rolling window of days used for each refit

# ── Output ────────────────────────────────────────────────────────────────────
TODAY = datetime.now().strftime("%Y-%m-%d")
