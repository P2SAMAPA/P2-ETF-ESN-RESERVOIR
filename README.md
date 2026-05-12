# 🌊 P2-ETF-ESN-RESERVOIR

**P2Quant Engine** · Echo State Network / Reservoir Computing · ETF Ranking

[![Daily ESN Engine](https://github.com/P2SAMAPA/P2-ETF-ESN-RESERVOIR/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-ESN-RESERVOIR/actions/workflows/daily_run.yml)

---

## What Is This?

This engine applies **Reservoir Computing / Echo State Networks** to rank ETFs
by predicted 1-day return. A large fixed random recurrent network (1 000 nodes)
acts as a nonlinear temporal feature extractor. Only the linear readout layer is
trained — via ridge regression — making this one of the fastest-to-train neural
architectures in the suite.

The key insight: the random reservoir maps the input time series into a
high-dimensional, nonlinearly transformed feature space. Rich temporal dynamics
emerge from the reservoir's recurrent connections without any backpropagation
through time.

---

## Scoring Formula

```
ESN z-score  = cross_sectional_zscore(ESN_predicted_return)
MOM z-score  = cross_sectional_zscore(EWM_21d_annualised_return)

Composite    = 0.70 × ESN z-score + 0.30 × MOM z-score
```

If `top_composite_score < CASH_THRESHOLD (−0.02)` → recommend CASH.

---

## Reservoir Architecture

| Component | Detail |
|---|---|
| **Reservoir nodes** | 1 000 leaky-integrator neurons |
| **Spectral radius ρ(W)** | 0.90 — ensures echo state property (ρ < 1) |
| **Sparsity** | 5 % non-zero connections in W |
| **Input scaling** | 0.10 × W_in entries |
| **Leak rate α** | 0.30 — `x(t) = (1-α)x(t-1) + α·tanh(W_in·u + W·x(t-1))` |
| **Readout** | Ridge regression (L2 α = 1e-4) — single matrix solve |
| **Ensembles** | 5 independent reservoirs (different random seeds) → mean prediction |
| **Refit schedule** | Readout weights re-estimated every 63 trading days on rolling 504-day window |

Only **W_out** is ever trained. **W** and **W_in** are fixed at initialisation.

---

## Input Features

Each input vector u(t) is a flattened 21-day window of:
- ETF log returns (all tickers in universe)
- Macro variables: VIX, DXY, T10Y2Y, TBILL_3M

All features are z-scored over full history before entering the reservoir.

---

## Data Split (2008 → 2026 YTD)

| Period | Dates | Purpose |
|---|---|---|
| **Warm-up** | 2008-01-01 → 2012-12-31 | Reservoir transient wash-out (4 years) |
| **Train** | 2013-01-01 → 2019-12-31 | Initial readout W_out fit (7 years) |
| **Validate** | 2020-01-01 → 2021-12-31 | Hyper-parameter validation (incl. COVID crash) |
| **OOS** | 2022-01-01 → present | Live walk-forward scoring |

The 4-year warm-up ensures the reservoir's hidden state has forgotten its
zero-initialisation before any fitting occurs. The train/validate split
deliberately places COVID (2020) in the validation set — the hardest
out-of-sample test for any financial model.

---

## Universes

| Universe | Tickers |
|---|---|
| EQUITY_SECTORS | SPY QQQ XLK XLF XLE XLV XLI XLY XLP XLU GDX XME IWF XSD XBI IWM |
| COMBINED | All above + TLT VCIT LQD HYG VNQ GLD SLV |

---

## Output

Results pushed daily to HuggingFace:

- `esn_YYYY-MM-DD_{universe}.json` — latest scores, rankings, config snapshot
- `daily_{universe}.csv` — top pick, top score, CASH flag, refit events
- `scores_{universe}.csv` — full daily composite score history per ETF
- `rankings_{universe}.csv` — full daily rank history per ETF (1 = best)

**Results repo:** [P2SAMAPA/p2-etf-esn-results](https://huggingface.co/datasets/P2SAMAPA/p2-etf-esn-results)

---

## Streamlit Dashboard — 5 Tabs

1. **Rankings & Scores** — composite score bar chart, raw ESN forecast bar, top-N
   recommendation cards with colour-coded score levels
2. **Score History** — time-series of composite scores (OOS / full), score heatmap
   (last 252 days, green = buy, red = avoid)
3. **Refit Calendar** — top-score timeline with refit events marked, top-pick
   frequency bar chart (how often each ETF is ranked #1)
4. **Cross-Sectional Spread** — daily max−min spread (signal conviction), rank-1
   day-over-day stability metric, rolling 63-day average rank per ETF
5. **Full Scores Table** — all composite scores + ESN forecasts + reservoir config

---

## Why Reservoir Computing?

> *"The reservoir is a randomly generated dynamical system. The key insight is
> that any sufficiently large random recurrent network, when driven by an input
> signal, will produce rich nonlinear transformations of that signal in its
> state. Learning reduces to finding the right linear combination of those
> states."*
> — Lukosevicius & Jaeger (2009)

Advantages over deep RNNs for financial time series:
- **No vanishing gradient** — no backprop through time
- **Training in <1 second** — a single ridge regression solve
- **Principled uncertainty** — 5 independent random reservoirs → ensemble variance
- **Theoretically grounded** — echo state property guarantees input-driven dynamics

---

## References

- Jaeger, H. (2001). *The "echo state" approach to analysing and training recurrent neural networks.* GMD Report 148.
- Lukosevicius, M. & Jaeger, H. (2009). *Reservoir computing approaches to recurrent neural network training.* Computer Science Review.
- Lukosevicius, M. (2012). *A practical guide to applying echo state networks.* Neural Networks: Tricks of the Trade.
- Maass, W. et al. (2002). *Real-time computing without stable states: A new framework for neural computation.* Neural Computation.

---

*P2Quant Engine Suite · Built by P2SAMAPA*
