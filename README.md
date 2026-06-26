# Stochastic Interest Rate Models — Indian G-Sec Market

Three short-rate models calibrated to RBI benchmark G-Sec yields (June 2024 – May 2025).
Covers bond pricing, yield curve fitting, and interest rate cap/floor pricing under each model.

---

## The CIR ↔ Heston Connection

This project starts from a familiar place. The Heston stochastic volatility model uses:

```
dV = κ(θ − V) dt + σ√V dW
```

as its variance process. That **is** the CIR interest rate model — same SDE, same Feller
condition, same affine bond pricing structure. The only difference is that in Heston, V(t)
is latent (unobserved), while in CIR, r(t) is directly observable from market yields.

The Feller condition `2κθ > σ²` appears identically in both contexts: it ensures the
variance process (or the short rate) stays non-negative almost surely.

Implementing both in the same portfolio makes the mathematical continuity explicit. The
`ncx2` calibration used here for CIR is the same non-central chi-squared distribution
that underlies Heston option pricing.

---

## Models

### Vasicek (1977)
```
dr = κ(θ − r) dt + σ dW
```
Gaussian short rate. Analytically tractable; rates can go negative. Calibrated via MLE
using the exact Gaussian transition density.

### CIR (Cox-Ingersoll-Ross, 1985)
```
dr = κ(θ − r) dt + σ√r dW
```
Non-negative rates when Feller condition holds. Calibrated via MLE using the exact
non-central chi-squared transition density (`scipy.stats.ncx2`).

### Hull-White (1990, Extended Vasicek)
```
dr = (θ(t) − a·r) dt + σ dW
```
θ(t) is computed **analytically** from the initial yield curve — not optimised. This
gives an exact fit to observed market yields at t = 0 by construction:

```
θ(t) = ∂f^M(0,t)/∂t + a·f^M(0,t) + σ²(1 − e^{−2at}) / (2a)
```

`a` and `σ` are calibrated historically from rate changes. Nelson-Siegel smoothing
stabilises the forward rate derivative before computing θ(t).

---

## Calibrated Parameters (Indian G-Sec, Jun 2024 – May 2025)

| Model      | κ / a  | θ      | σ      | AIC      |
|------------|--------|--------|--------|----------|
| Vasicek    | 0.073  | 0.0100 | 0.0012 | −4075.6  |
| CIR        | 0.500  | 0.0699 | 0.0500 | −3103.1  |
| Hull-White | 0.000  | —      | 0.0012 | —        |

Vasicek achieves better AIC on this dataset. Hull-White matches the yield curve exactly
at t = 0 (< 1 bp error across all tenors by construction).

---

## Project Structure

```
models/
  vasicek.py            # SDE, Euler + exact simulation, affine bond pricing
  cir.py                # SDE, Euler + ncx2 exact simulation, Feller check
  hull_white.py         # SDE, θ(t) computation, affine bond pricing
calibration/
  vasicek_calibrate.py  # Gaussian MLE
  cir_calibrate.py      # ncx2 MLE with Feller constraint
  hull_white_calibrate.py  # historical regression + analytical θ(t)
pricing/
  bond_pricing.py       # dispatcher: bond_price(model, r, params, tau)
  yield_curve.py        # Nelson-Siegel fit and discount factor evaluation
  caps_floors.py        # Black's formula for caps, floors, swap rate
data/
  rbi_loader.py         # load/cache RBI G-Sec yield CSV
  yield_curve_builder.py  # bootstrap discount factors from par yields
  gsec_yields.csv       # 253 trading days × 6 tenors (3m/6m/1y/2y/5y/10y)
analysis/
  model_comparison.py   # yield curve fit quality, RMSE table, parameter table
  rate_paths.py         # 5-year fan charts (5th/25th/50th/75th/95th percentile)
  term_structure.py     # model vs market 2y/5y/10y yields over 12 months
tests/                  # 17 tests covering bond prices, parity, Feller, NS fit
notebooks/
  01_rates_walkthrough.ipynb  # guided walkthrough of all three models
```

---

## Setup

```bash
pip install numpy scipy pandas matplotlib jupyter nbformat
pytest tests/          # all 17 tests should pass
```

To regenerate the notebook:
```bash
python notebooks/make_notebook.py
jupyter notebook notebooks/01_rates_walkthrough.ipynb
```

To run the analysis scripts:
```bash
python -m analysis.model_comparison
python -m analysis.rate_paths
python -m analysis.term_structure
```

---

## Data

RBI benchmark G-Sec yields sourced from the DBIE portal. The committed CSV
(`data/gsec_yields.csv`) covers June 2024 – May 2025 at six tenors: 3m, 6m, 1y, 2y,
5y, 10y. The 10-year yield series serves as the short-rate proxy for Vasicek and CIR
calibration.

---

## Validation Gates

- Flat yield curve → `P(0,t) = exp(−0.065·t)` within numerical precision
- `vasicek_bond_price(r=0.05, κ=0.5, θ=0.05, σ=0.02, τ=5)` ≈ 0.7788
- CIR bond prices converge to Vasicek as σ → 0
- All CIR simulated paths ≥ 0 when Feller condition holds
- Hull-White model yields match market within 1 bp for all tenors
- Cap-floor parity: `Cap(K*) − Floor(K*)` = 0 at the par swap rate
