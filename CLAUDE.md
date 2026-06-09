# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project implementing three stochastic interest rate models (Vasicek, CIR, Hull-White) calibrated to Indian G-Sec market data. The full specification is in [project_4_interest_rates.md](project_4_interest_rates.md).

## Setup and Commands

```bash
# Install dependencies
pip install numpy scipy pandas matplotlib jupyter

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_bond_pricing.py -v

# Run a single test function
pytest tests/test_bond_pricing.py::test_vasicek_bond_price -v

# Launch Jupyter notebook
jupyter notebook notebooks/01_rates_walkthrough.ipynb
```

## Architecture

### Module Responsibilities

- **`data/`** — Market data ingestion and yield curve bootstrapping. `rbi_loader.py` fetches/caches RBI G-Sec yields; `yield_curve_builder.py` converts par yields to zero-coupon discount factors P^M(0,t) and computes instantaneous forward rates via finite differences.

- **`models/`** — SDE implementations with both Euler and exact simulation for each model. Each file is self-contained: analytical bond price formulas, yield computation, and simulation live together. Hull-White additionally takes a precomputed `theta_grid` array (not a free parameter — it is derived analytically from the yield curve).

- **`calibration/`** — MLE calibration using each model's exact transition density: Gaussian for Vasicek, non-central chi-squared (`scipy.stats.ncx2`) for CIR. Hull-White calibrates `a` and `sigma` historically via regression, then computes `theta(t)` analytically — it does NOT optimize theta.

- **`pricing/`** — `bond_pricing.py` is a thin dispatcher to model-specific functions. `yield_curve.py` holds the Nelson-Siegel smoother used to stabilize forward rate computation (important for Hull-White theta). `caps_floors.py` prices interest rate caps/floors via Black's formula.

- **`analysis/`** — Visualization and empirical comparison scripts, not imported by other modules.

### Key Mathematical Relationships

**CIR ↔ Heston connection:** The CIR SDE `dr = κ(θ-r)dt + σ√r dW` is identical to Heston's variance process. The Feller condition `2κθ > σ²` appears in both contexts and must be enforced as a hard constraint during CIR calibration.

**Affine bond pricing:** All three models have the form `P(t,T) = A(t,T)·exp(-B(t,T)·r(t))`. The A and B functions differ per model — implement exactly as written in the spec, as sign errors are common in the CIR formula.

**Hull-White exact fit:** `theta(t)` is computed from the formula:
```
θ(t) = ∂f^M(0,t)/∂t + a·f^M(0,t) + σ²(1-e^(-2at))/(2a)
```
This requires stable forward rate derivatives. Use Nelson-Siegel (not raw linear interpolation) to compute `f^M(0,t)` before differencing, or numerical noise will corrupt `theta(t)`.

### Data Flow

```
RBI G-Sec yields (CSV/API)
  → yield_curve_builder: bootstrap P^M(0,t)
    → Nelson-Siegel smooth fit
      → forward rates f^M(0,t)
        → Hull-White: compute theta(t) analytically
        → Vasicek/CIR: MLE calibration on 10y yield series as short-rate proxy
          → bond_pricing dispatcher
          → caps_floors (Black's formula, uses P^M(0,t) directly)
```

### Validation Gates (in build order)

1. Flat yield curve → `P^M(0,t) = exp(-0.065·t)` within numerical precision
2. `vasicek_bond_price(r=0.05, κ=0.5, θ=0.05, σ=0.02, τ=5)` ≈ 0.7788
3. CIR bond price converges to Vasicek at σ→0 (use σ=0.001)
4. CIR simulated paths stay ≥ 0 when Feller condition holds
5. Hull-White model yields match market yields within 1 basis point for all tenors
6. Cap-floor parity: `cap(K*) - floor(K*)` = 0 at the par swap rate

## Data Notes

RBI G-Sec data source: DBIE portal or RBI Bulletin. If programmatic access is unreliable, commit a 12-month CSV of 10 benchmark tenor yields to `data/` and load via `load_cached_yields()`. The 10-year G-Sec yield series serves as the short-rate proxy for Vasicek and CIR calibration.
