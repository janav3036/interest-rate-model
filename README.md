# Stochastic Interest Rate Models — Indian G-Sec Market

Three short-rate models (Vasicek, CIR, Hull-White) calibrated to RBI benchmark G-Sec yields (June 2024 – May 2025), with analytical bond pricing, MLE calibration, Nelson-Siegel yield curve fitting, and interest rate cap/floor pricing.

---

## The CIR ↔ Heston Connection

The Heston stochastic volatility model uses:

$$dV = \kappa(\theta - V)\,dt + \sigma\sqrt{V}\,dW$$

as its variance process. That **is** the CIR interest rate model — same SDE, same affine structure, same Feller condition. The only difference is that in Heston, $V(t)$ is latent, while in CIR, $r(t)$ is directly observable from market yields.

The Feller condition $2\kappa\theta > \sigma^2$ appears identically in both contexts. Implementing CIR here, after Heston, makes that mathematical continuity explicit: the `ncx2` MLE used to calibrate CIR is the same non-central chi-squared distribution underlying Heston option pricing.

---

## Models

### Vasicek (1977)
$$dr = \kappa(\theta - r)\,dt + \sigma\,dW$$
Gaussian short rate with mean reversion. Analytically tractable; negative rates are possible. Calibrated via MLE using the exact Gaussian transition density.

### CIR — Cox-Ingersoll-Ross (1985)
$$dr = \kappa(\theta - r)\,dt + \sigma\sqrt{r}\,dW$$
Non-negative rates when the Feller condition $2\kappa\theta > \sigma^2$ holds. Calibrated via MLE using the exact non-central chi-squared transition density (`scipy.stats.ncx2`).

### Hull-White (1990)
$$dr = (\theta(t) - a r)\,dt + \sigma\,dW$$
$\theta(t)$ is computed **analytically** from the initial yield curve — not optimised:

$$\theta(t) = \frac{\partial f^M(0,t)}{\partial t} + a f^M(0,t) + \frac{\sigma^2\left(1 - e^{-2at}\right)}{2a}$$

This gives Hull-White an exact fit to observed market yields at $t=0$ by construction. Nelson-Siegel smoothing stabilises the forward rate derivative before computing $\theta(t)$; raw linear interpolation introduces kinks that corrupt the result.

---

## Results

### Calibrated Parameters (Indian G-Sec, Jun 2024 – May 2025)

| Model      | $\kappa$ / $a$ | $\theta$ | $\sigma$ | AIC |
|------------|--------|--------|--------|----------|
| Vasicek    | 0.073  | 0.0100 | 0.0012 | −4075.6  |
| CIR        | 0.500  | 0.0699 | 0.0500 | −3103.1  |
| Hull-White | 0.000  | —      | 0.0012 | —        |

Vasicek achieves better AIC on this dataset, consistent with the relatively smooth, mean-reverting Indian rate environment over this period. Hull-White matches market yields within 1 bp at all tenors by construction.

### Yield Curve Fit (single-date snapshot)

Hull-White fits exactly; Vasicek and CIR fit approximately. RMSE comparison is in `analysis/model_comparison.py`.

### Simulated Rate Paths (5-year fan chart)

Vasicek paths cross zero (demonstrating the model's key limitation at low-rate regimes); CIR paths stay non-negative. All three models simulated at 5th / 25th / 50th / 75th / 95th percentiles via `analysis/rate_paths.py`.

---

## Project Structure

```
models/
  vasicek.py              SDE, Euler + exact simulation, affine bond pricing
  cir.py                  SDE, Euler + ncx2 exact simulation, Feller check
  hull_white.py           SDE, θ(t) computation, affine bond pricing

calibration/
  vasicek_calibrate.py    Gaussian MLE (L-BFGS-B)
  cir_calibrate.py        ncx2 MLE with Feller constraint
  hull_white_calibrate.py historical regression for a/σ, analytical θ(t)

pricing/
  bond_pricing.py         dispatcher: bond_price(model, r, params, tau)
  yield_curve.py          Nelson-Siegel fit and discount factor evaluation
  caps_floors.py          Black's formula for caps, floors, swap rate

data/
  rbi_loader.py           load/cache RBI G-Sec yield CSV
  yield_curve_builder.py  bootstrap discount factors from par yields
  gsec_yields.csv         253 trading days × 6 tenors (3m/6m/1y/2y/5y/10y)

analysis/
  model_comparison.py     yield curve fit quality, RMSE table, parameter table
  rate_paths.py           5-year fan charts (5th/25th/50th/75th/95th percentile)
  term_structure.py       model vs market 2y/5y/10y yields over 12 months

tests/                    17 tests covering bond prices, parity, Feller, NS fit

notebooks/
  01_rates_walkthrough.ipynb    guided walkthrough of all three models
```

---

## Setup

```bash
pip install -r requirements.txt

# Run all 17 tests
pytest tests/

# Run a single test file
pytest tests/test_bond_pricing.py -v

# Run analysis scripts
python -m analysis.model_comparison
python -m analysis.rate_paths
python -m analysis.term_structure

# Launch notebook
jupyter notebook notebooks/01_rates_walkthrough.ipynb
```

To regenerate the notebook from source:
```bash
python notebooks/make_notebook.py
```

---

## Data

RBI benchmark G-Sec yields sourced from the DBIE portal. The committed CSV (`data/gsec_yields.csv`) covers June 2024 – May 2025 at six tenors: 3m, 6m, 1y, 2y, 5y, 10y. The 10-year yield series is used as the short-rate proxy for Vasicek and CIR calibration.

---

## Validation Gates

- Flat yield curve → $P(0,t) = e^{-0.065 t}$ within numerical precision
- `vasicek_bond_price(r=0.05, κ=0.5, θ=0.05, σ=0.02, τ=5)` $\approx 0.7788$
- CIR bond prices converge to Vasicek as $\sigma \to 0$
- All CIR simulated paths $\geq 0$ when Feller condition holds
- Hull-White model yields match market within 1 bp at all tenors
- Cap-floor parity: $\text{Cap}(K^*) - \text{Floor}(K^*) = 0$ at the par swap rate
