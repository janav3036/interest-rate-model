# Build Phases

Seven sessions to complete the full pipeline. Each session has a clear entry point,
defined files to produce, and validation gates that must pass before moving on.

---

## Phase 1 — Data Pipeline
**Goal:** Get real market data into memory as clean discount factors and forward rates.

**Files to build:**
- `data/rbi_loader.py` — download or load cached RBI G-Sec yields
- `data/yield_curve_builder.py` — bootstrap discount factors, interpolate, compute forward rates

**Exit criteria:**
- `load_cached_yields()` returns a DataFrame with columns `date, tenor_3m, tenor_6m, tenor_1y, tenor_2y, tenor_5y, tenor_10y` (yields as decimals)
- Flat yield curve test: all tenors at 6.5% → `P^M(0,t) == exp(-0.065 * t)` within 1e-6
- `compute_forward_rate()` returns 0.065 everywhere on the flat curve

**Notes:**
- If RBI programmatic access fails, commit a hand-encoded CSV (`data/gsec_yields.csv`) with ~12 months of 10-tenor data and load that. Document the fallback in the file header.
- Keep `rbi_loader.py` and `yield_curve_builder.py` independent — loader returns raw yields, builder does all the math.

---

## Phase 2 — Vasicek Model
**Goal:** Working analytical bond pricer and both simulation methods for Vasicek.

**Files to build:**
- `models/vasicek.py` — bond price, yield, Euler simulation, exact simulation

**Exit criteria:**
- `vasicek_bond_price(r=0.05, kappa=0.5, theta=0.05, sigma=0.02, tau=5.0)` ≈ 0.7788 (±0.001)
- Simulated yield curve (exact simulation, 10k paths) matches analytical yield curve within 5 bps at all tenors
- Euler and exact simulations produce visually identical fan charts (confirm with a quick plot)

**Notes:**
- Implement exact simulation first — it's strictly better and serves as the reference.
- Euler is needed for comparison and as a pattern for CIR/Hull-White later.

---

## Phase 3 — CIR Model
**Goal:** Non-negative short rate model with exact ncx2 simulation.

**Files to build:**
- `models/cir.py` — Feller check, bond price, yield, truncated Euler, exact simulation via `scipy.stats.ncx2`

**Exit criteria:**
- `cir_bond_price` converges to `vasicek_bond_price` at σ=0.001 (relative error < 0.1%)
- `simulate_cir_euler` with Feller-satisfied params: `np.all(paths >= 0)` across 1000 paths × 1000 steps
- A(t,T) and B(t,T) formulas produce values that round-trip: `R(t,T) = -ln(P) / tau` gives back a sensible yield

**Notes:**
- The CIR affine formula is the most error-prone part. Implement B(t,T) first, validate it independently, then A(t,T).
- γ = sqrt(κ² + 2σ²) — reuse this scalar in both A and B to avoid inconsistency.
- Connect to Heston in a comment: the only difference is V(t) is latent in Heston, observable here.

---

## Phase 4 — Vasicek & CIR Calibration
**Goal:** MLE calibration of both models to the Indian G-Sec 10y yield series.

**Files to build:**
- `calibration/vasicek_calibrate.py` — Gaussian MLE, L-BFGS-B
- `calibration/cir_calibrate.py` — ncx2 MLE with Feller penalty

**Exit criteria:**
- Both calibrators return `{kappa, theta, sigma, log_likelihood, aic}` dicts
- `calibrate_vasicek` on synthetic data generated from known params recovers those params within 10%
- `calibrate_cir` enforces Feller: if `2κθ ≤ σ²` at convergence, raise or retry
- Run both on the actual 10y G-Sec series; log calibrated params and AIC side-by-side

**Notes:**
- Use dt = 1/252 (daily business day convention).
- Vasicek log-likelihood uses exact Gaussian transition — no approximation needed.
- CIR: the ncx2 parameterization uses `c`, `df`, `λ` as defined in the spec. Precompute `c` outside the inner loop.

---

## Phase 5 — Nelson-Siegel + Hull-White
**Goal:** Smooth yield curve fit and the exact-fit Hull-White model.

**Files to build:**
- `pricing/yield_curve.py` — Nelson-Siegel fit and evaluation
- `models/hull_white.py` — `compute_theta_t`, bond price, simulation
- `calibration/hull_white_calibrate.py` — historical regression for `a`/`σ`, analytical θ(t)

**Exit criteria:**
- Nelson-Siegel RMSE < 5 bps on the Indian G-Sec yield curve snapshot
- `compute_theta_t` on the Nelson-Siegel-smoothed curve produces a smooth (no spiky) θ(t) — plot it
- Hull-White model yields match market yields within **1 basis point** at all observed tenors (`validate_yield_curve_fit`)
- If the 1 bps gate fails, the bug is in `compute_theta_t` (finite-difference noise) — switch to Nelson-Siegel-based analytic derivatives

**Notes:**
- Do NOT use raw linear interpolation to compute forward rates for θ(t). Linear interpolation introduces kinks that blow up when differenced. Nelson-Siegel first, then differentiate analytically or with fine finite differences.
- Hull-White calibration: regress `Δr` on `r` for `a` and `σ`; then compute θ(t) analytically. No optimization loop needed for θ.

---

## Phase 6 — Bond Pricing Dispatcher + Caps/Floors
**Goal:** Unified pricing interface and interest rate derivatives pricing.

**Files to build:**
- `pricing/bond_pricing.py` — dispatcher for all three models + coupon bond pricer
- `pricing/caps_floors.py` — Black's formula for caps/floors, swap rate
- `tests/test_bond_pricing.py` — full suite
- `tests/test_yield_curve.py`
- `tests/test_cap_floor_parity.py`
- `tests/test_feller.py`

**Exit criteria:**
- `pytest tests/` passes all four test files
- Cap-floor parity: `cap(K*) - floor(K*)` < 1e-6 at the par swap rate K*
- `coupon_bond_price` equals the sum of zero-coupon bond prices for each cash flow (by construction — easy to verify)

**Notes:**
- `caps_floors.py` takes a `discount_factors_fn: Callable[[float], float]` — keeps it decoupled from the yield curve implementation.
- Demonstrate cap/floor pricing at three strikes: ATM (K = swap rate), ATM+100bps, ATM-100bps.

---

## Phase 7 — Analysis, Notebook & Polish
**Goal:** All deliverables ready for the portfolio.

**Files to build:**
- `analysis/model_comparison.py` — yield curve fit plot + RMSE table + calibrated params table
- `analysis/rate_paths.py` — fan chart (5th/25th/50th/75th/95th percentiles, all three models)
- `analysis/term_structure.py` — time-series of model vs market 2y/5y/10y yields
- `notebooks/01_rates_walkthrough.ipynb` — guided narrative walkthrough
- `README.md` — CIR↔Heston connection, data sourcing, model comparison summary

**Exit criteria:**
- `pytest tests/` still passes (no regressions from analysis code)
- Fan chart shows Vasicek paths crossing zero (demonstrating the weakness) while CIR stays non-negative
- Hull-White RMSE is visibly near-zero vs Vasicek/CIR RMSE on the yield curve fit plot
- Notebook runs top-to-bottom without errors (`jupyter nbconvert --to notebook --execute`)
- README explains why CIR appears in two projects (this + Heston's variance process)

---

## Phase Dependency Map

```
Phase 1 (Data)
    ↓
Phase 2 (Vasicek)  →  Phase 4 (Calibration)
Phase 3 (CIR)      →  Phase 4
    ↓
Phase 5 (Nelson-Siegel + Hull-White)
    ↓
Phase 6 (Pricing + Tests)
    ↓
Phase 7 (Analysis + Polish)
```

Phases 2 and 3 can be built in parallel once Phase 1 is done.
Phase 4 requires both Phase 2 and 3.
Phase 5 requires Phase 1 (for market data) and Phase 3 (for CIR intuition, not code).
