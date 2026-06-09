# Project 4 — Stochastic Interest Rate Models & Fixed Income Pricing

## Strategic Context

This project fills the fixed income gap in the portfolio. Projects 1–3 cover equity
derivatives exclusively. CMU MSCF and Columbia IEOR both have heavy rates content in
their curricula; having zero fixed income work before applying is a visible hole.

The mathematical entry point is deliberate: the CIR process is already familiar from
Heston's variance dynamics. Project 1 uses `dV = κ(θ-V)dt + σ√V dW` as its variance
process. That is the CIR model. This project starts there, prices bonds under it, then
climbs the ladder: Vasicek (simpler, Gaussian) → CIR (non-negative rates) → Hull-White
(exact yield curve fit). The progression mirrors how rates modeling evolved historically
and demonstrates that the learning is genuine, not surface-level.

**What this demonstrates to adcoms:** command of term structure theory, ability to
calibrate to real market data (Indian G-Sec yield curve), and mathematical continuity
across the portfolio (CIR appears in two projects for two different purposes).

**Repository name:** `interest-rate-models`

---

## Mathematical Foundations

### 1. Vasicek Model (1977)

```
dr(t) = κ(θ - r(t))·dt + σ·dW(t)
```

Parameters:
- `kappa (κ)` — mean reversion speed (κ > 0)
- `theta (θ)` — long-run mean rate
- `sigma (σ)` — instantaneous volatility
- `r0`         — current short rate

The short rate is Gaussian under Vasicek. Exact conditional distribution:

```
r(t) | r(0) ~ Normal( r(0)·e^(-κt) + θ·(1 - e^(-κt)),
                      σ²·(1 - e^(-2κt)) / (2κ) )
```

**Main weakness:** rates can go negative. Vasicek is analytically tractable but
unrealistic for high-inflation environments. In the current Indian context (repo rate ~6.5%),
Vasicek can still produce useful results but negative rate risk should be quantified.

**Zero-coupon bond pricing (analytical — affine term structure):**

```
P(t, T) = A(t,T) · exp(-B(t,T) · r(t))
```

where:
```
B(t,T) = (1 - e^(-κ(T-t))) / κ

A(t,T) = exp( [(B(t,T) - (T-t)) · (κ²θ - σ²/2)] / κ²
              - σ²·B(t,T)² / (4κ) )
```

**Continuously compounded yield:**
```
R(t,T) = -ln(P(t,T)) / (T-t)
```

**Simulation (Euler-Maruyama):**
```
r(t+Δt) = r(t) + κ(θ - r(t))·Δt + σ·√Δt·Z,  Z ~ N(0,1)
```

Since the conditional distribution is exact and Gaussian, you can also simulate exactly
(exact simulation, not just Euler) by drawing directly from the conditional normal.
Implement both methods and compare — exact simulation is strictly better for large Δt.

---

### 2. CIR Model (Cox-Ingersoll-Ross, 1985)

```
dr(t) = κ(θ - r(t))·dt + σ·√r(t)·dW(t)
```

Parameters: same as Vasicek (κ, θ, σ, r0).

**Feller condition:** `2κθ > σ²` ensures r(t) > 0 almost surely. Enforce this as a hard
constraint during calibration (identical to the Feller condition in Heston — the connection
should be made explicit in the README).

The variance process in Heston's model IS the CIR model. The only difference is that in
Heston, V(t) is latent (unobserved), while here r(t) is directly observable.

**Zero-coupon bond pricing (analytical — affine term structure):**

```
P(t, T) = A(t,T) · exp(-B(t,T) · r(t))
```

where:
```
γ = √(κ² + 2σ²)

B(t,T) = 2(e^(γ(T-t)) - 1) / [(γ + κ)(e^(γ(T-t)) - 1) + 2γ]

A(t,T) = [ 2γ · exp((κ+γ)(T-t)/2) / ((γ+κ)(e^(γ(T-t))-1) + 2γ) ]^(2κθ/σ²)
```

**Conditional distribution:** non-central chi-squared. Given r(s):
```
r(t) = (σ²(1 - e^(-κ(t-s)))) / (4κ) · χ²(df, λ)
```
where:
```
df = 4κθ/σ²           (degrees of freedom)
λ  = 4κ·e^(-κ(t-s))·r(s) / (σ²(1 - e^(-κ(t-s))))  (non-centrality)
```

For simulation, use the exact transition distribution via `scipy.stats.ncx2` for
large Δt. For small Δt (intraday or fine grids), truncated Euler suffices:
```
r(t+Δt) = max(0, r(t) + κ(θ - r(t))·Δt + σ·√r(t)·√Δt·Z)
```

---

### 3. Hull-White Model (1990, Extended Vasicek)

```
dr(t) = (θ(t) - a·r(t))·dt + σ·dW(t)
```

Parameters:
- `a` — mean reversion speed (constant)
- `σ` — volatility (constant)
- `θ(t)` — time-dependent drift, chosen to fit the initial yield curve exactly

Key insight: θ(t) is not a free parameter to be calibrated by optimization. It is
**determined analytically** from the initial yield curve. Once a and σ are fixed
(from cap/floor volatilities), θ(t) is given by:

```
θ(t) = ∂f^M(0,t)/∂t + a·f^M(0,t) + σ²(1 - e^(-2at)) / (2a)
```

where `f^M(0,t)` is the instantaneous forward rate observed in the market:
```
f^M(0,t) = -∂ln(P^M(0,t)) / ∂t
```

and `P^M(0,t)` is the market discount factor interpolated from observed yields.

This means Hull-White is the only model of the three that **exactly** reproduces the
observed yield curve at t=0. Vasicek and CIR will generally fit only approximately.

**Bond pricing (Hull-White):**
```
P(t,T) = A(t,T) · exp(-B(t,T) · r(t))
```

where:
```
B(t,T) = (1 - e^(-a(T-t))) / a

ln A(t,T) = ln(P^M(0,T) / P^M(0,t))
             + B(t,T) · f^M(0,t)
             - σ² · (e^(-at) - e^(-aT))² · (e^(2at) - 1) / (4a³)
```

Note: P^M(0,T) and P^M(0,t) are market discount factors from the fitted yield curve,
not model outputs. This is what makes Hull-White exact.

**Simulation (Euler on r(t) directly):**
```
r(t+Δt) = r(t) + (θ(t) - a·r(t))·Δt + σ·√Δt·Z
```

Requires computing θ(t) on a fine grid first (precompute from yield curve).

**Calibration strategy:**
1. a and σ: in a full implementation, these come from fitting to cap/floor implied vols.
   If cap vol data is unavailable, fit a and σ to minimize the error in simulated yield
   curve dynamics against historical rate changes (historical calibration).
2. θ(t): computed analytically from (a, σ, initial yield curve). No optimization needed.

For the project scope: calibrate a and σ via MLE on historical rate data (same as
Vasicek/CIR), then compute θ(t) analytically. Document this simplification honestly.

---

### 4. Term Structure Notation

A few conventions used throughout:

- `P(t,T)` — price at time t of a zero-coupon bond paying 1 at time T
- `R(t,T)` — continuously compounded yield: `R = -ln(P)/(T-t)`
- `f(t,T)` — instantaneous forward rate: `f = -∂ln(P)/∂T`
- `f^M(0,t)` — market instantaneous forward rate (derived from observed yield curve)
- `P^M(0,T)` — market discount factor (bootstrapped from observed yields)

---

### 5. Cap and Floor Pricing (Black's Model)

A **cap** is a strip of caplets. A caplet with reset date `t_i` and payment date `t_{i+1}`
pays `max(L(t_i) - K, 0) · δ` at `t_{i+1}`, where L(t_i) is the LIBOR/MIBOR setting
and δ = t_{i+1} - t_i (year fraction).

Under Black's market model, each caplet price is:
```
Caplet(t_i) = δ · P^M(0, t_{i+1}) · [F_i · N(d₁) - K · N(d₂)]
```

where:
```
F_i = (P^M(0,t_i) / P^M(0,t_{i+1}) - 1) / δ    (forward rate for period [t_i, t_{i+1}])
d₁  = [ln(F_i/K) + σ_i²·t_i/2] / (σ_i·√t_i)
d₂  = d₁ - σ_i·√t_i
σ_i = flat vol for this caplet (from cap vol surface or assumed flat)
```

Cap price = sum of all caplet prices.

A **floor** is a strip of floorlets:
```
Floorlet(t_i) = δ · P^M(0, t_{i+1}) · [K · N(-d₂) - F_i · N(-d₁)]
```

**Cap-floor parity (validation):**
```
Cap - Floor = Swap(payer)
```
A payer swap pays fixed K and receives floating. This is the key validation for the
cap/floor pricer.

For simplicity in the project, implement caps and floors using a **flat vol** input
(single σ across all caplets). Indian MIBOR cap/floor data is sparse; if not available,
demonstrate the pricing machinery with illustrative vol levels (10%, 15%, 20%) and
document the data limitation honestly.

---

### 6. Yield Curve Construction from Indian Market Data

**Data source:** RBI Developmental Research Group publishes benchmark G-Sec yields daily.

Primary URL: `https://www.rbi.org.in/Scripts/BS_ViewBulletin.aspx`
DBIE portal: `https://dbie.rbi.org.in/DBIE/dbie.rbi?site=statistics`
Direct API (if accessible): `https://api.rbi.org.in` (check availability)
Alternative: FBIL (Financial Benchmarks India Ltd) publishes the FBIL-GSEC yield curve
daily in PDF format. A fallback is to manually collect 10 benchmark tenor yields from
RBI's published tables.

**Available tenors (typical):** 91d, 182d, 364d (T-bills), 2y, 3y, 4y, 5y, 6y, 7y, 8y,
9y, 10y, 14y, 30y G-Sec yields.

**Bootstrapping (simplified):**
For the purposes of this project, treat all observed yields as par yields and bootstrap
zero rates. Full coupon stripping is not required. Use linear interpolation between
observed tenors for `P^M(0,t)` at arbitrary t.

If the RBI data pipeline proves unreliable, fall back to NSE's bond futures prices or
manually encoded a single representative yield curve snapshot with documentation.

---

## Project Structure

```
interest-rate-models/
├── models/
│   ├── vasicek.py               # Vasicek SDE, simulation, bond pricing
│   ├── cir.py                   # CIR SDE, simulation, bond pricing, Feller check
│   └── hull_white.py            # Hull-White SDE, theta(t) computation, simulation
├── pricing/
│   ├── bond_pricing.py          # ZCB pricing for all three models
│   ├── yield_curve.py           # yield curve bootstrap + interpolation
│   └── caps_floors.py           # cap/floor pricing via Black's formula
├── calibration/
│   ├── vasicek_calibrate.py     # MLE on historical rate series
│   ├── cir_calibrate.py         # MLE using ncx2 density + Feller constraint
│   └── hull_white_calibrate.py  # fit a, sigma (historical); compute theta(t) exactly
├── data/
│   ├── rbi_loader.py            # download / parse RBI G-Sec yield data
│   └── yield_curve_builder.py   # bootstrap discount factors from par yields
├── analysis/
│   ├── model_comparison.py      # Vasicek vs CIR vs HW: yield curve fit quality
│   ├── rate_paths.py            # simulated rate path visualizations
│   └── term_structure.py        # model vs market yield curves across time
├── tests/
│   ├── test_bond_pricing.py     # analytical bond price validation
│   ├── test_yield_curve.py      # P(t,T) vs bootstrapped market prices
│   ├── test_cap_floor_parity.py # cap - floor = swap parity
│   └── test_feller.py           # CIR stays non-negative when Feller satisfied
├── notebooks/
│   └── 01_rates_walkthrough.ipynb
└── README.md
```

---

## Build Order

Build strictly in this sequence. Each step has a validation gate before proceeding.

### Step 1 — `data/rbi_loader.py`

```python
def download_rbi_yields(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads RBI benchmark G-Sec yields.
    Returns DataFrame with columns: date, tenor_3m, tenor_6m, tenor_1y,
    tenor_2y, tenor_5y, tenor_10y, (+ others as available).
    All yields as decimals (e.g. 0.065 for 6.5%).
    """

def load_cached_yields(filepath: str) -> pd.DataFrame:
    """Load from local CSV/Parquet cache."""
```

Strategy: attempt programmatic download first. If the RBI API/page structure makes
scraping unreliable, hardcode the fallback: manually encode a 12-month historical
series of benchmark G-Sec yields (10 tenors × ~250 dates) as a CSV committed to the
`data/` directory. Document clearly in the README.

---

### Step 2 — `data/yield_curve_builder.py`

```python
def bootstrap_discount_factors(yields: pd.Series, tenors: list[float]) -> pd.Series:
    """
    Input: par yields at observed tenors.
    Output: zero-coupon discount factors P^M(0,t) at the same tenors.
    Uses simplified bootstrap (treat par yields as zero rates for T <= 1y;
    strip coupon bonds for T > 1y).
    """

def interpolate_discount_factor(tenors: np.ndarray,
                                  discount_factors: np.ndarray,
                                  t: float) -> float:
    """Linear interpolation of P^M(0,t) at arbitrary t."""

def compute_forward_rate(tenors: np.ndarray,
                          discount_factors: np.ndarray,
                          t: float) -> float:
    """
    Instantaneous forward rate f^M(0,t) = -d/dt ln(P^M(0,t)).
    Use finite differences: f^M(0,t) ≈ -(ln(P^M(0,t+dt)) - ln(P^M(0,t))) / dt
    dt = 1/365 (1 day).
    """
```

Validation gate: for a flat yield curve (all tenors yield 6.5%), discount factors
must satisfy `P^M(0,t) = exp(-0.065·t)` within numerical precision.

---

### Step 3 — `models/vasicek.py`

```python
def vasicek_bond_price(r: float, kappa: float, theta: float,
                        sigma: float, tau: float) -> float:
    """
    Analytical ZCB price under Vasicek.
    tau = T - t (time to maturity).
    Returns P(t,T) using affine formula.
    """

def vasicek_yield(r: float, kappa: float, theta: float,
                   sigma: float, tau: float) -> float:
    """R(t,T) = -ln(P(t,T)) / tau"""

def simulate_vasicek_euler(r0: float, kappa: float, theta: float,
                            sigma: float, T: float,
                            N_paths: int, N_steps: int,
                            seed: int = None) -> np.ndarray:
    """Returns paths of shape (N_paths, N_steps+1)."""

def simulate_vasicek_exact(r0: float, kappa: float, theta: float,
                            sigma: float, T: float,
                            N_paths: int, N_steps: int,
                            seed: int = None) -> np.ndarray:
    """
    Exact simulation via conditional Gaussian distribution.
    For each step, draw from Normal(mean(r,dt), var(r,dt)) directly.
    Preferred over Euler for large dt.
    """
```

Validation gate: `vasicek_bond_price` with κ=0.5, θ=0.05, σ=0.02, r0=0.05, τ=5.0
should return ≈ 0.7788 (verify against textbook or numerical integration).
Simulated yield curve from exact simulation must converge to analytical yield curve
as N_paths increases.

---

### Step 4 — `models/cir.py`

```python
def feller_condition_satisfied(kappa: float, theta: float, sigma: float) -> bool:
    """Returns True if 2*kappa*theta > sigma^2."""

def cir_bond_price(r: float, kappa: float, theta: float,
                    sigma: float, tau: float) -> float:
    """Analytical ZCB price under CIR using affine formula."""

def cir_yield(r: float, kappa: float, theta: float,
               sigma: float, tau: float) -> float:

def simulate_cir_euler(r0: float, kappa: float, theta: float,
                        sigma: float, T: float,
                        N_paths: int, N_steps: int,
                        seed: int = None) -> np.ndarray:
    """Truncated Euler: r = max(0, r + kappa*(theta-r)*dt + sigma*sqrt(r)*sqrt(dt)*Z)"""

def simulate_cir_exact(r0: float, kappa: float, theta: float,
                        sigma: float, T: float,
                        N_paths: int, N_steps: int,
                        seed: int = None) -> np.ndarray:
    """
    Exact simulation via non-central chi-squared distribution.
    Uses scipy.stats.ncx2.rvs() for each time step.
    Slower but exact — use for validation.
    """
```

**CIR gamma parameter:**
```
gamma = sqrt(kappa^2 + 2*sigma^2)
```

Implement the B(t,T) and A(t,T) formulas exactly as written in the math foundations section.
Double-check the exponents — the CIR affine formula is easy to implement with sign errors.

Validation gate: at σ → 0 limit (say σ=0.001), CIR bond prices must converge to
Vasicek bond prices (same κ, θ). Also verify: when Feller condition is met, simulated
rates must stay non-negative across all paths (assert `np.all(paths >= 0)`).

---

### Step 5 — `pricing/bond_pricing.py`

Thin wrapper that dispatches to model-specific functions:

```python
def bond_price(model: str, r: float, params: dict, tau: float) -> float:
    """
    model: 'vasicek' | 'cir' | 'hull_white'
    params: dict of model parameters
    tau: time to maturity
    """

def yield_curve(model: str, r: float, params: dict,
                 tenors: np.ndarray) -> np.ndarray:
    """Returns array of yields for a range of tenors."""

def coupon_bond_price(model: str, r: float, params: dict,
                       coupon_rate: float, face: float,
                       payment_schedule: list[float]) -> float:
    """
    Price a coupon bond as sum of discounted cash flows.
    payment_schedule: list of times to each coupon payment.
    Last payment includes face value.
    """
```

---

### Step 6 — `calibration/vasicek_calibrate.py`

Maximum likelihood estimation using the exact conditional Gaussian distribution.

Given observed rate time series r_0, r_1, ..., r_N at equal intervals Δt:

The log-likelihood is:
```
ln L = sum_{i=1}^{N} ln φ(r_i | r_{i-1}; κ, θ, σ)
```

where φ is the normal density with:
```
μ_i  = r_{i-1}·e^(-κΔt) + θ(1 - e^(-κΔt))
σ_i² = σ²(1 - e^(-2κΔt)) / (2κ)
```

```python
def vasicek_log_likelihood(params: tuple, rate_series: np.ndarray,
                            dt: float) -> float:
    """
    params = (kappa, theta, sigma)
    Returns negative log-likelihood (for minimization).
    """

def calibrate_vasicek(rate_series: np.ndarray, dt: float,
                       initial_guess: tuple = (1.0, 0.065, 0.01)
                       ) -> dict:
    """
    Runs scipy.optimize.minimize (L-BFGS-B) on negative log-likelihood.
    Bounds: kappa in (0.01, 10), theta in (0.001, 0.3), sigma in (0.001, 0.2)
    Returns dict with keys: kappa, theta, sigma, log_likelihood, aic
    """
```

Use the 10-year G-Sec yield series as the representative short-rate proxy.
Report AIC = 2k - 2·ln(L) where k=3 (number of parameters). This enables model
comparison across Vasicek, CIR, and Hull-White.

---

### Step 7 — `calibration/cir_calibrate.py`

MLE using the non-central chi-squared transition density.

The CIR transition can be written as a scaled non-central chi-squared:

```
r(t+Δt) given r(t) has density:
f(x | r(t)) = c · ncx2.pdf(2cx, df, λ)
```

where:
```
c   = 2κ / (σ²(1 - e^(-κΔt)))
df  = 4κθ/σ²
λ   = 2c · e^(-κΔt) · r(t)    (non-centrality parameter)
```

```python
def cir_log_likelihood(params: tuple, rate_series: np.ndarray,
                        dt: float) -> float:
    """
    Uses scipy.stats.ncx2.logpdf for exact density.
    Enforces Feller condition as penalty: return +inf if 2*kappa*theta <= sigma^2.
    """

def calibrate_cir(rate_series: np.ndarray, dt: float,
                   initial_guess: tuple = (1.0, 0.065, 0.05)) -> dict:
    """
    Bounds: kappa in (0.01, 10), theta in (0.001, 0.3), sigma in (0.001, 0.3)
    Hard constraint: 2*kappa*theta > sigma^2 (Feller condition).
    Returns dict with keys: kappa, theta, sigma, log_likelihood, aic, feller_satisfied
    """
```

---

### Step 8 — `models/hull_white.py`

```python
def compute_theta_t(tenors: np.ndarray, discount_factors: np.ndarray,
                     a: float, sigma: float,
                     fine_grid: np.ndarray) -> np.ndarray:
    """
    Computes θ(t) = df^M(0,t)/dt + a·f^M(0,t) + σ²(1-e^(-2at))/(2a)
    at each point in fine_grid using finite differences on f^M(0,t).

    Steps:
    1. Interpolate P^M(0,t) at fine_grid
    2. Compute f^M(0,t) = -(ln P^M(0,t+dt) - ln P^M(0,t)) / dt
    3. Compute df^M/dt via finite differences
    4. Apply the formula above
    """

def hull_white_bond_price(r: float, t: float, T: float,
                           a: float, sigma: float,
                           tenors: np.ndarray,
                           discount_factors: np.ndarray) -> float:
    """
    Analytical ZCB price under Hull-White.
    Uses market discount factors P^M(0,t) and P^M(0,T) directly.
    ln A(t,T) formula as given in Mathematical Foundations section.
    """

def simulate_hull_white(r0: float, a: float, sigma: float,
                         theta_grid: np.ndarray,  # θ(t) precomputed
                         dt: float, N_paths: int, N_steps: int,
                         seed: int = None) -> np.ndarray:
    """
    Euler simulation with precomputed θ(t) array.
    r(t+dt) = r(t) + (theta_grid[i] - a*r(t))*dt + sigma*sqrt(dt)*Z
    Returns paths of shape (N_paths, N_steps+1).
    """
```

---

### Step 9 — `calibration/hull_white_calibrate.py`

```python
def calibrate_hull_white_historical(rate_series: np.ndarray,
                                     dt: float) -> tuple[float, float]:
    """
    Fit a and sigma from historical rate changes.
    Method: regress (r_{t+1} - r_t) on r_t.
    Intercept = a*theta_eff*dt (but theta_eff is absorbed into theta(t)),
    Slope = -a*dt → a = -slope/dt
    Residual std = sigma*sqrt(dt) → sigma = std(residuals)/sqrt(dt)
    """

def fit_theta_to_yield_curve(a: float, sigma: float,
                               tenors: np.ndarray,
                               discount_factors: np.ndarray,
                               fine_grid_dt: float = 1/365
                               ) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes θ(t) on a fine grid using the analytical formula.
    Returns (time_grid, theta_values).
    """

def validate_yield_curve_fit(a: float, sigma: float,
                              r0: float,
                              tenors: np.ndarray,
                              market_yields: np.ndarray,
                              discount_factors: np.ndarray) -> pd.DataFrame:
    """
    For each observed tenor, compute Hull-White model yield and compare to market.
    Returns DataFrame with columns: tenor, market_yield, model_yield, error_bps.
    Errors should be near-zero for Hull-White (exact fit by construction).
    """
```

Validation gate: Hull-White model yields must match market yields within 1 basis point
for all observed tenors (by construction of θ(t)). If this fails, the θ(t) computation
has an error.

---

### Step 10 — `pricing/yield_curve.py`

```python
def fit_nelson_siegel(tenors: np.ndarray, yields: np.ndarray) -> dict:
    """
    Fit Nelson-Siegel parametrization:
    R(τ) = β₀ + β₁·(1 - e^(-λτ))/(λτ) + β₂·[(1-e^(-λτ))/(λτ) - e^(-λτ)]

    Returns dict with keys: beta0, beta1, beta2, lambda_, rmse
    beta0 = long-run yield (level)
    beta1 = short-end slope
    beta2 = hump (medium-term factor)
    """

def nelson_siegel_yield(tau: float, beta0: float, beta1: float,
                         beta2: float, lambda_: float) -> float:
    """Evaluate Nelson-Siegel formula at a single maturity tau."""

def nelson_siegel_discount_factor(tau: float, **ns_params) -> float:
    """P(0,tau) = exp(-R(tau) * tau)"""
```

Nelson-Siegel is used as the smooth curve-fitting method for interpolation across the
observed sparse tenors. It gives more stable forward rates than linear interpolation,
which matters for computing df^M(0,t)/dt in the Hull-White θ(t) formula.

---

### Step 11 — `pricing/caps_floors.py`

```python
def forward_rate(t1: float, t2: float,
                  discount_factors_fn) -> float:
    """
    Simply-compounded forward rate for period [t1, t2]:
    F = (P^M(0,t1)/P^M(0,t2) - 1) / (t2 - t1)
    """

def caplet_price_black(t_reset: float, t_pay: float, K: float,
                        vol: float, discount_factors_fn) -> float:
    """
    Single caplet price via Black's formula.
    discount_factors_fn(t): callable returning P^M(0,t)
    """

def cap_price(reset_dates: list[float], payment_dates: list[float],
               K: float, vols: list[float],
               discount_factors_fn) -> float:
    """
    Cap price = sum of caplet prices.
    vols: one vol per caplet (or single flat vol replicated).
    """

def floorlet_price_black(t_reset: float, t_pay: float, K: float,
                          vol: float, discount_factors_fn) -> float:

def floor_price(reset_dates: list[float], payment_dates: list[float],
                 K: float, vols: list[float],
                 discount_factors_fn) -> float:

def swap_rate(reset_dates: list[float], payment_dates: list[float],
               discount_factors_fn) -> float:
    """
    Par swap rate K* such that payer swap = 0.
    K* = (P^M(0,t_0) - P^M(0,t_N)) / sum(delta_i * P^M(0,t_{i+1}))
    """
```

---

### Step 12 — `analysis/model_comparison.py`

For each of the three models (Vasicek, CIR, Hull-White), calibrated to Indian G-Sec data:

1. Plot model yield curve vs market yield curve for a single representative date.
2. Compute RMSE of model yield vs market yield across all observed tenors.
3. Plot simulated rate paths (fan chart: 5th, 25th, 50th, 75th, 95th percentiles) over 5 years.
4. Report calibrated parameters for each model in a table.

Expected finding to document: Hull-White fits the initial yield curve exactly (near-zero
RMSE by construction), while Vasicek and CIR fit approximately. The tradeoff: Vasicek and
CIR have fewer parameters and may generalize better out-of-sample.

---

### Step 13 — `analysis/term_structure.py`

Time-series analysis of the yield curve across the historical data window:

1. Compute the model yield curve for each trading day using the calibrated
   (static) parameters and observed r(t) on each day.
2. Compare to market yield curve on each day.
3. Plot the evolution of model vs market 2y, 5y, 10y yields over time.

This is the primary empirical section: does the model with fixed calibrated
parameters track the market yield curve over time?

---

### Step 14 — `analysis/rate_paths.py`

Visualization of simulated rate paths:

```python
def plot_rate_fan(model: str, params: dict, r0: float,
                   T: float, N_paths: int, N_steps: int,
                   percentiles: list = [5, 25, 50, 75, 95]):
    """
    Fan chart showing distribution of future rate paths.
    X-axis: time (years). Y-axis: short rate (%).
    Solid line at median; shaded bands at other percentiles.
    """
```

Plot all three models on the same axes for direct comparison of distributional
properties (Vasicek can go negative; CIR stays positive; Hull-White inherits Vasicek's
Gaussian tails but is calibrated to current rates).

---

## Tests

### `tests/test_bond_pricing.py`

```python
# Vasicek bond price at known parameters
# Benchmark: kappa=0.5, theta=0.05, sigma=0.02, r=0.05, tau=5
# Expected: approximately 0.7788 (verify independently)
def test_vasicek_bond_price():
    price = vasicek_bond_price(r=0.05, kappa=0.5, theta=0.05, sigma=0.02, tau=5.0)
    assert abs(price - 0.7788) < 0.001

# CIR reduces to Vasicek when sigma -> 0
def test_cir_converges_to_vasicek_at_low_vol():
    cir_p = cir_bond_price(r=0.05, kappa=0.5, theta=0.05, sigma=0.001, tau=5.0)
    vas_p = vasicek_bond_price(r=0.05, kappa=0.5, theta=0.05, sigma=0.001, tau=5.0)
    assert abs(cir_p - vas_p) / vas_p < 0.001

# Hull-White exact yield curve fit
def test_hull_white_exact_yield_curve_fit():
    # After calibration, model yields must match market within 1 bps
    errors_bps = validate_yield_curve_fit(...)['error_bps'].abs()
    assert errors_bps.max() < 1.0
```

### `tests/test_yield_curve.py`

```python
# Flat yield curve: discount factors should equal exp(-r*tau)
def test_flat_yield_curve_discount_factors():
    ...

# Nelson-Siegel fit: RMSE < 5 bps on observed Indian data
def test_nelson_siegel_fit_quality():
    ...
```

### `tests/test_cap_floor_parity.py`

```python
# Cap - Floor = Payer Swap (at the swap rate K*)
# cap(K*) - floor(K*) = 0 by definition
def test_cap_floor_parity_at_par():
    K_star = swap_rate(...)
    cap = cap_price(..., K=K_star, ...)
    floor = floor_price(..., K=K_star, ...)
    assert abs(cap - floor) < 1e-6  # should be exactly zero
```

### `tests/test_feller.py`

```python
# CIR with Feller satisfied: all simulated paths must stay >= 0
def test_cir_non_negative_when_feller_satisfied():
    paths = simulate_cir_euler(r0=0.065, kappa=2.0, theta=0.065,
                                sigma=0.05, T=10.0,
                                N_paths=1000, N_steps=1000, seed=42)
    assert np.all(paths >= 0)

# Confirm Feller: 2*kappa*theta = 2*2.0*0.065 = 0.26 > sigma^2 = 0.0025 ✓
```

---

## Deliverables

1. Public GitHub repo (`interest-rate-models`) with README explaining the CIR → Heston
   connection clearly
2. `notebooks/01_rates_walkthrough.ipynb`: guided walkthrough of all three models,
   calibration, and yield curve comparison
3. Three-model calibration table: parameters + AIC for Vasicek, CIR, Hull-White
4. Yield curve fit plot: model vs market, all three models overlaid
5. Rate fan chart: 5-year simulated rate distributions, all three models
6. Cap/floor pricing table: at three strike levels (ATM, 100bps above, 100bps below)
   under Hull-White calibrated to current yield curve

---

## Timeline

| Week | Milestone |
|---|---|
| 1 | RBI data pipeline working; Vasicek model implemented and calibrated |
| 2 | CIR implemented and calibrated; bond pricing validated for both |
| 3 | Nelson-Siegel yield curve fit; Hull-White θ(t) computation |
| 4 | Hull-White calibration; cap/floor pricer; all three models compared |
| 5 | Analysis plots, model comparison table, fan charts |
| 6 | Notebook, README, polish |

---

## Key Interview Talking Points This Project Enables

- "I noticed that the CIR process in Heston's variance equation is formally identical
  to the CIR interest rate model — same SDE, same Feller condition, same affine bond
  pricing formula. I implemented both and the math is the same."
- "Hull-White fits the yield curve exactly by construction via θ(t) — it doesn't minimize
  a loss function against market rates, it computes θ(t) analytically. That's a
  fundamental difference from Vasicek and CIR calibration."
- "For CIR, I calibrated using the non-central chi-squared MLE — the exact transition
  density, not an Euler approximation. AIC comparison showed [CIR/Vasicek] had better
  fit to the Indian G-Sec time series over the calibration window."
- "I priced caps and floors using Black's formula and verified cap-floor parity:
  cap(K*) - floor(K*) = 0 at the par swap rate, to the cent."
