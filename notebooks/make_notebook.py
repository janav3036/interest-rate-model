import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(src):
    return nbf.v4.new_code_cell(src)

# ── Section 1: Introduction ───────────────────────────────────────────────────
cells.append(md("""# Stochastic Interest Rate Models — Indian G-Sec Market

Three models calibrated to real RBI G-Sec yield data:
- **Vasicek (1977)** — Gaussian rates, analytically tractable
- **CIR (1985)** — Non-negative rates; its SDE is identical to Heston's variance process
- **Hull-White (1990)** — Exact yield curve fit via time-dependent drift θ(t)

> **CIR ↔ Heston connection:** `dV = κ(θ−V)dt + σ√V dW` is the Heston variance SDE.
> `dr = κ(θ−r)dt + σ√r dW` is the CIR interest rate SDE. Same math, different interpretation.
"""))

cells.append(code("""\
import numpy as np
import matplotlib.pyplot as plt
from data.rbi_loader import load_cached_yields, TENORS, TENOR_COLS
from pricing.yield_curve import fit_nelson_siegel, nelson_siegel_yield, nelson_siegel_discount_factor

df = load_cached_yields()
plt.figure(figsize=(10, 3))
plt.plot(df["date"], df["tenor_10y"] * 100)
plt.title("10-Year G-Sec Yield (Historical)")
plt.ylabel("Yield (%)")
plt.tight_layout()
plt.show()
"""))

# ── Section 2: Nelson-Siegel Yield Curve Fit ─────────────────────────────────
cells.append(md(r"""## Yield Curve Bootstrapping & Nelson-Siegel Fit

We fit the Nelson-Siegel parametrization to observed G-Sec par yields:

$$R(\tau) = \beta_0 + \beta_1 \frac{1-e^{-\lambda\tau}}{\lambda\tau} + \beta_2\left[\frac{1-e^{-\lambda\tau}}{\lambda\tau} - e^{-\lambda\tau}\right]$$

This smooth fit is critical for Hull-White: computing θ(t) requires a stable derivative of the forward rate.
"""))

cells.append(code("""\
tenors = np.array(TENORS)
last_row = df.iloc[-1]
market_yields = last_row[TENOR_COLS].values.astype(float)
ns_params = fit_nelson_siegel(tenors, market_yields)

fine = np.linspace(0.01, 12, 200)
ns_yields = np.array([nelson_siegel_yield(t, **ns_params) for t in fine])

plt.figure(figsize=(8, 4))
plt.plot(tenors, market_yields * 100, "ko", label="Market (observed)")
plt.plot(fine, ns_yields * 100, "b-", label="Nelson-Siegel fit")
plt.xlabel("Tenor (years)")
plt.ylabel("Yield (%)")
plt.title("Nelson-Siegel Yield Curve Fit — Indian G-Sec")
plt.legend()
plt.tight_layout()
plt.show()
print(f"NS params: {ns_params}")
"""))

# ── Section 3: Vasicek Model ──────────────────────────────────────────────────
cells.append(md(r"""## Vasicek Model

$$dr(t) = \kappa(\theta - r(t))\,dt + \sigma\,dW(t)$$

Affine bond price: $P(t,T) = A(t,T)\cdot e^{-B(t,T)\,r(t)}$

Rates are **Gaussian** — analytically tractable but can go negative.
Calibrated via MLE using the exact Gaussian transition density.
"""))

cells.append(code("""\
from models.vasicek import vasicek_bond_price, vasicek_yield, simulate_vasicek_exact
from calibration.vasicek_calibrate import calibrate_vasicek

# Validation gate
p = vasicek_bond_price(r=0.05, kappa=0.5, theta=0.05, sigma=0.02, tau=5.0)
print(f"Vasicek bond price (κ=0.5, θ=0.05, σ=0.02, r=0.05, τ=5): {p:.4f}  (expected ≈0.7788)")

rates = df["tenor_10y"].values
vas_p = calibrate_vasicek(rates)
print(f"\\nCalibrated: κ={vas_p['kappa']:.4f}, θ={vas_p['theta']:.4f}, σ={vas_p['sigma']:.4f}, AIC={vas_p['aic']:.1f}")

r0 = rates[-1]
paths = simulate_vasicek_exact(r0, vas_p["kappa"], vas_p["theta"], vas_p["sigma"],
                               T=5, N_paths=200, N_steps=252*5, seed=42)
t_grid = np.linspace(0, 5, paths.shape[1])
pcts = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
plt.figure(figsize=(8, 4))
plt.fill_between(t_grid, pcts[0]*100, pcts[4]*100, alpha=0.15, color="blue")
plt.fill_between(t_grid, pcts[1]*100, pcts[3]*100, alpha=0.25, color="blue")
plt.plot(t_grid, pcts[2]*100, "b-", label="Median")
plt.xlabel("Years"); plt.ylabel("Rate (%)"); plt.title("Vasicek — Simulated Rate Paths")
plt.legend(); plt.tight_layout(); plt.show()
"""))

# ── Section 4: CIR Model ──────────────────────────────────────────────────────
cells.append(md(r"""## CIR Model

$$dr(t) = \kappa(\theta - r(t))\,dt + \sigma\sqrt{r(t)}\,dW(t)$$

**Feller condition:** $2\kappa\theta > \sigma^2$ ensures $r(t) > 0$ almost surely.
This is identical to the Feller condition in Heston's variance model.

Calibrated via MLE using the **non-central chi-squared** transition density (exact, not Euler).
"""))

cells.append(code("""\
from models.cir import cir_yield, simulate_cir_exact, feller_condition_satisfied
from calibration.cir_calibrate import calibrate_cir

cir_p = calibrate_cir(rates)
print(f"Calibrated: κ={cir_p['kappa']:.4f}, θ={cir_p['theta']:.4f}, σ={cir_p['sigma']:.4f}")
print(f"Feller satisfied: {feller_condition_satisfied(cir_p['kappa'], cir_p['theta'], cir_p['sigma'])}")
print(f"AIC={cir_p['aic']:.1f}")

paths_cir = simulate_cir_exact(r0, cir_p["kappa"], cir_p["theta"], cir_p["sigma"],
                               T=5, N_paths=200, N_steps=252*5, seed=42)
print(f"\\nAll CIR paths non-negative: {np.all(paths_cir >= 0)}")
pcts_cir = np.percentile(paths_cir, [5, 25, 50, 75, 95], axis=0)
plt.figure(figsize=(8, 4))
plt.fill_between(t_grid, pcts_cir[0]*100, pcts_cir[4]*100, alpha=0.15, color="green")
plt.fill_between(t_grid, pcts_cir[1]*100, pcts_cir[3]*100, alpha=0.25, color="green")
plt.plot(t_grid, pcts_cir[2]*100, "g-", label="Median")
plt.xlabel("Years"); plt.ylabel("Rate (%)"); plt.title("CIR — Simulated Rate Paths (non-negative)")
plt.legend(); plt.tight_layout(); plt.show()
"""))

# ── Section 5: Hull-White Model ───────────────────────────────────────────────
cells.append(md(r"""## Hull-White Model (Extended Vasicek)

$$dr(t) = (\theta(t) - a\,r(t))\,dt + \sigma\,dW(t)$$

θ(t) is **not** a free parameter — it is computed analytically from the market yield curve:

$$\theta(t) = \frac{\partial f^M(0,t)}{\partial t} + a\,f^M(0,t) + \frac{\sigma^2(1-e^{-2at})}{2a}$$

This guarantees an **exact fit** to the initial yield curve by construction.
`a` and `σ` are calibrated from historical rate changes via regression.
"""))

cells.append(code("""\
from calibration.hull_white_calibrate import (
    calibrate_hull_white_historical, validate_yield_curve_fit
)
from models.hull_white import hull_white_bond_price

a, sigma_hw = calibrate_hull_white_historical(rates, dt=1/252)
r0_hw = nelson_siegel_yield(1e-4, **ns_params)
print(f"Hull-White: a={a:.4f}, σ={sigma_hw:.4f}")

hw_df = validate_yield_curve_fit(a, sigma_hw, r0_hw, tenors, market_yields, ns_params)
print("\\nYield curve fit (basis points error):")
print(hw_df[["tenor", "market_yield", "model_yield", "error_bps"]].to_string(index=False))
print(f"\\nMax error: {hw_df['error_bps'].abs().max():.2f} bps  (target: < 1 bps)")
"""))

# ── Section 6: Model Comparison ───────────────────────────────────────────────
cells.append(md("""## Model Comparison

All three models calibrated to the same G-Sec data.
Hull-White fits exactly (near-zero RMSE by construction).
Vasicek and CIR fit approximately — fewer parameters, potentially better out-of-sample.
"""))

cells.append(code("""\
import pandas as pd

vas_yields = np.array([vasicek_yield(r0, vas_p["kappa"], vas_p["theta"], vas_p["sigma"], t) for t in tenors])
cir_yields = np.array([cir_yield(r0, cir_p["kappa"], cir_p["theta"], cir_p["sigma"], t) for t in tenors])
hw_yields  = hw_df["model_yield"].values

def rmse_bps(m, mkt): return np.sqrt(np.mean((m - mkt)**2)) * 10000

plt.figure(figsize=(9, 5))
plt.plot(tenors, market_yields*100, "ko-", label="Market", linewidth=2)
plt.plot(tenors, vas_yields*100, "b--", label=f"Vasicek  RMSE={rmse_bps(vas_yields, market_yields):.1f} bps")
plt.plot(tenors, cir_yields*100, "g--", label=f"CIR      RMSE={rmse_bps(cir_yields, market_yields):.1f} bps")
plt.plot(tenors, hw_yields*100,  "r--", label=f"Hull-White RMSE={rmse_bps(hw_yields, market_yields):.1f} bps")
plt.xlabel("Tenor (years)"); plt.ylabel("Yield (%)")
plt.title("Model vs Market Yield Curve — Indian G-Sec")
plt.legend(); plt.tight_layout(); plt.show()

print(pd.DataFrame({
    "Model":     ["Vasicek", "CIR", "Hull-White"],
    "kappa/a":   [f"{vas_p['kappa']:.4f}", f"{cir_p['kappa']:.4f}", f"{a:.4f}"],
    "theta":     [f"{vas_p['theta']:.4f}", f"{cir_p['theta']:.4f}", "—"],
    "sigma":     [f"{vas_p['sigma']:.4f}", f"{cir_p['sigma']:.4f}", f"{sigma_hw:.4f}"],
    "AIC":       [f"{vas_p['aic']:.1f}", f"{cir_p['aic']:.1f}", "—"],
    "RMSE(bps)": [f"{rmse_bps(vas_yields, market_yields):.1f}",
                  f"{rmse_bps(cir_yields, market_yields):.1f}",
                  f"{rmse_bps(hw_yields, market_yields):.1f}"],
}).to_string(index=False))
"""))

# ── Section 7: Cap/Floor Pricing ──────────────────────────────────────────────
cells.append(md(r"""## Cap / Floor Pricing (Black's Model)

A **cap** is a strip of caplets. Each caplet pays $\max(L(t_i) - K, 0)\cdot\delta$ at $t_{i+1}$.

Under Black's market model:
$$\text{Caplet}(t_i) = \delta \cdot P^M(0,t_{i+1})\,[F_i\,N(d_1) - K\,N(d_2)]$$

**Cap-floor parity:** $\text{Cap}(K^*) - \text{Floor}(K^*) = 0$ at the par swap rate $K^*$.
"""))

cells.append(code("""\
from pricing.caps_floors import cap_price, floor_price, swap_rate

discount_fn = lambda t: float(nelson_siegel_discount_factor(t, **ns_params))

reset_dates   = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
payment_dates = [0.5,  0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
K_star = swap_rate(reset_dates, payment_dates, discount_fn)
print(f"Par swap rate: {K_star*100:.3f}%")

flat_vol = 0.15
cap   = cap_price(reset_dates, payment_dates, K_star, flat_vol, discount_fn)
floor = floor_price(reset_dates, payment_dates, K_star, flat_vol, discount_fn)
print(f"Cap(K*)  = {cap:.6f}")
print(f"Floor(K*)= {floor:.6f}")
print(f"Cap - Floor = {cap - floor:.2e}  (should be ≈ 0 by parity)")

strikes = [K_star - 0.01, K_star, K_star + 0.01]
labels  = ["ATM-100bps", "ATM", "ATM+100bps"]
print("\\nPricing table (flat vol=15%):")
print(f"{'Strike':>12} {'Cap':>12} {'Floor':>12}")
for K, lbl in zip(strikes, labels):
    c = cap_price(reset_dates, payment_dates, K, flat_vol, discount_fn)
    f = floor_price(reset_dates, payment_dates, K, flat_vol, discount_fn)
    print(f"{lbl:>12} {c:>12.6f} {f:>12.6f}")
"""))

# ── Section 8: Term Structure Over Time ───────────────────────────────────────
cells.append(md("""## Term Structure Over Time

Model yields (with static calibrated parameters, observed r(t) each day)
vs market yields at 2y, 5y, and 10y tenors over the full historical window.

Hull-White is anchored to the final-date yield curve, so it diverges further back in time.
Vasicek and CIR track the level shift but miss curve shape changes.
"""))

cells.append(code("""\
from analysis.term_structure import run_term_structure
import matplotlib.dates as mdates

result = run_term_structure()
"""))

# ── Write notebook ────────────────────────────────────────────────────────────
nb.cells = cells
output = "notebooks/01_rates_walkthrough.ipynb"
with open(output, "w") as f:
    nbf.write(nb, f)
print(f"Written: {output}")
