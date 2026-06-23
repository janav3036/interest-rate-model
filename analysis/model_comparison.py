import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.rbi_loader import load_cached_yields, get_yield_snapshot, TENORS, TENOR_COLS
from pricing.yield_curve import fit_nelson_siegel, nelson_siegel_yield
from calibration.vasicek_calibrate import calibrate_vasicek
from calibration.cir_calibrate import calibrate_cir
from calibration.hull_white_calibrate import (
    calibrate_hull_white_historical,
    validate_yield_curve_fit,
)
from models.vasicek import vasicek_yield
from models.cir import cir_yield


def run_comparison():
    # --- DATA ---
    df = load_cached_yields()
    rates = df["tenor_10y"].values

    snapshot = get_yield_snapshot(df)
    market_yields = snapshot[TENOR_COLS].values.astype(float)
    tenors = np.array(TENORS)

    ns_params = fit_nelson_siegel(tenors, market_yields)

    # --- CALIBRATE ---
    r0 = rates[-1]
    # Hull-White exact fit requires r0 = instantaneous short rate from NS curve
    r0_hw = nelson_siegel_yield(1e-4, **ns_params)
    vas_params = calibrate_vasicek(rates)
    cir_params = calibrate_cir(rates)
    a, sigma_hw = calibrate_hull_white_historical(rates, dt=1 / 252)

    # --- MODEL YIELDS ---
    vas_yields = np.array([
        vasicek_yield(r0, vas_params["kappa"], vas_params["theta"], vas_params["sigma"], tau)
        for tau in tenors
    ])

    cir_yields = np.array([
        cir_yield(r0, cir_params["kappa"], cir_params["theta"], cir_params["sigma"], tau)
        for tau in tenors
    ])

    hw_df = validate_yield_curve_fit(a, sigma_hw, r0_hw, tenors, market_yields, ns_params)
    hw_yields = hw_df["model_yield"].values

    # --- RMSE (in basis points) ---
    def rmse_bps(model_y, market_y):
        return np.sqrt(np.mean((model_y - market_y) ** 2)) * 10000

    vas_rmse = rmse_bps(vas_yields, market_yields)
    cir_rmse = rmse_bps(cir_yields, market_yields)
    hw_rmse = rmse_bps(hw_yields, market_yields)

    # --- PLOT ---
    plt.figure(figsize=(10, 6))
    plt.plot(tenors, market_yields * 100, "ko-", label="Market", linewidth=2)
    plt.plot(tenors, vas_yields * 100, "b--", label=f"Vasicek (RMSE={vas_rmse:.1f} bps)")
    plt.plot(tenors, cir_yields * 100, "g--", label=f"CIR (RMSE={cir_rmse:.1f} bps)")
    plt.plot(tenors, hw_yields * 100, "r--", label=f"Hull-White (RMSE={hw_rmse:.1f} bps)")
    plt.xlabel("Tenor (years)")
    plt.ylabel("Yield (%)")
    plt.title("Model Yield Curve vs Market (Indian G-Sec)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- PARAMETER TABLE ---
    table = pd.DataFrame({
        "Model":  ["Vasicek", "CIR", "Hull-White"],
        "kappa/a": [vas_params["kappa"], cir_params["kappa"], a],
        "theta":   [vas_params["theta"], cir_params["theta"], None],
        "sigma":   [vas_params["sigma"], cir_params["sigma"], sigma_hw],
        "AIC":     [vas_params["aic"], cir_params["aic"], None],
        "RMSE(bps)": [vas_rmse, cir_rmse, hw_rmse],
    })
    print("\n=== Model Comparison ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    run_comparison()
