import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data.rbi_loader import load_cached_yields, TENOR_COLS, TENORS
from pricing.yield_curve import fit_nelson_siegel, nelson_siegel_yield
from calibration.vasicek_calibrate import calibrate_vasicek
from calibration.cir_calibrate import calibrate_cir
from calibration.hull_white_calibrate import calibrate_hull_white_historical
from models.vasicek import vasicek_yield
from models.cir import cir_yield
from models.hull_white import hull_white_bond_price

TARGET_TENORS = [2.0, 5.0, 10.0]
TARGET_COLS = ["tenor_2y", "tenor_5y", "tenor_10y"]

def run_term_structure():
    # -- DATA --
    df = load_cached_yields()
    rates = df["tenor_10y"].values

    # Using last snapshot for NS fit 
    last_row = df.iloc[-1]
    market_yields_snap = last_row[TENOR_COLS].values.astype(float)
    tenors_all = np.array(TENORS)
    ns_params = fit_nelson_siegel(tenors_all, market_yields_snap)

    # -- CALIBRATE ONCE FOR STATIC PARAMS --
    vas_p = calibrate_vasicek(rates)
    cir_p = calibrate_cir(rates)
    a, sigma_hw = calibrate_hull_white_historical(rates, dt = 1 / 252)

    # -- DAILY LOOP -- 
    records = []
    for _, row in df.iterrows():
        r_t = row["tenor_10y"]
        r0_hw = nelson_siegel_yield(1e-4, **ns_params)

        entry = {"date": row["date"]}
        for tau, col in zip(TARGET_TENORS, TARGET_COLS):
            entry[f"market_{col}"] = row[col]
            entry[f"vasicek_{col}"] = vasicek_yield(
                  r_t, vas_p["kappa"], vas_p["theta"], vas_p["sigma"], tau
              )
            entry[f"cir_{col}"] = cir_yield(
                  r_t, cir_p["kappa"], cir_p["theta"], cir_p["sigma"], tau
              )
            p_hw = hull_white_bond_price(r0_hw, 0.0, tau, a, sigma_hw, ns_params)
            entry[f"hw_{col}"] = -np.log(p_hw) / tau
        records.append(entry)
    result = pd.DataFrame(records)

    # --- PLOT ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    titles = ["2-Year Yield", "5-Year Yield", "10-Year Yield"]

    for ax, tau, col, title in zip(axes, TARGET_TENORS, TARGET_COLS, titles):
        ax.plot(result["date"], result[f"market_{col}"]  * 100, "k-",  label="Market",     linewidth=1.5)
        ax.plot(result["date"], result[f"vasicek_{col}"] * 100, "b--", label="Vasicek",     linewidth=1.0)
        ax.plot(result["date"], result[f"cir_{col}"]     * 100, "g--", label="CIR",         linewidth=1.0)
        ax.plot(result["date"], result[f"hw_{col}"]      * 100, "r--", label="Hull-White",  linewidth=1.0)
        ax.set_ylabel("Yield (%)")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")
    fig.suptitle("Model vs Market Yields Over Time (Indian G-Sec)", fontsize=13)
    plt.tight_layout()
    plt.show()

    return result


if __name__ == "__main__":
    run_term_structure()