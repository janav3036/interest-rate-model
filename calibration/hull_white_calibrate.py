import numpy as np
import pandas as pd
from models.hull_white import compute_theta_t, hull_white_bond_price
from pricing.yield_curve import fit_nelson_siegel, nelson_siegel_discount_factor


def calibrate_hull_white_historical(rate_series: np.ndarray,
                                     dt: float) -> tuple[float, float]:
    """
    Regress Δr on r to extract a and sigma.
    slope = -a*dt  →  a = -slope/dt
    residual std = sigma*sqrt(dt)  →  sigma = std(residuals)/sqrt(dt)
    """
    r = rate_series[:-1]
    dr = np.diff(rate_series)

    # OLS: dr = intercept + slope * r
    A = np.column_stack([np.ones_like(r), r])
    coeffs, _, _, _ = np.linalg.lstsq(A, dr, rcond=None)
    intercept, slope = coeffs

    a = max(-slope / dt, 1e-4)
    residuals = dr - (intercept + slope * r)
    sigma = np.std(residuals) / np.sqrt(dt)

    return float(a), float(sigma)


def fit_theta_to_yield_curve(a: float, sigma: float,
                              ns_params: dict,
                              fine_grid_dt: float = 1/365,
                              max_tenor: float = 30.0
                              ) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes θ(t) on a fine daily grid using the analytical formula.
    Returns (time_grid, theta_values).
    """
    fine_grid = np.arange(fine_grid_dt, max_tenor + fine_grid_dt, fine_grid_dt)
    theta = compute_theta_t(ns_params, a, sigma, fine_grid)
    return fine_grid, theta


def validate_yield_curve_fit(a: float, sigma: float,
                              r0: float,
                              tenors: np.ndarray,
                              market_yields: np.ndarray,
                              ns_params: dict) -> pd.DataFrame:
    """
    Compare Hull-White model yields to market yields at each tenor.
    Returns DataFrame with columns: tenor, market_yield, model_yield, error_bps.
    """
    rows = []
    for T, y_mkt in zip(tenors, market_yields):
        P_model = hull_white_bond_price(r0, 0.0, T, a, sigma, ns_params)
        y_model = -np.log(P_model) / T
        rows.append({
            'tenor': T,
            'market_yield': y_mkt,
            'model_yield': y_model,
            'error_bps': (y_model - y_mkt) * 10000
        })
    return pd.DataFrame(rows)
