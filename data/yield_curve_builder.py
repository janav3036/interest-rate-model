import numpy as np
from scipy.interpolate import interp1d

def bootstrap_discount_factors(yields: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    discount_factors = np.exp(-yields * tenors)
    return discount_factors

def interpolate_discount_factors(tenors: np.ndarray, discount_factors: np.ndarray, t: float) -> float:
    t = np.clip(t, tenors[0], tenors[-1])
    log_df = np.log(discount_factors)
    interp = interp1d(tenors, log_df, kind="linear")
    return float(np.exp(interp(t)))

def compute_forward_rate(tenors: np.ndarray, discount_factors: np.ndarray, t: float, dt: float = 1/365) -> float:
    p_t = interpolate_discount_factors(tenors, discount_factors, t)
    p_t_dt = interpolate_discount_factors(tenors, discount_factors, t+dt)
    return -(np.log(p_t_dt) - np.log(p_t)) / dt