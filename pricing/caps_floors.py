import numpy as np
from scipy.stats import norm

def black_caplet(F: float, K: float, sigma: float, T: float,
                 P: float, tau: float) -> float:
    if sigma <= 0 or T <= 0:
        return max(F - K, 0.0) * P * tau
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return P * tau * (F * norm.cdf(d1) - K * norm.cdf(d2))

def black_floorlet(F: float, K: float, sigma: float, T: float,
                 P: float, tau: float) -> float:
    if sigma <= 0 or T <= 0:
        return max(K - F, 0.0) * P * tau
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return P * tau * (-F * norm.cdf(-d1) + K * norm.cdf(-d2))

def cap_price(schedule: list[tuple[float, float, float]],
              K: float, sigma: float,
              discount_factors_fn) -> float:
    """
    schedule: list of (T_start, T_end, tau) for each caplet
    discount_factors_fn: callable float -> P(0, T)
    """
    total = 0.0
    for T_start, T_end, tau in schedule:
        P_start = discount_factors_fn(T_start)
        P_end = discount_factors_fn(T_end)
        F = (P_start / P_end - 1.0) / tau
        P = P_end
        total += black_caplet(F, K, sigma, T_start, P, tau)
    return total


def floor_price(schedule: list[tuple[float, float, float]],
                K: float, sigma: float,
                discount_factors_fn) -> float:
    total = 0.0
    for T_start, T_end, tau in schedule:
        P_start = discount_factors_fn(T_start)
        P_end = discount_factors_fn(T_end)
        F = (P_start / P_end - 1.0) / tau
        P = P_end
        total += black_floorlet(F, K, sigma, T_start, P, tau)
    return total


def par_swap_rate(schedule: list[tuple[float, float, float]],
                  discount_factors_fn) -> float:
    numerator = discount_factors_fn(schedule[0][0]) - discount_factors_fn(schedule[-1][1])
    denominator = sum(tau * discount_factors_fn(T_end)
                      for _, T_end, tau in schedule)
    return numerator / denominator
