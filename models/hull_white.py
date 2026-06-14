import numpy as np
from pricing.yield_curve import nelson_siegel_discount_factor

def compute_theta_t(ns_params: dict, a: float, sigma: float,
                    fine_grid: np.ndarray) -> np.ndarray:
    """
    θ(t) = df^M(0,t)/dt + a·f^M(0,t) + σ²(1 - e^(-2at)) / (2a)
    Computed on fine_grid using NS-smoothed discount factors.
    """
    dt = fine_grid[1] - fine_grid[0]

    def ln_P(t):
        if t <= 0:
            return 0.0
        df = nelson_siegel_discount_factor(t, **ns_params)
        return np.log(df)
    # instantaneous forward rate: f^M(0,t) = -(d/dt) ln P^M(0,t)
    f = np.array([-(ln_P(t + dt) - ln_P(t)) / dt for t in fine_grid])

    # df/dt via central differences (forward diff at boundary)
    dfdt = np.gradient(f, fine_grid)

    # convexity correction term
    convexity = (sigma ** 2) * (1 - np.exp(-2 * a * fine_grid)) / (2 * a)

    return dfdt + a * f + convexity

def hull_white_bond_price(r: float, t: float, T: float,
                          a: float, sigma: float, 
                          ns_params: dict) -> float:
    """
    Analytical ZCB price under Hull-White: P(t, T) = A(t, T) * exp(-B(t, T) * r)
    Uses NS-smoothed market discount factors
    """

    tau = T - t
    B = (1 - np.exp(-a * tau)) / a

    ln_P_T = np.log(nelson_siegel_discount_factor(T, **ns_params))
    ln_P_t = np.log(nelson_siegel_discount_factor(t, **ns_params)) if t > 0 else 0.0

    # f^M(0,t): instantaneous forward rate at t
    dt_fwd = 1e-5
    if t <= 0:
        ln_P_base = 0.0  # ln P(0,0) = 0 by definition
        ln_P_fwd = np.log(nelson_siegel_discount_factor(dt_fwd, **ns_params))
    else:
        ln_P_base = np.log(nelson_siegel_discount_factor(t, **ns_params))
        ln_P_fwd = np.log(nelson_siegel_discount_factor(t + dt_fwd, **ns_params))
    fM_t = -(ln_P_fwd - ln_P_base) / dt_fwd

    ln_A = (ln_P_T - ln_P_t
            + B * fM_t
            - (sigma ** 2) * (np.exp(-a * t) - np.exp(-a * T)) ** 2
              * (np.exp(2 * a * t) - 1) / (4 * a ** 3))

    return np.exp(ln_A - B * r)

def simulate_hull_white(r0: float, a: float, sigma: float,
                         theta_grid: np.ndarray,
                         dt: float, N_paths: int, N_steps: int,
                         seed: int = None) -> np.ndarray:
    """
    Euler simulation: r(t+dt) = r(t) + (theta(t) - a*r(t))*dt + sigma*sqrt(dt)*Z
    theta_grid must have at least N_steps entries.
    Returns paths of shape (N_paths, N_steps + 1).
    """
    if seed is not None:
        np.random.seed(seed)

    paths = np.zeros((N_paths, N_steps + 1))
    paths[:, 0] = r0

    Z = np.random.randn(N_paths, N_steps)

    for i in range(N_steps):
        theta_i = theta_grid[i]
        r = paths[:, i]
        paths[:, i + 1] = r + (theta_i - a * r) * dt + sigma * np.sqrt(dt) * Z[:, i]

    return paths
