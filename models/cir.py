import numpy as np
from scipy.stats import ncx2

def feller_condition_satisfied(kappa: float, theta: float, sigma: float) -> bool:
    return 2 * kappa * theta > sigma**2

def cir_bond_price(r: float, kappa: float, theta: float,
                   sigma: float, tau: float) -> float:
    gamma = np.sqrt(kappa**2 + 2 * sigma ** 2)
    exp_gt = np.exp(gamma * tau)

    denom = (gamma + kappa) * (exp_gt - 1) + 2 * gamma
    B = 2 * (exp_gt - 1) / denom
    A = (2 * gamma * np.exp((kappa + gamma) * tau / 2) / denom) ** (2 * kappa * theta / sigma ** 2)
    return A * np.exp(-B * r)

def cir_yield(r: float, kappa: float, theta: float,
              sigma: float, tau: float) -> float:
    P = cir_bond_price(r, kappa, theta, sigma, tau)
    return -np.log(P) / tau

def simulate_cir_euler(r0: float, kappa: float, theta: float, 
                       sigma: float, T: float, 
                       N_paths: int, N_steps: int,
                       seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    dt = T / N_steps
    paths = np.zeros((N_paths, N_steps + 1))
    paths[:, 0] = r0
    Z = np.random.standard_normal((N_paths, N_steps))
    for i in range(N_steps):
        r = paths[:, i]
        dr = kappa * (theta - r) * dt + sigma * np.sqrt(np.maximum(r, 0)) * np.sqrt(dt) * Z[:, i]
        paths[:, i + 1] = np.maximum(r + dr, 0)
    return paths


def simulate_cir_exact(r0: float, kappa: float, theta: float,
                       sigma: float, T: float,
                       N_paths: int, N_steps: int,
                       seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    dt = T / N_steps
    paths = np.zeros((N_paths, N_steps + 1))
    paths[:, 0] = r0
    df = 4 * kappa * theta / sigma ** 2
    for i in range(N_steps):
        r = paths[:, i]
        c = 2 * kappa / (sigma ** 2 * (1 - np.exp(-kappa * dt)))
        lam = 2 * c * np.exp(-kappa * dt) * r
        paths[:, i + 1] = ncx2.rvs(df, lam, size=N_paths, random_state=None) / (2 * c)
    return paths