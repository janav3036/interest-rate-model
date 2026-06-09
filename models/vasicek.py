import numpy as np

def vasicek_bond_price(r: float, kappa: float, theta: float,
                      sigma: float, tau: float) -> float:
    B = (1 - np.exp(-kappa * tau)) / kappa
    A = np.exp(
        (B - tau) * (kappa**2 * theta - sigma**2 / 2) / kappa**2
        - sigma**2 * B**2 / (4 * kappa)
    )
    return A * np.exp(-B * r)

def vasicek_yield(r: float, kappa: float, theta: float, 
                  sigma: float, tau: float) -> float:
    P = vasicek_bond_price(r, kappa, theta, sigma, tau)
    return -np.log(P) / tau

def simulate_vasicek_euler(r0: float, kappa: float, theta: float,
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
        paths[:, i+1] = r + kappa * (theta - r) * dt + sigma * np.sqrt(dt) * Z[:, i]
    return paths

def simulate_vasicek_exact(r0: float, kappa: float, theta: float,
                           sigma: float, T: float,
                           N_paths: int, N_steps: int,
                           seed: int = None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / N_steps
    paths = np.zeros((N_paths, N_steps + 1))
    paths[:, 0] = r0
    exp_k = np.exp(-kappa * dt)
    cond_std = np.sqrt(sigma**2 * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa))
    Z = np.random.standard_normal((N_paths, N_steps))
    for i in range(N_steps):
        r = paths[:, i]
        cond_mean = r * exp_k + theta * (1 - exp_k)
        paths[:, i+1] = cond_mean + cond_std * Z[:, i]
    return paths
