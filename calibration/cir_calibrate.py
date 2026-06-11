import numpy as np
from scipy.optimize import minimize
from scipy.stats import ncx2

def cir_log_likelihood(params: np.ndarray, rates: np.ndarray, dt: float) -> float:
    kappa, theta, sigma = params
    if kappa <= 0 or theta <= 0 or sigma <= 0:
        return np.inf

    c = 2 * kappa / (sigma**2 * (1 - np.exp(-kappa * dt)))
    df = 4 * kappa * theta / sigma**2
    lam = 2 * c * np.exp(-kappa * dt) * rates[:-1]
    scaled_next = 2 * c * rates[1:]

    ll = np.sum(ncx2.logpdf(scaled_next, df=df, nc=lam)) + len(rates[1:]) * np.log(2 * c)
    return -ll

def calibrate_cir(rates: np.ndarray, dt: float = 1/252) -> dict:
    def objective(params):
        kappa, theta, sigma = params
        nll = cir_log_likelihood(params, rates, dt)
        feller = 2 * kappa * theta - sigma**2
        penalty = 1e6 * max(0, -feller)
        return nll + penalty

    r0 = [0.5, np.mean(rates), 0.05]
    bounds = [(1e-4, None), (1e-4, None), (1e-4, None)]

    result = minimize(objective, r0, method="L-BFGS-B", bounds=bounds)

    kappa, theta, sigma = result.x
    if 2 * kappa * theta <= sigma**2:
        raise ValueError(
            f"Feller condition violated at convergence: "
            f"2κθ={2*kappa*theta:.4f} <= σ²={sigma**2:.4f}"
        )

    ll = -cir_log_likelihood(result.x, rates, dt)
    aic = 2 * 3 - 2 * ll

    return {
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "log_likelihood": ll,
        "aic": aic,
    }
