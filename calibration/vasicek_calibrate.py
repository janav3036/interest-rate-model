import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def vasicek_log_likelihood(params: np.ndarray, rates: np.ndarray, dt: float) -> float:
    kappa, theta, sigma = params
    if kappa<=0 or sigma <= 0:
        return np.inf
    
    exp_k = np.exp(-kappa * dt)
    mu = rates[:-1] * exp_k + theta * (1-exp_k)
    var = sigma**2 * (1 - np.exp(-2 * kappa * dt)) / (2 * kappa)

    ll = np.sum(norm.logpdf(rates[1:], loc=mu, scale=np.sqrt(var)))
    return -ll

def calibrate_vasicek(rates: np.ndarray, dt: float = 1/252) -> dict:
    r0 = [0.5, np.mean(rates), 0.01]
    bounds = [(1e-4, None), (0.01, 0.20), (1e-4, None)]

    result = minimize(
        vasicek_log_likelihood,
        r0, 
        args = (rates, dt),
        method = "L-BFGS-B",
        bounds=bounds
    )

    kappa, theta, sigma = result.x
    ll = -result.fun
    aic = 2*3 - 2*ll

    return {
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "log_likelihood": ll,
        "aic": aic,
    }