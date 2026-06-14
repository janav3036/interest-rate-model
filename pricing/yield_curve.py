import numpy as np
from scipy.optimize import minimize

def nelson_siegel_yield(tau: float, beta0: float, beta1: float, 
                        beta2: float, lambda_: float, **_) -> float:
    """Evaluate Nelson-Seigel yield at maturity tau (in years)"""
    if tau <= 0:
        return beta0 + beta1
    factor = (1 - np.exp(-lambda_ * tau)) / (lambda_ * tau)
    return beta0 + beta1 * factor + beta2 * (factor - np.exp(-lambda_ * tau))

def fit_nelson_siegel(tenors: np.ndarray, yields: np.ndarray) -> dict:
    """Fit Nelson-Siegel to observed (tenor, yield) pairs"""
    def objective(params):
        beta0, beta1, beta2, lambda_ = params
        if lambda_ <= 0 or beta0 <= 0:
            return 1e10
        predicted = np.array([
            nelson_siegel_yield(t, beta0, beta1, beta2, lambda_)
            for t in tenors
        ])
        return np.sum((predicted - yields) ** 2)
    
    best = None
    for lam_init in [0.5, 1.0, 2.0, 3.0]:
        x0 = [yields.mean(), yields[0] - yields[-1], 0.0, lam_init]
        res = minimize(
            objective,
            x0, 
            method="Nelder-Mead",
            options={
                'maxiter': 10000,
                'xatol': 1e-8,
                'fatol': 1e-10
            }
        )
        if best is None or res.fun < best.fun:
            best = res

    beta0, beta1, beta2, lambda_ = best.x
    predicted = np.array([
        nelson_siegel_yield(t, beta0, beta1, beta2, lambda_)
            for t in tenors
    ])
    rmse = np.sqrt(np.mean((predicted - yields) ** 2))
    return {'beta0': beta0, 'beta1': beta1, 'beta2': beta2,
            'lambda_': lambda_, 'rmse': rmse}
        
def nelson_siegel_discount_factor(tau: float, beta0: float, beta1: float,
                                   beta2: float, lambda_: float, **_) -> float:
    """P(0, tau) = exp(-R(tau) * tau)"""
    if tau <= 0:
        return 1.0
    r = nelson_siegel_yield(tau, beta0, beta1, beta2, lambda_)
    return np.exp(-r * tau)
