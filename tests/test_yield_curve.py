import numpy as np
import pytest
from pricing.yield_curve import fit_nelson_siegel, nelson_siegel_yield, nelson_siegel_discount_factor


FLAT_TENORS = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
FLAT_YIELDS = np.full(len(FLAT_TENORS), 0.065)

GSEC_TENORS = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
GSEC_YIELDS = np.array([0.0650, 0.0672, 0.0690, 0.0710, 0.0720, 0.0725])


def test_fit_returns_required_keys():
    params = fit_nelson_siegel(FLAT_TENORS, FLAT_YIELDS)
    for key in ("beta0", "beta1", "beta2", "lambda_", "rmse"):
        assert key in params


def test_flat_curve_discount_factors():
    params = fit_nelson_siegel(FLAT_TENORS, FLAT_YIELDS)
    for t in FLAT_TENORS:
        p_model = nelson_siegel_discount_factor(t, **params)
        p_exact = np.exp(-0.065 * t)
        assert abs(p_model - p_exact) < 1e-4, f"Discount factor error at t={t}: {abs(p_model - p_exact)}"


def test_gsec_rmse_under_5bps():
    params = fit_nelson_siegel(GSEC_TENORS, GSEC_YIELDS)
    assert params["rmse"] < 0.0005, f"RMSE too large: {params['rmse']:.6f}"


def test_long_rate_equals_beta0():
    params = fit_nelson_siegel(GSEC_TENORS, GSEC_YIELDS)
    long_rate = nelson_siegel_yield(100.0, **params)
    assert abs(long_rate - params["beta0"]) < 1e-4
