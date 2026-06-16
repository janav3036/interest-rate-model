import pytest
import numpy as np
from pricing.bond_pricing import bond_price, coupon_bond_price


VASICEK_PARAMS = {"kappa": 0.5, "theta": 0.05, "sigma": 0.02}
CIR_PARAMS     = {"kappa": 0.5, "theta": 0.05, "sigma": 0.02}


def test_vasicek_bond_price():
    p = bond_price("vasicek", r=0.05, params=VASICEK_PARAMS, tau=5.0)
    assert abs(p - 0.7788) < 0.002


def test_cir_bond_price_positive():
    p = bond_price("cir", r=0.05, params=CIR_PARAMS, tau=5.0)
    assert p > 0


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        bond_price("unknown", r=0.05, params=VASICEK_PARAMS, tau=1.0)


def test_coupon_bond_price_equals_sum_of_zcbs():
    cash_flows = [(1.0, 0.065), (2.0, 0.065), (3.0, 1.065)]
    coupon_price = coupon_bond_price("vasicek", r=0.05,
                                     params=VASICEK_PARAMS,
                                     cash_flows=cash_flows)
    zcb_sum = sum(
        amount * bond_price("vasicek", r=0.05, params=VASICEK_PARAMS, tau=tau)
        for tau, amount in cash_flows
    )
    assert abs(coupon_price - zcb_sum) < 1e-10
