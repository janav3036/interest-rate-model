from models.vasicek import vasicek_bond_price
from models.cir import cir_bond_price
from models.hull_white import hull_white_bond_price

def bond_price(model:str, r:float, params: dict, tau: float) -> float:
    if model == "vasicek":
        return vasicek_bond_price(r, params["kappa"], params["theta"], params["sigma"], tau)
    elif model == "cir":
        return cir_bond_price(r, params["kappa"], params["theta"],
                              params["sigma"], tau)
    elif model == "hull_white":
        return hull_white_bond_price(r, params["t"], params["t"] + tau,
                                     params["a"], params["sigma"],
                                     params["theta_grid"], params["dt"])
    else:
        raise ValueError(f"Unknown model: {model}")


def coupon_bond_price(model: str, r: float, params: dict,
                      cash_flows: list[tuple[float, float]]) -> float:
    """
    cash_flows: list of (tau, amount) pairs
    returns sum of PV of each cash flow
    """
    return sum(
        amount * bond_price(model, r, params, tau)
        for tau, amount in cash_flows
    )