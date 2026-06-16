import numpy as np
import pytest
from models.cir import feller_condition_satisfied, simulate_cir_exact, simulate_cir_euler


def test_feller_satisfied():
    # 2 * 0.5 * 0.05 = 0.05 > 0.02^2 = 0.0004
    assert feller_condition_satisfied(kappa=0.5, theta=0.05, sigma=0.02)


def test_feller_violated():
    # 2 * 0.1 * 0.01 = 0.002 < 0.2^2 = 0.04
    assert not feller_condition_satisfied(kappa=0.1, theta=0.01, sigma=0.2)


def test_feller_boundary():
    # exactly 2κθ = σ² → not strictly satisfied
    kappa, theta, sigma = 0.5, 0.02, 0.2
    # 2 * 0.5 * 0.02 = 0.02, sigma^2 = 0.04 → violated
    assert not feller_condition_satisfied(kappa, theta, sigma)


def test_cir_exact_paths_nonnegative():
    paths = simulate_cir_exact(
        r0=0.05, kappa=0.5, theta=0.05, sigma=0.02,
        T=10.0, N_paths=500, N_steps=1000, seed=42
    )
    assert np.all(paths >= 0), "CIR exact paths went negative under Feller condition"


def test_cir_euler_paths_nonnegative():
    paths = simulate_cir_euler(
        r0=0.05, kappa=0.5, theta=0.05, sigma=0.02,
        T=10.0, N_paths=500, N_steps=1000, seed=42
    )
    assert np.all(paths >= 0), "CIR Euler paths went negative under Feller condition"
