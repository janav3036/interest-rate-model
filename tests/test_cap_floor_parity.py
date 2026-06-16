import numpy as np
import pytest
from pricing.caps_floors import cap_price, floor_price, par_swap_rate


def make_schedule(start: float, end: float, freq: float):
    times = np.arange(start, end + freq / 2, freq)
    schedule = []
    for i in range(len(times) - 1):
        schedule.append((times[i], times[i+1], freq))
    return schedule


def flat_discount_fn(rate: float):
    return lambda t: np.exp(-rate * t)


SCHEDULE = make_schedule(0.0, 5.0, 0.5)
DISC_FN   = flat_discount_fn(0.065)
SIGMA     = 0.20


def test_cap_floor_parity_at_par():
    K_par = par_swap_rate(SCHEDULE, DISC_FN)
    cap   = cap_price(SCHEDULE, K_par, SIGMA, DISC_FN)
    floor = floor_price(SCHEDULE, K_par, SIGMA, DISC_FN)

    # cap - floor = swap value = P(0, T_start) - P(0, T_end) - K * annuity
    T_start = SCHEDULE[0][0]
    T_end   = SCHEDULE[-1][1]
    annuity = sum(tau * DISC_FN(T_end_i)
                  for _, T_end_i, tau in SCHEDULE)
    swap_value = DISC_FN(T_start) - DISC_FN(T_end) - K_par * annuity

    assert abs((cap - floor) - swap_value) < 1e-6


def test_cap_floor_parity_off_par():
    K = 0.08
    cap   = cap_price(SCHEDULE, K, SIGMA, DISC_FN)
    floor = floor_price(SCHEDULE, K, SIGMA, DISC_FN)

    T_start = SCHEDULE[0][0]
    T_end   = SCHEDULE[-1][1]
    annuity = sum(tau * DISC_FN(T_end_i)
                  for _, T_end_i, tau in SCHEDULE)
    swap_value = DISC_FN(T_start) - DISC_FN(T_end) - K * annuity

    assert abs((cap - floor) - swap_value) < 1e-6


def test_cap_increases_with_strike_decreases():
    caps = [cap_price(SCHEDULE, K, SIGMA, DISC_FN)
            for K in [0.05, 0.065, 0.08]]
    assert caps[0] > caps[1] > caps[2]


def test_floor_increases_with_strike():
    floors = [floor_price(SCHEDULE, K, SIGMA, DISC_FN)
              for K in [0.05, 0.065, 0.08]]
    assert floors[0] < floors[1] < floors[2]


