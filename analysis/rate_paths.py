import numpy as np
import matplotlib.pyplot as plt
from models.vasicek import simulate_vasicek_exact
from models.cir import simulate_cir_exact
from models.hull_white import simulate_hull_white

def plot_rate_fan(model: str, params: dict, r0: float,
                  T: float, N_paths: int, N_steps: int,
                  percentiles: list = [5, 25, 50, 75, 95]):
    time = np.linspace(0, T, N_steps + 1)

    if model == 'vasicek': 
        paths = simulate_vasicek_exact(r0, params["kappa"], params["theta"],
                                   params["sigma"], T, N_paths, N_steps, seed=42)
    elif model == 'cir':
        paths = simulate_cir_exact(r0, params["kappa"], params["theta"],
                                   params["sigma"], T, N_paths, N_steps, seed=42)
    elif model == 'hull_white':
        dt = T / N_steps
        paths = simulate_hull_white(r0, params["a"], params["sigma"], params["theta_grid"], dt, N_paths, N_steps)
    else:
        raise ValueError(f"Unknown Model: {model}")
        
    pct_matrix = np.array([np.percentile(paths, p, axis=0) for p in percentiles])

    plt.fill_between(time, pct_matrix[0], pct_matrix[-1], alpha=0.2, label=f"{model} 5-95%")
    plt.fill_between(time, pct_matrix[1], pct_matrix[-2], alpha=0.3)
    plt.plot(time, pct_matrix[2], label=f"{model} median")
    plt.xlabel("Time (years)")
    plt.ylabel("Short rate")
    plt.title("Rate Fan Chart")
    plt.legend()


if __name__ == "__main__":
    vasicek_params = {"kappa": 0.5, "theta": 0.065, "sigma": 0.02}
    cir_params = {"kappa": 0.5, "theta": 0.065, "sigma": 0.05}

    plot_rate_fan("vasicek", vasicek_params, r0=0.065, T=5, N_paths=1000, N_steps=252)
    plot_rate_fan("cir", cir_params, r0=0.065, T=5, N_paths=1000, N_steps=252)

    plt.tight_layout()
    plt.show()

