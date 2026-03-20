import latent
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SimulationParams:
    """
    Parameters for a simulation.
    """
    T: float
    N: int
    sigma: float
    A1: float
    A0: float
    S0: float
    lambda_: float

def latent_to_drift(latent_path:np.ndarray, params:SimulationParams) -> np.ndarray:
    """
    :param latent_path: simulated latent path
    :param params: simulation parameters
    :return: latent drift
    """
    latent_path = np.asarray(latent_path, dtype=np.int8)
    if not np.all(np.isin(latent_path, [0, 1])):
        raise ValueError("latent_path must consist of 0 and 1")
    return np.where(latent_path == 0, params.A0, params.A1)

def simulate_fundamental_path(latent_path:np.ndarray, params:SimulationParams) -> np.ndarray:
    """
    :param latent_path: simulated latent path
    :param params: simulation parameters
    :return: Stock price movement w/o price impact
    """
    if params.T <= 0:
        raise ValueError("T must be positive")
    if params.N <= 0:
        raise ValueError('N must be positive')

    dt = params.T / params.N
    drift = latent_to_drift(latent_path, params)
    sim_path = np.empty(params.N + 1, dtype=np.float64)
    sim_path[0] = params.S0
    for i in range(params.N):
        Z = np.random.normal()
        sim_path[i + 1] = sim_path[i] + drift[i] * dt + params.sigma * np.sqrt(dt) * Z

    return sim_path

def simulate_impacted_price(F_t:np.ndarray, params:SimulationParams) -> np.ndarray:
    """
    :param F_t: Unimpacted Stock Price
    :param params: Simulation parameters
    :return: Impacted Stock Price
    """

    dt = params.T / params.N
    t_grid = np.linspace(0, params.T, params.N + 1)
    nu_hat = np.sin(t_grid) # Placeholder
    cumulative_impact = np.zeros(params.N + 1)
    cumulative_impact[1:] = np.cumsum(nu_hat[:-1]) * dt
    total_impact = params.lambda_ * cumulative_impact
    return F_t + total_impact