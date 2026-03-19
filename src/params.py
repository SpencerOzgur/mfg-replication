import numpy as np
from dataclasses import dataclass

@dataclass
class SimulationParams:
    T: float = 1.0
    dt: float = 0.0001
    NSTEPS = T / dt

    time_grid = np.linspace(0, T, NSTEPS)
