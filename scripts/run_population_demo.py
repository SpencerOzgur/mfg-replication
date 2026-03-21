from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import latent
import simulate
import filtering
import plotting
import population
import control
from params import latent_params, simulation_params, control_params, SubPopParams


def main():
    np.random.seed(42)

    subpop1 = SubPopParams(
        name="SubPop1",
        weight=0.5,
        prior=0.8,
        Q0=100.0,
        kappa=0.5,
    )

    subpop2 = SubPopParams(
        name="SubPop2",
        weight=0.5,
        prior=0.2,
        Q0=0.0,
        kappa=2.0,
    )

    subpops = [subpop1, subpop2]
    K = len(subpops)

    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )

    pi_k = np.empty((K, simulation_params.N + 1))
    A_hat_k = np.empty((K, simulation_params.N + 1))

    for i, sp in enumerate(subpops):
        pi_k[i] = filtering.filter_fundamental_prob_state_1(
            F_t=F_t,
            latent_params=latent_params,
            sim_params=simulation_params,
            prior=[1.0 - sp.prior, sp.prior],
        )

        A_hat_k[i] = (
            pi_k[i] * simulation_params.A1
            + (1.0 - pi_k[i]) * simulation_params.A0
        )

    pop_results = population.simulate_agent_inventory_paths(
        A_hat_k=A_hat_k,
        subpops=subpops,
        control_params=control_params,
        sim_params=simulation_params,
        n_agents=20,
        seed=42,
    )

    plotting.plot_individual_vs_mean_inventory(
        agent_inventories=pop_results["agent_inventories"],
        q_bar_mfg=pop_results["q_bar_mfg"],
        q_bar_emp=pop_results["q_bar_emp"],
        sim_params=simulation_params,
        subpops=subpops,
    )


if __name__ == "__main__":
    main()