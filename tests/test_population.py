import numpy as np
import population
import latent
import simulate
import filtering
from params import (
    latent_params,
    simulation_params,
    control_params,
    SubPopParams,
)


def make_test_subpops():
    return [
        SubPopParams(
            name="SubPop1",
            weight=0.5,
            prior=0.8,
            Q0=100.0,
            kappa=0.5,
        ),
        SubPopParams(
            name="SubPop2",
            weight=0.5,
            prior=0.2,
            Q0=100.0,
            kappa=2.0,
        ),
    ]


def test_population_shapes():
    subpops = make_test_subpops()
    K = len(subpops)

    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )

    A_hat_k = np.empty((K, simulation_params.N + 1))
    for i, sp in enumerate(subpops):
        pi = filtering.filter_fundamental_prob_state_1(
            F_t=F_t,
            latent_params=latent_params,
            sim_params=simulation_params,
            prior=[1.0 - sp.prior, sp.prior],
        )
        A_hat_k[i] = pi * simulation_params.A1 + (1.0 - pi) * simulation_params.A0

    results = population.simulate_agent_inventory_paths(
        A_hat_k=A_hat_k,
        subpops=subpops,
        control_params=control_params,
        sim_params=simulation_params,
        n_agents=10,
        seed=42,
    )

    assert len(results["agent_inventories"]) == K
    assert len(results["agent_controls"]) == K
    assert results["q_bar_mfg"].shape == (K, simulation_params.N + 1)
    assert results["q_bar_emp"].shape == (K, simulation_params.N + 1)

    for k in range(K):
        assert results["agent_inventories"][k].shape == (10, simulation_params.N + 1)
        assert results["agent_controls"][k].shape == (10, simulation_params.N)


def test_population_empirical_mean_matches_manual_average():
    subpops = make_test_subpops()
    K = len(subpops)

    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )

    A_hat_k = np.empty((K, simulation_params.N + 1))
    for i, sp in enumerate(subpops):
        pi = filtering.filter_fundamental_prob_state_1(
            F_t=F_t,
            latent_params=latent_params,
            sim_params=simulation_params,
            prior=[1.0 - sp.prior, sp.prior],
        )
        A_hat_k[i] = pi * simulation_params.A1 + (1.0 - pi) * simulation_params.A0

    results = population.simulate_agent_inventory_paths(
        A_hat_k=A_hat_k,
        subpops=subpops,
        control_params=control_params,
        sim_params=simulation_params,
        n_agents=10,
        seed=42,
    )

    for k in range(K):
        manual_mean = results["agent_inventories"][k].mean(axis=0)
        assert np.allclose(manual_mean, results["q_bar_emp"][k])


def test_population_terminal_inventories_are_close_to_zero_if_model_forces_liquidation():
    subpops = make_test_subpops()
    K = len(subpops)

    latent_path = latent.simulate_latent_path(params=latent_params)
    F_t = simulate.simulate_fundamental_path(
        latent_path=latent_path,
        params=simulation_params,
    )

    A_hat_k = np.empty((K, simulation_params.N + 1))
    for i, sp in enumerate(subpops):
        pi = filtering.filter_fundamental_prob_state_1(
            F_t=F_t,
            latent_params=latent_params,
            sim_params=simulation_params,
            prior=[1.0 - sp.prior, sp.prior],
        )
        A_hat_k[i] = pi * simulation_params.A1 + (1.0 - pi) * simulation_params.A0

    results = population.simulate_agent_inventory_paths(
        A_hat_k=A_hat_k,
        subpops=subpops,
        control_params=control_params,
        sim_params=simulation_params,
        n_agents=10,
        seed=42,
    )

    for k in range(K):
        terminal_vals = results["agent_inventories"][k][:, -1]
        assert np.all(np.abs(terminal_vals) < 1e-6)