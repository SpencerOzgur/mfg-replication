import numpy as np
import control


def sample_initial_inventory(rng, center: float, scale: float) -> float:
    return rng.normal(loc=center, scale=scale)


def inventory_from_control(nu: np.ndarray, Q0: float, T: float, N: int) -> np.ndarray:
    dt = T / N
    q = np.empty(N + 1)
    q[0] = Q0
    q[1:] = Q0 - np.cumsum(nu) * dt
    return q


def simulate_agent_inventory_paths(A_hat_k, subpops, control_params, sim_params, n_agents=20, seed=42):
    rng = np.random.default_rng(seed)

    agent_inventories = []
    agent_controls = []
    q_bar_mfg = []
    q_bar_emp = []

    for k, sp in enumerate(subpops):
        q_group = np.empty((n_agents, sim_params.N + 1))
        nu_group = np.empty((n_agents, sim_params.N))

        for j in range(n_agents):
            Q0_j = sample_initial_inventory(
                rng=rng,
                center=sp.Q0,
                scale=max(1.0, 0.25 * abs(sp.Q0)),
            )

            local_params = control.ControlParams(
                T=control_params.T,
                N=control_params.N,
                Q0=Q0_j,
            )

            nu_j = control.alpha_inventory_control(
                A_hat_k[k, :-1],
                params=local_params,
                kappa=sp.kappa,
            )

            q_j = inventory_from_control(
                nu=nu_j,
                Q0=Q0_j,
                T=sim_params.T,
                N=sim_params.N,
            )

            nu_group[j] = nu_j
            q_group[j] = q_j

        rep_params = control.ControlParams(
            T=control_params.T,
            N=control_params.N,
            Q0=sp.Q0,
        )

        nu_rep = control.alpha_inventory_control(
            A_hat_k[k, :-1],
            params=rep_params,
            kappa=sp.kappa,
        )

        q_rep = inventory_from_control(
            nu=nu_rep,
            Q0=sp.Q0,
            T=sim_params.T,
            N=sim_params.N,
        )

        agent_controls.append(nu_group)
        agent_inventories.append(q_group)
        q_bar_mfg.append(q_rep)
        q_bar_emp.append(q_group.mean(axis=0))

    return {
        "agent_controls": agent_controls,
        "agent_inventories": agent_inventories,
        "q_bar_mfg": np.array(q_bar_mfg),
        "q_bar_emp": np.array(q_bar_emp),
    }