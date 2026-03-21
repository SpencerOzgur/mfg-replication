import numpy as np
import control
from params import control_params, simulation_params


def inventory_from_control(nu_hat: np.ndarray, Q0: float, T: float, N: int) -> np.ndarray:
    dt = T / N
    q = np.empty(N + 1)
    q[0] = Q0
    q[1:] = Q0 - np.cumsum(nu_hat) * dt
    return q


def test_alpha_inventory_control_shape():
    A_hat = np.zeros(control_params.N)

    nu_hat = control.alpha_inventory_control(
        A_hat=A_hat,
        params=control_params,
        kappa=1.0,
    )

    assert isinstance(nu_hat, np.ndarray)
    assert nu_hat.shape == (control_params.N,)


def test_validate_controls_accepts_valid_shape():
    nu_hat = np.zeros(control_params.N)
    control.validate_controls(nu_hat, control_params)


def test_validate_controls_rejects_wrong_length():
    nu_hat = np.zeros(control_params.N - 1)

    try:
        control.validate_controls(nu_hat, control_params)
        assert False, "Expected ValueError for wrong control length"
    except ValueError:
        assert True


def test_inventory_recursion_from_control():
    nu_hat = np.ones(control_params.N)
    q = inventory_from_control(
        nu_hat=nu_hat,
        Q0=control_params.Q0,
        T=control_params.T,
        N=control_params.N,
    )

    dt = control_params.T / control_params.N
    for n in range(control_params.N):
        assert np.isclose(q[n + 1], q[n] - nu_hat[n] * dt)


def test_zero_signal_control_has_correct_shape_and_finite_values():
    A_hat = np.zeros(control_params.N)

    nu_hat = control.alpha_inventory_control(
        A_hat=A_hat,
        params=control_params,
        kappa=1.0,
    )

    assert np.all(np.isfinite(nu_hat))