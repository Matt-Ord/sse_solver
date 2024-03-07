from __future__ import annotations

import numpy as np
import pytest
import qutip

from sse_solver.sse_solver import solve_sse_euler

rng = np.random.default_rng()


@pytest.fixture()
def n_states() -> int:
    return rng.integers(1, 10)


@pytest.fixture()
def diagonal_hamiltonian(
    n_states: int,
) -> np.ndarray[tuple[int, int], np.dtype[np.complex128]]:
    return np.diag(rng.random(n_states)).astype(np.complex128)


def test_zero_time(
    diagonal_hamiltonian: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
) -> None:
    initial_state = np.zeros_like(diagonal_hamiltonian[0])
    initial_state[0] = 1
    out = solve_sse_euler(
        initial_state,
        diagonal_hamiltonian.reshape(-1),
        [],
        [],
        [],
        1,
        1,
        0,
    )
    np.testing.assert_array_equal(initial_state, out)


def test_small_time(
    diagonal_hamiltonian: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
) -> None:
    initial_state = np.zeros_like(diagonal_hamiltonian[0])
    initial_state[0] = 1
    dt = np.pi / (10000 * diagonal_hamiltonian[0, 0])
    out = solve_sse_euler(
        initial_state,
        diagonal_hamiltonian.reshape(-1),
        [],
        [],
        [],
        3,
        10000,
        dt,
    )
    out = np.array(out).reshape(3, -1)
    np.testing.assert_array_equal(initial_state, out[0])
    np.testing.assert_array_almost_equal(
        -initial_state,
        out[1],
        decimal=3,
    )
    np.testing.assert_array_almost_equal(
        initial_state,
        out[2],
        decimal=3,
    )


def test_same_as_qutip(
    diagonal_hamiltonian: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
) -> None:
    initial_state = np.zeros_like(diagonal_hamiltonian[0])
    initial_state[0] = 1
    E = np.max(diagonal_hamiltonian[0, 0]).real
    dt = np.pi / (1000 * E)
    N = 10 * 1000

    times = np.linspace(0, N * dt, N + 1, endpoint=True)

    np.divmod(np.pi / E, dt)
    out = qutip.ssesolve(
        qutip.Qobj(diagonal_hamiltonian),
        qutip.Qobj(initial_state),
        tlist=times,
        ntraj=1,
        options={
            "dt": dt,
            "method": "euler",
            "store_states": True,
            "keep_runs_results": True,
            "normalize_output": False,
        },
    )

    expected = np.array([s.full() for s in out.states[0]]).reshape(times.size, -1)

    actual_raw = solve_sse_euler(
        initial_state,
        diagonal_hamiltonian.reshape(-1),
        [],
        [],
        [],
        times.size,
        1,
        dt,
    )
    actual = np.array(actual_raw).reshape(times.size, -1)

    for i in range(times.size):
        np.testing.assert_array_almost_equal(
            expected[i],
            actual[i],
        )
