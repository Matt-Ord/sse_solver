from __future__ import annotations

from ._solver import (
    BandedData,
    HarmonicLangevinSystemParameters,
    SimulationConfig,
    SplitOperatorData,
    solve_harmonic_langevin,
    solve_harmonic_quantum_langevin,
    solve_simple_stochastic,
    solve_sse,
    solve_sse_banded,
    solve_sse_measured_split_operator,
)
from ._solver import (
    solve_sse_split_operator as solve_sse_state_split_operator,
)
from ._sse_method import SSEMethod


def solve_sse_split_operator(
    initial_state: list[complex],
    hamiltonian: SplitOperatorData,
    noise_operators: list[SplitOperatorData],
    config: SimulationConfig,
    *,
    measurement_operators: list[SplitOperatorData] | None = None,
) -> list[complex]:
    if measurement_operators is None:
        return solve_sse_state_split_operator(
            initial_state,
            hamiltonian,
            noise_operators,
            config,
        )
    return solve_sse_measured_split_operator(
        initial_state,
        hamiltonian,
        noise_operators,
        measurement_operators,
        config,
    )


__all__ = [
    "BandedData",
    "HarmonicLangevinSystemParameters",
    "SSEMethod",
    "SimulationConfig",
    "SplitOperatorData",
    "solve_harmonic_langevin",
    "solve_harmonic_quantum_langevin",
    "solve_simple_stochastic",
    "solve_sse",
    "solve_sse_banded",
    "solve_sse_split_operator",
]
