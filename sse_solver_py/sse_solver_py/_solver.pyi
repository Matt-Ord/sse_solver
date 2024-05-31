from ._sse_method import SSEMethod

def solve_sse(
    initial_state: list[complex],
    hamiltonian: list[list[complex]],
    operators: list[list[list[complex]]],
    config: SimulationConfig,
) -> list[complex]: ...
def solve_sse_bra_ket(  # noqa: PLR0913
    initial_state: list[complex],
    hamiltonian: list[complex],
    amplitudes: list[complex],
    bra: list[complex],
    ket: list[complex],
    config: SimulationConfig,
) -> list[complex]: ...
def solve_sse_banded(  # noqa: PLR0913
    initial_state: list[complex],
    hamiltonian_diagonal: list[list[complex]],
    hamiltonian_offset: list[int],
    operators_diagonals: list[list[list[complex]]],
    operators_offsets: list[list[int]],
    config: SimulationConfig,
) -> list[complex]: ...

class SimulationConfig:
    def __init__(  # noqa: PLR0913
        self: SimulationConfig,
        *,
        n: int,
        step: int,
        dt: float,
        n_trajectories: int = 1,
        method: SSEMethod,
    ) -> None: ...
