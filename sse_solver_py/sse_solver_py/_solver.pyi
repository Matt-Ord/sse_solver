def solve_sse_euler(  # noqa: PLR0913
    initial_state: list[complex],
    hamiltonian: list[list[complex]],
    operators: list[list[list[complex]]],
    n: int,
    step: int,
    dt: float,
) -> list[complex]: ...
def solve_sse_euler_bra_ket(  # noqa: PLR0913
    initial_state: list[complex],
    hamiltonian: list[complex],
    amplitudes: list[complex],
    bra: list[complex],
    ket: list[complex],
    n: int,
    step: int,
    dt: float,
) -> list[complex]: ...
def solve_sse_euler_banded(  # noqa: PLR0913
    initial_state: list[complex],
    hamiltonian_diagonal: list[list[complex]],
    hamiltonian_offset: list[int],
    operators_diagonals: list[list[list[complex]]],
    operators_offsets: list[list[int]],
    n: int,
    step: int,
    dt: float,
) -> list[complex]: ...
def solve_sse_milsten_banded(  # noqa: PLR0913
    initial_state: list[complex],
    hamiltonian_diagonal: list[list[complex]],
    hamiltonian_offset: list[int],
    operators_diagonals: list[list[list[complex]]],
    operators_offsets: list[list[int]],
    n: int,
    step: int,
    dt: float,
) -> list[complex]: ...
