def solve_sse_euler(
    initial_state: list[complex],
    hamiltonian: list[list[complex]],
    operators: list[list[list[complex]]],
    n: int,
    step: int,
    dt: float,
) -> list[complex]: ...
def solve_sse_euler_bra_ket(
    initial_state: list[complex],
    hamiltonian: list[complex],
    amplitudes: list[complex],
    bra: list[complex],
    ket: list[complex],
    n: int,
    step: int,
    dt: float,
) -> list[complex]: ...
