def solve_sse_euler(
    initial_state: list[complex],
    hamiltonian: list[complex],
    amplitudes: list[complex],
    bra: list[complex],
    ket: list[complex],
    n: int,
    step: int,
    dt: float,
) -> list[complex]: ...
