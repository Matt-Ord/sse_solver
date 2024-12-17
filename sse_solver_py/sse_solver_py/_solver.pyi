from typing import Self

from ._sse_method import SSEMethod

def solve_sse(
    initial_state: list[complex],
    hamiltonian: list[list[complex]],
    noise_operators: list[list[list[complex]]],
    config: SimulationConfig,
) -> list[complex]: ...
def solve_sse_banded(
    initial_state: list[complex],
    hamiltonian: BandedData,
    noise_operators: list[BandedData],
    config: SimulationConfig,
) -> list[complex]: ...
def solve_sse_split_operator(
    initial_state: list[complex],
    hamiltonian: SplitOperatorData,
    noise_operators: list[SplitOperatorData],
    config: SimulationConfig,
) -> list[complex]: ...
def solve_sse_measured_split_operator(
    initial_state: list[complex],
    hamiltonian: SplitOperatorData,
    noise_operators: list[SplitOperatorData],
    measurement_operators: list[SplitOperatorData],
    config: SimulationConfig,
) -> list[complex]: ...

class SimulationConfig:
    def __init__[_A: float | None, _B: float | None](  # noqa: PLR0913
        self: SimulationConfig,
        *,
        n: int,
        step: int,
        dt: float,
        delta: tuple[_A, float, _B] | None = None,
        n_trajectories: int = 1,
        n_realizations: int = 1,
        method: SSEMethod,
    ) -> None: ...

    n: int
    step: int
    dt: float
    n_trajectories: int
    n_realizations: int
    @property
    def method(self: Self) -> SSEMethod: ...

class BandedData:
    def __init__(
        self,
        *,
        diagonals: list[list[complex]],
        offsets: list[int],
        shape: tuple[int, int],
    ) -> None: ...
    @property
    def diagonals(self: Self) -> SSEMethod: ...
    @property
    def offsets(self: Self) -> list[int]: ...
    @property
    def shape(self: Self) -> tuple[int, int]: ...

class SplitOperatorData:
    def __init__(
        self,
        *,
        a: list[complex] | None = None,
        b: list[complex] | None = None,
        c: list[complex],
        d: list[complex] | None = None,
    ) -> None: ...
    @property
    def a(self: Self) -> list[complex] | None: ...
    @property
    def b(self: Self) -> list[complex] | None: ...
    @property
    def c(self: Self) -> list[complex]: ...
    @property
    def d(self: Self) -> list[complex] | None: ...
