from __future__ import annotations

from typing import Self

from ._solver import (
    SimulationConfig as SimulationConfigRust,
)
from ._solver import (
    solve_sse,
    solve_sse_banded,
    solve_sse_bra_ket,
)
from ._sse_method import SSEMethod


class SimulationConfig(SimulationConfigRust):
    """Python class used to configure the Simulation."""

    def __hash__(self: Self) -> int:
        """Calculate the Hash."""
        return hash(
            (
                self.n,
                self.step,
                self.n_realizations,
                self.n_trajectories,
                self.method,
                self.dt,
            ),
        )


__all__ = [
    "SimulationConfig",
    "SSEMethod",
    "solve_sse",
    "solve_sse_banded",
    "solve_sse_bra_ket",
]
