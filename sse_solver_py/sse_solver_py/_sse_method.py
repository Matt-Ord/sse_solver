from typing import Literal

SSEMethod = Literal[
    "Euler",
    "NormalizedEuler",
    "Milsten",
    "Order2ExplicitWeak",
    "Order2ExplicitWeakR5",
]
