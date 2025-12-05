from typing import Literal

SSEMethod = Literal[
    "Euler",
    "NormalizedEuler",
    "Milstein",
    "Order2ExplicitWeak",
    "NormalizedOrder2ExplicitWeak",
    "Order2ExplicitWeakR5",
    "NormalizedOrder2ExplicitWeakR5",
]
