"""Abstract gravity-model protocol and shared data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class StressEnergy:
    """Container for stress-energy tensor components on a radial grid."""

    rho: np.ndarray
    p_r: np.ndarray
    p_t: np.ndarray

    @property
    def nec_radial(self) -> np.ndarray:
        """rho + p_r.  Negative => NEC violated (exotic matter)."""
        return self.rho + self.p_r

    @property
    def nec_transverse(self) -> np.ndarray:
        """rho + p_t."""
        return self.rho + self.p_t

    @property
    def violated(self) -> np.ndarray:
        """Boolean array: True where radial NEC is violated."""
        return self.nec_radial < 0


@runtime_checkable
class GravityModel(Protocol):
    """Interface every gravity model must satisfy.

    Concrete implementations live in physics/models/.
    """

    name: str

    def stress_energy(self, r: np.ndarray, params: dict) -> StressEnergy:
        """Compute rho, p_r, p_t on radial grid r given params dict."""
        ...

    def nec_at_throat(self, params: dict) -> float:
        """Return rho + p_r evaluated exactly at the throat r0."""
        ...

    def is_traversable(self, params: dict) -> bool:
        """True if the wormhole satisfies throat + NEC-violation conditions."""
        ...
