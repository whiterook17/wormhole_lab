"""FRModel — f(R) = R + alpha*R^2 wormhole model.

Wraps physics.fR_gravity to satisfy the GravityModel protocol.
"""

import numpy as np

from physics.model import GravityModel, StressEnergy
from physics.fR_gravity import fR_effective_stress_energy


class FRModel:
    """f(R) = R + alpha*R^2 wormhole model.

    Expected params keys: r0 (float), shape (str), alpha (float).
    """

    name: str = "fR"

    def stress_energy(self, r: np.ndarray, params: dict) -> StressEnergy:
        """Effective f(R) stress-energy on grid r.  Sotiriou & Faraoni (2010)."""
        r0 = float(params["r0"])
        alpha = float(params.get("alpha", 0.15))
        shape = str(params.get("shape", "power"))
        r = np.asarray(r, dtype=float)
        result = fR_effective_stress_energy(r, r0, alpha, shape)
        return StressEnergy(
            rho=result["rho_eff"],
            p_r=result["p_r_eff"],
            p_t=result["rho_eff"],   # approximate: use rho_eff as p_t proxy
        )

    def nec_at_throat(self, params: dict) -> float:
        """rho_eff + p_r_eff at r = r0."""
        r0 = float(params["r0"])
        alpha = float(params.get("alpha", 0.15))
        shape = str(params.get("shape", "power"))
        result = fR_effective_stress_energy(np.array([r0]), r0, alpha, shape)
        return float(result["nec_eff"][0])

    def is_traversable(self, params: dict) -> bool:
        """True if the effective NEC is violated at the throat."""
        return self.nec_at_throat(params) < 0
