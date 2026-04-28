"""GRModel — General Relativity wormhole (Morris-Thorne, Phi=0).

Wraps physics.morris_thorne and physics.energy_conditions to satisfy
the GravityModel protocol defined in physics.model.
"""

import numpy as np

from physics.model import GravityModel, StressEnergy
from physics.morris_thorne import rho_GR, p_r_GR, p_t_GR, wormhole_conditions
from physics.energy_conditions import NEC_GR_wormhole


class GRModel:
    """General Relativity Morris-Thorne wormhole model.

    Expected params keys: r0 (float), shape (str).
    """

    name: str = "GR"

    def stress_energy(self, r: np.ndarray, params: dict) -> StressEnergy:
        """MT stress-energy on grid r.  Morris & Thorne (1988) Eq 6."""
        r0 = float(params["r0"])
        shape = str(params.get("shape", "power"))
        r = np.asarray(r, dtype=float)
        return StressEnergy(
            rho=rho_GR(r, r0, shape),
            p_r=p_r_GR(r, r0, shape),
            p_t=p_t_GR(r, r0, shape),
        )

    def nec_at_throat(self, params: dict) -> float:
        """rho + p_r at r = r0.  Always negative for traversable wormhole."""
        r0 = float(params["r0"])
        shape = str(params.get("shape", "power"))
        result = NEC_GR_wormhole(np.array([r0]), r0, shape)
        return float(result["nec_r"][0])

    def is_traversable(self, params: dict) -> bool:
        """True if b(r0)=r0, b'<1, and NEC violated at throat."""
        r0 = float(params["r0"])
        shape = str(params.get("shape", "power"))
        cond = wormhole_conditions(r0, shape)
        nec = self.nec_at_throat(params)
        return bool(cond["valid"] and nec < 0)
