"""GaussBonnetModel — placeholder for Gauss-Bonnet gravity wormholes.

Not yet implemented.  Satisfies the GravityModel protocol interface so it
can be registered and tested for protocol compliance without crashing the app.
"""

import numpy as np

from physics.model import StressEnergy


class GaussBonnetModel:
    """Stub: Gauss-Bonnet gravity wormhole model.

    Reference: Kanti et al. (2011) Phys. Rev. D 85, 044010.
    Implementation pending — all methods raise NotImplementedError.
    """

    name: str = "Gauss-Bonnet"

    def stress_energy(self, r: np.ndarray, params: dict) -> StressEnergy:
        raise NotImplementedError(
            "Gauss-Bonnet stress-energy not yet implemented. "
            "See Kanti et al. (2011) for the required Riemann-tensor corrections."
        )

    def nec_at_throat(self, params: dict) -> float:
        raise NotImplementedError("Gauss-Bonnet model is a stub.")

    def is_traversable(self, params: dict) -> bool:
        raise NotImplementedError("Gauss-Bonnet model is a stub.")
