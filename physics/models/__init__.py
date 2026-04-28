"""Concrete gravity-model implementations satisfying the GravityModel protocol."""

from physics.models.general_relativity import GRModel
from physics.models.fR_gravity import FRModel
from physics.models.gauss_bonnet import GaussBonnetModel

MODEL_REGISTRY: dict = {
    "GR": GRModel(),
    "fR": FRModel(),
    "GB": GaussBonnetModel(),
}
