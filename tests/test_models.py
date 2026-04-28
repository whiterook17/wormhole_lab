"""Tests that every registered GravityModel satisfies the protocol."""

import numpy as np
import pytest

from physics.model import GravityModel, StressEnergy
from physics.models import MODEL_REGISTRY
from physics.models.general_relativity import GRModel
from physics.models.fR_gravity import FRModel
from physics.models.gauss_bonnet import GaussBonnetModel


GR_PARAMS  = {"r0": 1.2, "shape": "power"}
FR_PARAMS  = {"r0": 1.2, "shape": "power", "alpha": 0.15}
GB_PARAMS  = {"r0": 1.2}


# ── Protocol compliance ───────────────────────────────────────────────────────

def test_all_models_in_registry():
    assert "GR" in MODEL_REGISTRY
    assert "fR" in MODEL_REGISTRY
    assert "GB" in MODEL_REGISTRY


def test_all_models_implement_protocol():
    for name, model in MODEL_REGISTRY.items():
        assert isinstance(model, GravityModel), f"{name} does not satisfy GravityModel"


def test_all_models_have_name():
    for name, model in MODEL_REGISTRY.items():
        assert isinstance(model.name, str) and model.name, f"{name} has empty .name"


# ── GRModel ───────────────────────────────────────────────────────────────────

def test_gr_nec_at_throat_negative():
    gr = GRModel()
    nec = gr.nec_at_throat(GR_PARAMS)
    assert nec < 0.0


def test_gr_is_traversable():
    gr = GRModel()
    assert gr.is_traversable(GR_PARAMS) is True


def test_gr_stress_energy_returns_StressEnergy():
    gr = GRModel()
    r = np.linspace(1.2, 6.0, 50)
    se = gr.stress_energy(r, GR_PARAMS)
    assert isinstance(se, StressEnergy)
    assert se.rho.shape == (50,)


def test_gr_stress_energy_finite():
    gr = GRModel()
    r = np.linspace(1.2, 6.0, 50)
    se = gr.stress_energy(r, GR_PARAMS)
    assert np.all(np.isfinite(se.rho))
    assert np.all(np.isfinite(se.p_r))


def test_gr_nec_radial_violated_at_throat():
    gr = GRModel()
    r = np.array([1.2])
    se = gr.stress_energy(r, GR_PARAMS)
    assert float(se.nec_radial[0]) < 0.0


# ── FRModel ───────────────────────────────────────────────────────────────────

def test_fr_is_traversable():
    fr = FRModel()
    assert fr.is_traversable(FR_PARAMS) is True


def test_fr_stress_energy_returns_StressEnergy():
    fr = FRModel()
    r = np.linspace(1.2, 6.0, 50)
    se = fr.stress_energy(r, FR_PARAMS)
    assert isinstance(se, StressEnergy)
    assert se.rho.shape == (50,)


def test_fr_nec_at_throat_type():
    fr = FRModel()
    nec = fr.nec_at_throat(FR_PARAMS)
    assert isinstance(nec, float)


# ── GaussBonnetModel (stub) ───────────────────────────────────────────────────

def test_gb_raises_not_implemented_for_stress_energy():
    gb = GaussBonnetModel()
    with pytest.raises(NotImplementedError):
        gb.stress_energy(np.array([1.2]), GB_PARAMS)


def test_gb_raises_not_implemented_for_nec():
    gb = GaussBonnetModel()
    with pytest.raises(NotImplementedError):
        gb.nec_at_throat(GB_PARAMS)


def test_gb_implements_protocol():
    assert isinstance(GaussBonnetModel(), GravityModel)


# ── StressEnergy dataclass ────────────────────────────────────────────────────

def test_stress_energy_nec_properties():
    se = StressEnergy(
        rho=np.array([-0.1, -0.2]),
        p_r=np.array([0.05, 0.10]),
        p_t=np.array([0.03, 0.06]),
    )
    np.testing.assert_allclose(se.nec_radial, [-0.05, -0.10])
    np.testing.assert_allclose(se.nec_transverse, [-0.07, -0.14])
    assert np.all(se.violated)
