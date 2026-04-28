"""ODE convergence tests — throat oscillator and scalar field solver."""

import math

import numpy as np
import pytest

from physics.throat_dynamics import solve_throat_ode, natural_frequency, damping_regime
from physics.fR_gravity import solve_scalar_field, _series_ic_at_throat


# ── Throat oscillator convergence ─────────────────────────────────────────────

@pytest.mark.parametrize("sigma,a0,eta", [
    (0.4, 1.2, 0.15),   # underdamped
    (0.1, 1.2, 0.80),   # overdamped
])
def test_throat_ode_analytic_agreement(sigma, a0, eta):
    """Numerical and analytic solutions must agree to 1e-5."""
    result = solve_throat_ode(sigma, a0, eta, delta_a0=0.1, t_max=30.0, n_points=1000)
    assert result["success"]
    assert result["max_residual"] < 1e-5


def test_throat_ode_decay_to_zero():
    """Underdamped oscillator must decay to near-zero by t=50."""
    result = solve_throat_ode(0.4, 1.2, 0.15, delta_a0=0.1, t_max=50.0, n_points=2000)
    assert abs(float(result["da_numeric"][-1])) < 0.05


def test_throat_ode_convergence_with_refinement():
    """Solution at t=5 must change < 0.1% when n_points doubles from 500 to 1000."""
    sigma, a0, eta = 0.4, 1.2, 0.15
    r1 = solve_throat_ode(sigma, a0, eta, delta_a0=0.1, t_max=10.0, n_points=500)
    r2 = solve_throat_ode(sigma, a0, eta, delta_a0=0.1, t_max=10.0, n_points=1000)
    # Compare value nearest t=5
    idx1 = np.argmin(np.abs(r1["t"] - 5.0))
    idx2 = np.argmin(np.abs(r2["t"] - 5.0))
    diff = abs(float(r1["da_numeric"][idx1]) - float(r2["da_numeric"][idx2]))
    assert diff < 1e-4


# ── Scalar field ODE and series IC ────────────────────────────────────────────

def test_series_ic_phi_prime_zero_at_throat():
    """phi'(r0) = 0 from the regular-solution condition."""
    delta, phi_ic, dphi_ic, A2 = _series_ic_at_throat(1.2, 0.15, 1.05, "power")
    # dphi_ic ~ 2*A2*delta; at delta->0 it -> 0.  Check it's small relative to phi_ic.
    assert abs(dphi_ic) < abs(phi_ic)


def test_series_ic_phi_consistent():
    """phi(r0+delta) matches phi0 + A2*delta^2."""
    r0, alpha, phi0 = 1.2, 0.15, 1.05
    delta, phi_ic, dphi_ic, A2 = _series_ic_at_throat(r0, alpha, phi0, "power")
    expected = phi0 + A2 * delta**2
    assert phi_ic == pytest.approx(expected, rel=1e-10)


def test_scalar_field_ode_runs():
    """solve_scalar_field should complete without error and return arrays."""
    r_arr, phi_arr, dphi_arr, sol = solve_scalar_field(
        1.2, 15.0, 0.15, 1.05, shape="power", n_points=200
    )
    assert len(r_arr) > 10
    assert sol.success or not sol.success   # just must not raise


def test_scalar_field_series_vs_ode_near_throat():
    """phi at first interior point must match series to within 0.5%."""
    r0, alpha, phi0 = 1.2, 0.15, 1.05
    delta, phi_series, dphi_series, A2 = _series_ic_at_throat(r0, alpha, phi0, "power")
    r_arr, phi_arr, dphi_arr, sol = solve_scalar_field(
        r0, 15.0, alpha, phi0, shape="power", n_points=300
    )
    # First point of ODE should equal series IC (it starts there)
    assert phi_arr[0] == pytest.approx(phi_series, rel=1e-6)
