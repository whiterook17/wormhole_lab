"""Tests for series-expansion throat IC vs full ODE, and oscillator regimes."""

import math

import numpy as np
import pytest

from physics.throat_dynamics import (
    natural_frequency, damping_regime, throat_displacement_analytic,
    solve_throat_ode, echo_frequency, echo_spectrum,
)
from physics.fR_gravity import _series_ic_at_throat, solve_scalar_field


# ── Throat oscillator regimes ─────────────────────────────────────────────────

def test_underdamped_regime():
    regime, omega_d = damping_regime(0.4, 1.2, 0.15)
    assert regime == "UNDERDAMPED"
    assert omega_d > 0.0


def test_overdamped_regime():
    regime, omega_d = damping_regime(0.1, 1.2, 0.8)
    assert regime == "OVERDAMPED"
    assert omega_d == 0.0


def test_critical_regime():
    omega0 = natural_frequency(0.4, 1.2)
    regime, omega_d = damping_regime(0.4, 1.2, omega0)
    assert regime == "CRITICAL"


def test_underdamped_oscillates():
    t = np.linspace(0, 20, 500)
    da = throat_displacement_analytic(t, 0.1, 0.4, 1.2, 0.15)
    # Must cross zero at least once
    sign_changes = np.sum(np.diff(np.sign(da)) != 0)
    assert sign_changes >= 2


def test_overdamped_monotone():
    t = np.linspace(0, 10, 200)
    da = throat_displacement_analytic(t, 0.1, 0.1, 1.2, 0.8)
    # Overdamped: no sign changes after t=0
    assert np.all(da >= 0.0)


# ── Series expansion IC for scalar field ─────────────────────────────────────

def test_series_ic_delta_proportional():
    """Doubling delta should double phi'(r0+delta) (linear in delta)."""
    r0, alpha, phi0 = 1.2, 0.15, 1.05
    _, _, dphi1, _ = _series_ic_at_throat(r0, alpha, phi0, "power", delta=1e-3)
    _, _, dphi2, _ = _series_ic_at_throat(r0, alpha, phi0, "power", delta=2e-3)
    assert dphi2 == pytest.approx(2.0 * dphi1, rel=1e-8)


def test_series_ic_phi_quadratic():
    """phi(r0+delta) - phi0 should scale as delta^2."""
    r0, alpha, phi0 = 1.2, 0.15, 1.05
    _, phi1, _, _ = _series_ic_at_throat(r0, alpha, phi0, "power", delta=1e-3)
    _, phi2, _, _ = _series_ic_at_throat(r0, alpha, phi0, "power", delta=2e-3)
    ratio = (phi2 - phi0) / (phi1 - phi0)
    assert ratio == pytest.approx(4.0, rel=1e-6)


def test_ode_starts_at_series_point():
    """First ODE r-point should equal r0+delta from series."""
    r0, alpha, phi0 = 1.2, 0.15, 1.05
    delta_expected = 1e-3 * r0
    r_arr, phi_arr, _, _ = solve_scalar_field(
        r0, 15.0, alpha, phi0, shape="power", n_points=200, use_series_ic=True
    )
    assert r_arr[0] == pytest.approx(r0 + delta_expected, rel=1e-6)


def test_ode_first_phi_matches_series():
    """phi at first ODE point should match series prediction."""
    r0, alpha, phi0 = 1.2, 0.15, 1.05
    delta, phi_series, _, _ = _series_ic_at_throat(r0, alpha, phi0, "power")
    r_arr, phi_arr, _, _ = solve_scalar_field(
        r0, 15.0, alpha, phi0, shape="power", n_points=200, use_series_ic=True
    )
    assert float(phi_arr[0]) == pytest.approx(phi_series, rel=1e-6)


# ── Echo spectrum ─────────────────────────────────────────────────────────────

def test_echo_spectrum_zero_at_f_equals_zero():
    f0 = echo_frequency(0.4, 1.2)
    assert float(echo_spectrum(0.0, 1.0, f0, 0.15)) == pytest.approx(0.0)


def test_echo_spectrum_positive():
    f0 = echo_frequency(0.4, 1.2)
    assert float(echo_spectrum(f0, 1.0, f0, 0.15)) > 0.0


def test_echo_spectrum_falls_at_high_frequency():
    f0 = echo_frequency(0.4, 1.2)
    h_peak = float(echo_spectrum(f0, 1.0, f0, 0.15))
    h_high = float(echo_spectrum(50.0 * f0, 1.0, f0, 0.15))
    assert h_high < h_peak
