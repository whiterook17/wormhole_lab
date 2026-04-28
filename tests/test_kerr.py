"""Unit tests for physics/kerr.py — every public function."""

import math

import numpy as np
import pytest

from physics.kerr import (
    sigma_BL, delta, frame_dragging, ergosphere_radius,
    horizon_radii, isco_radius, kerr_suppression,
    tau_static, tau_kerr, tau_reduction_percent,
)


# ── sigma_BL ──────────────────────────────────────────────────────────────────

def test_sigma_BL_equatorial():
    assert sigma_BL(5.0, math.pi / 2, 1.0) == pytest.approx(25.0)


def test_sigma_BL_pole():
    # theta=0: sigma = r^2 + a^2
    assert sigma_BL(3.0, 0.0, 2.0) == pytest.approx(9.0 + 4.0)


def test_sigma_BL_vectorised():
    r = np.array([2.0, 3.0, 4.0])
    result = sigma_BL(r, math.pi / 2, 1.0)
    expected = r**2
    np.testing.assert_allclose(result, expected)


# ── delta ─────────────────────────────────────────────────────────────────────

def test_delta_schwarzschild():
    # Schwarzschild: delta = r^2 - 2r => zero at r=2 for M=1
    assert delta(2.0, 1.0, 0.0) == pytest.approx(0.0)


def test_delta_extremal_horizon():
    # Extremal Kerr a=M=1: delta = r^2 - 2r + 1 = (r-1)^2
    assert delta(1.0, 1.0, 1.0) == pytest.approx(0.0)


# ── kerr_suppression ──────────────────────────────────────────────────────────

def test_kerr_suppression_zero_spin():
    assert float(kerr_suppression(0.0, 1.0)) == pytest.approx(1.0)


def test_kerr_suppression_near_extremal():
    assert float(kerr_suppression(0.9999, 1.0)) < 0.02


def test_kerr_suppression_monotone():
    sups = [float(kerr_suppression(a, 1.0)) for a in [0.0, 0.3, 0.6, 0.9, 0.99]]
    for i in range(len(sups) - 1):
        assert sups[i] > sups[i + 1]


def test_kerr_suppression_zero_mass():
    # M=0 should return 1 (no suppression — undefined physically, safe fallback)
    assert float(kerr_suppression(0.5, 0.0)) == pytest.approx(1.0)


def test_kerr_suppression_array_input():
    a_arr = np.array([0.0, 0.5, 0.99])
    result = kerr_suppression(a_arr, 1.0)
    assert result.shape == (3,)
    assert float(result[0]) == pytest.approx(1.0)
    assert float(result[2]) < 0.2


def test_kerr_suppression_broadcasting():
    # Both a and M as arrays
    a_arr = np.array([0.0, 0.5])
    M_arr = np.array([1.0, 1.0])
    result = kerr_suppression(a_arr, M_arr)
    assert result.shape == (2,)


# ── frame_dragging ────────────────────────────────────────────────────────────

def test_frame_dragging_zero_spin():
    assert float(frame_dragging(5.0, math.pi / 2, 1.0, 0.0)) == pytest.approx(0.0)


def test_frame_dragging_large_r_decay():
    w_near = float(frame_dragging(5.0, math.pi / 2, 1.0, 0.9))
    w_far = float(frame_dragging(1e5, math.pi / 2, 1.0, 0.9))
    assert w_far < w_near
    assert abs(w_far) < 1e-4


def test_frame_dragging_positive_spin():
    w = float(frame_dragging(5.0, math.pi / 2, 1.0, 0.5))
    assert w > 0.0


# ── ergosphere_radius ─────────────────────────────────────────────────────────

def test_ergosphere_schwarzschild():
    # a=0: r_erg = M + sqrt(M^2) = 2M
    assert float(ergosphere_radius(math.pi / 2, 1.0, 0.0)) == pytest.approx(2.0)


def test_ergosphere_outside_horizon():
    r_erg = float(ergosphere_radius(math.pi / 2, 1.0, 0.5))
    r_plus, _ = horizon_radii(1.0, 0.5)
    assert r_erg >= r_plus


# ── horizon_radii ─────────────────────────────────────────────────────────────

def test_horizon_schwarzschild():
    r_plus, r_minus = horizon_radii(1.0, 0.0)
    assert r_plus == pytest.approx(2.0)
    assert r_minus == pytest.approx(0.0)


def test_horizon_extremal():
    r_plus, r_minus = horizon_radii(1.0, 1.0)
    assert r_plus == pytest.approx(1.0)
    assert r_minus == pytest.approx(1.0)


def test_horizon_naked_singularity():
    r_plus, _ = horizon_radii(1.0, 1.1)
    assert math.isnan(r_plus)


# ── tau functions ─────────────────────────────────────────────────────────────

def test_tau_kerr_zero_spin_matches_static():
    r0 = 1.5
    assert tau_kerr(r0, 1.0, 0.0) == pytest.approx(tau_static(r0))


def test_tau_kerr_less_than_static_at_high_spin():
    r0 = 1.2
    assert tau_kerr(r0, 1.0, 0.99) < tau_static(r0)


def test_reduction_percent_high_spin():
    pct = tau_reduction_percent(0.99, 1.0)
    assert pct > 85.0
