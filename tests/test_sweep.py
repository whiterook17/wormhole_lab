"""Tests for vectorised and parallel parameter sweeps."""

import numpy as np
import pytest

from physics.kerr import kerr_suppression, tau_reduction_percent
from physics.parameter_explorer import (
    sweep_2d_vectorised, sweep_1d, sweep_1d_parallel,
)


# ── kerr_suppression vectorisation ────────────────────────────────────────────

def test_kerr_suppression_scalar_array_match():
    """Array call must match scalar calls element-wise."""
    a_arr = np.array([0.0, 0.3, 0.5, 0.9, 0.99])
    result_arr = kerr_suppression(a_arr, 1.0)
    for i, a in enumerate(a_arr):
        scalar = float(kerr_suppression(a, 1.0))
        assert float(result_arr[i]) == pytest.approx(scalar, rel=1e-10)


def test_kerr_suppression_broadcasting():
    """Both a and M as arrays of different shapes broadcast correctly."""
    a = np.linspace(0, 0.99, 10)
    result = kerr_suppression(a, 1.0)
    assert result.shape == (10,)
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


# ── sweep_2d_vectorised ───────────────────────────────────────────────────────

def _suppression_fn(p: dict) -> np.ndarray:
    return kerr_suppression(p["a"], p["M"])


def test_sweep_2d_output_shape():
    x, y, Z = sweep_2d_vectorised(
        "a", (0.0, 0.99), "M", (0.5, 2.0),
        {"a": 0.5, "M": 1.0}, _suppression_fn, nx=10, ny=8,
    )
    assert x.shape == (10,)
    assert y.shape == (8,)
    assert Z.shape == (8, 10)


def test_sweep_2d_values_in_range():
    x, y, Z = sweep_2d_vectorised(
        "a", (0.0, 0.9), "M", (0.5, 2.0),
        {"a": 0.5, "M": 1.0}, _suppression_fn, nx=6, ny=5,
    )
    assert np.all(Z >= 0.0) and np.all(Z <= 1.0)


def test_sweep_2d_matches_loop():
    """Vectorised sweep must agree with a naive double loop."""
    nx, ny = 5, 4
    x, y, Z = sweep_2d_vectorised(
        "a", (0.0, 0.8), "M", (0.5, 2.0),
        {"a": 0.5, "M": 1.0}, _suppression_fn, nx=nx, ny=ny,
    )
    for i, yi in enumerate(y):
        for j, xj in enumerate(x):
            expected = float(kerr_suppression(xj, yi))
            assert float(Z[i, j]) == pytest.approx(expected, rel=1e-8)


# ── sweep_1d and sweep_1d_parallel ────────────────────────────────────────────

def _reduction_fn(p: dict) -> float:
    return float(tau_reduction_percent(p["a"], 1.0))


def test_sweep_1d_output_shape():
    vals, results = sweep_1d("a", (0.0, 0.99), {"a": 0.5}, _reduction_fn, n=8)
    assert vals.shape == (8,) and results.shape == (8,)


def test_sweep_1d_monotone_reduction():
    vals, results = sweep_1d("a", (0.0, 0.9), {"a": 0.5}, _reduction_fn, n=10)
    for i in range(len(results) - 1):
        assert results[i] <= results[i + 1]


def test_sweep_1d_parallel_matches_sequential():
    """Parallel and sequential sweeps must produce identical results."""
    params = {"a": 0.5}
    vals_seq, res_seq = sweep_1d("a", (0.0, 0.9), params, _reduction_fn, n=8)
    vals_par, res_par = sweep_1d_parallel("a", (0.0, 0.9), params, _reduction_fn, n=8)
    np.testing.assert_allclose(vals_seq, vals_par)
    np.testing.assert_allclose(res_seq, res_par, rtol=1e-8)
