"""Vectorised and parallel parameter sweep utilities (Sec 5B.5).

Algebraic functions (kerr_suppression, NEC profile, etc.) use
sweep_2d_vectorised with broadcasting.

ODE-based functions (f(R) alpha sweep) use sweep_1d_parallel
with joblib for multicore speedup.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

try:
    from joblib import Parallel, delayed
    _JOBLIB = True
except ImportError:
    _JOBLIB = False
    log.warning("joblib not found — parallel sweeps will run sequentially.")


# ── Algebraic (vectorised) sweeps ─────────────────────────────────────────────

def sweep_2d_vectorised(
    param_x: str,
    range_x: tuple,
    param_y: str,
    range_y: tuple,
    base_params: dict,
    compute_fn: Callable[[dict], np.ndarray],
    nx: int = 40,
    ny: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised 2-D parameter sweep over algebraic (non-ODE) functions.

    compute_fn must accept a params dict where param_x and param_y are
    2-D arrays (ny*nx,) and return a 1-D array of the same length.

    Returns (x, y, Z) where Z.shape == (ny, nx).
    """
    x = np.linspace(*range_x, nx)
    y = np.linspace(*range_y, ny)
    X, Y = np.meshgrid(x, y)

    p = dict(base_params)
    p[param_x] = X.ravel()
    p[param_y] = Y.ravel()

    Z = np.asarray(compute_fn(p), dtype=float).reshape(ny, nx)
    return x, y, Z


def sweep_1d(
    param: str,
    param_range: tuple,
    base_params: dict,
    compute_fn: Callable[[dict], float],
    n: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple 1-D sequential sweep.

    compute_fn receives a copy of base_params with the swept param set to a
    scalar.  Returns (param_values, results).
    """
    values = np.linspace(*param_range, n)
    results = np.zeros(n)
    for i, v in enumerate(values):
        p = dict(base_params)
        p[param] = float(v)
        try:
            results[i] = float(compute_fn(p))
        except Exception:
            results[i] = float("nan")
    return values, results


# ── ODE-based parallel sweeps ─────────────────────────────────────────────────

def _safe_call(compute_fn, params):
    try:
        return float(compute_fn(params))
    except Exception:
        return float("nan")


def sweep_1d_parallel(
    param: str,
    param_range: tuple,
    base_params: dict,
    compute_fn: Callable[[dict], float],
    n: int = 40,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Parallel 1-D sweep using joblib (falls back to sequential if unavailable).

    Intended for ODE-based functions like the f(R) alpha sweep where each
    evaluation is ~100 ms and perfectly independent.

    compute_fn receives a params dict with the swept param set to a scalar.
    Returns (param_values, results).
    """
    values = np.linspace(*param_range, n)
    params_list = [{**base_params, param: float(v)} for v in values]

    if _JOBLIB:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_safe_call)(compute_fn, p) for p in params_list
        )
    else:
        results = [_safe_call(compute_fn, p) for p in params_list]

    return values, np.asarray(results, dtype=float)


def sweep_2d_parallel(
    param_x: str,
    range_x: tuple,
    param_y: str,
    range_y: tuple,
    base_params: dict,
    compute_fn: Callable[[dict], float],
    nx: int = 20,
    ny: int = 20,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel 2-D sweep for ODE-based functions.

    Returns (x, y, Z) where Z.shape == (ny, nx).
    """
    x = np.linspace(*range_x, nx)
    y = np.linspace(*range_y, ny)
    X, Y = np.meshgrid(x, y)

    params_list = []
    for xi, yi in zip(X.ravel(), Y.ravel()):
        p = dict(base_params)
        p[param_x] = float(xi)
        p[param_y] = float(yi)
        params_list.append(p)

    if _JOBLIB:
        flat = Parallel(n_jobs=n_jobs)(
            delayed(_safe_call)(compute_fn, p) for p in params_list
        )
    else:
        flat = [_safe_call(compute_fn, p) for p in params_list]

    Z = np.asarray(flat, dtype=float).reshape(ny, nx)
    return x, y, Z
