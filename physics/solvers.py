"""ODE solver abstraction with automatic stiffness detection.

All ODE-solving functions in the physics package accept an optional
`solver` keyword argument of type Solver.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from scipy.integrate import solve_ivp

log = logging.getLogger(__name__)


@dataclass
class SolverResult:
    """Unified result container returned by every Solver."""

    t: np.ndarray
    y: np.ndarray
    success: bool
    method_used: str
    n_steps: int
    max_residual: float
    stiffness_ratio: float | None = None


@runtime_checkable
class Solver(Protocol):
    """Interface every concrete solver must satisfy."""

    method: str
    rtol: float
    atol: float

    def solve(
        self,
        rhs: Callable,
        t_span: tuple,
        y0: list,
        t_eval: np.ndarray,
        args: tuple = (),
    ) -> SolverResult:
        ...


# ── Concrete solvers ───────────────────────────────────────────────────────────

class AdaptiveRK45Solver:
    """Explicit RK45 — good for smooth, non-stiff problems."""

    method = "RK45"
    rtol = 1e-8
    atol = 1e-10

    def __init__(self, rtol=1e-8, atol=1e-10):
        self.rtol = rtol
        self.atol = atol

    def solve(self, rhs, t_span, y0, t_eval, args=()):
        sol = solve_ivp(
            rhs, t_span, y0, method="RK45",
            t_eval=t_eval, args=args,
            rtol=self.rtol, atol=self.atol, dense_output=False,
        )
        return SolverResult(
            t=sol.t, y=sol.y[0] if sol.y.ndim > 1 else sol.y,
            success=sol.success,
            method_used="RK45",
            n_steps=sol.t.size,
            max_residual=0.0,
        )


class RadauSolver:
    """Implicit Radau — good for mildly to moderately stiff problems."""

    method = "Radau"
    rtol = 1e-8
    atol = 1e-10

    def __init__(self, rtol=1e-8, atol=1e-10):
        self.rtol = rtol
        self.atol = atol

    def solve(self, rhs, t_span, y0, t_eval, args=()):
        sol = solve_ivp(
            rhs, t_span, y0, method="Radau",
            t_eval=t_eval, args=args,
            rtol=self.rtol, atol=self.atol, dense_output=False,
        )
        return SolverResult(
            t=sol.t, y=sol.y[0] if sol.y.ndim > 1 else sol.y,
            success=sol.success,
            method_used="Radau",
            n_steps=sol.t.size,
            max_residual=0.0,
        )


class BDFSolver:
    """Implicit BDF — good for very stiff problems."""

    method = "BDF"
    rtol = 1e-8
    atol = 1e-10

    def __init__(self, rtol=1e-8, atol=1e-10):
        self.rtol = rtol
        self.atol = atol

    def solve(self, rhs, t_span, y0, t_eval, args=()):
        sol = solve_ivp(
            rhs, t_span, y0, method="BDF",
            t_eval=t_eval, args=args,
            rtol=self.rtol, atol=self.atol, dense_output=False,
        )
        return SolverResult(
            t=sol.t, y=sol.y[0] if sol.y.ndim > 1 else sol.y,
            success=sol.success,
            method_used="BDF",
            n_steps=sol.t.size,
            max_residual=0.0,
        )


def _estimate_stiffness(rhs, t0: float, y0: list, args: tuple) -> float | None:
    """Numerically estimate stiffness ratio at (t0, y0).

    Returns max(|eigenvalues|)/min(|eigenvalues|) of the Jacobian,
    or None on failure.
    """
    try:
        y0_arr = np.asarray(y0, dtype=float)
        n = len(y0_arr)
        f0 = np.asarray(rhs(t0, y0_arr, *args), dtype=float)
        h = max(1e-6 * np.max(np.abs(y0_arr) + 1e-8), 1e-8)

        J = np.zeros((n, n))
        for j in range(n):
            yp = y0_arr.copy(); yp[j] += h
            ym = y0_arr.copy(); ym[j] -= h
            J[:, j] = (np.asarray(rhs(t0, yp, *args)) -
                       np.asarray(rhs(t0, ym, *args))) / (2.0 * h)

        eigs = np.abs(np.linalg.eigvals(J))
        eigs = eigs[eigs > 1e-14]
        if len(eigs) < 2:
            return None
        return float(np.max(eigs) / np.min(eigs))
    except Exception:
        return None


class AutoSolver:
    """Detects stiffness at the first step and selects the best method.

    Stiffness ratio > 100 → Radau; otherwise RK45.
    Logs which method was chosen.
    """

    method = "Auto"
    rtol: float
    atol: float
    _stiffness_threshold: float

    def __init__(self, rtol=1e-8, atol=1e-10, stiffness_threshold=100.0):
        self.rtol = rtol
        self.atol = atol
        self._stiffness_threshold = stiffness_threshold

    def solve(self, rhs, t_span, y0, t_eval, args=()):
        ratio = _estimate_stiffness(rhs, t_span[0], y0, args)
        if ratio is not None and ratio > self._stiffness_threshold:
            chosen = "Radau"
            log.debug("AutoSolver: stiffness_ratio=%.1f > %.0f → Radau",
                      ratio, self._stiffness_threshold)
        else:
            chosen = "RK45"
            log.debug("AutoSolver: stiffness_ratio=%s → RK45", ratio)

        sol = solve_ivp(
            rhs, t_span, y0, method=chosen,
            t_eval=t_eval, args=args,
            rtol=self.rtol, atol=self.atol, dense_output=False,
        )
        return SolverResult(
            t=sol.t,
            y=sol.y[0] if sol.y.ndim > 1 else sol.y,
            success=sol.success,
            method_used=chosen,
            n_steps=sol.t.size,
            max_residual=0.0,
            stiffness_ratio=ratio,
        )


# Default instance used when callers pass no explicit solver
DEFAULT_SOLVER = AutoSolver()
