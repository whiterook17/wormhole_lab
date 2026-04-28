"""f(R) = R + alpha*R^2 gravity scalar-field analysis in Morris-Thorne background.

Reference: Sotiriou & Faraoni (2010) Rev. Mod. Phys. 82, 451.
Scalar-tensor equivalence: phi = df/dR = 1 + 2*alpha*R.

v2 changes:
  - Throat singularity handled via series expansion (Method 1, Sec 5B.3).
  - solve_scalar_field accepts an optional Solver instance.
  - shoot_phi0_with_convergence_test tests asymptotic convergence (Sec 5B.4).
"""

import logging
import warnings

import numpy as np
from scipy.integrate import solve_ivp

from physics.constants import EPS
from physics.morris_thorne import get_shape, b_prime as _b_prime

log = logging.getLogger(__name__)

# ── Background Ricci scalar in MT metric ──────────────────────────────────────

def ricci_scalar_MT(r, r0, shape):
    """Approximate Ricci scalar R from the Phi=0 Morris-Thorne metric.

    R ~ -2*b'(r)/r^2 + 2*b(r)/r^3   (leading order, Phi=0).
    Sotiriou & Faraoni (2010) Appendix A.
    """
    r = np.asarray(r, dtype=float)
    b = get_shape(shape, r, r0)
    bp = np.array([_b_prime(float(ri), r0, shape) for ri in r.flat]).reshape(r.shape)
    return -2.0 * bp / (r**2 + EPS) + 2.0 * b / (r**3 + EPS)


# ── f(R) algebra ──────────────────────────────────────────────────────────────

def f_R(R, alpha):
    """f(R) = R + alpha*R^2.  Starobinsky (1980) form."""
    R = np.asarray(R, dtype=float)
    return R + alpha * R**2


def f_prime(R, alpha):
    """df/dR = 1 + 2*alpha*R."""
    R = np.asarray(R, dtype=float)
    return 1.0 + 2.0 * alpha * R


def phi_from_R(R, alpha):
    """Scalar field phi = df/dR = 1 + 2*alpha*R."""
    return f_prime(R, alpha)


def R_from_phi(phi, alpha):
    """Invert phi = 1 + 2*alpha*R  =>  R = (phi - 1) / (2*alpha)."""
    return (np.asarray(phi, dtype=float) - 1.0) / (2.0 * alpha + EPS)


def scalar_potential(phi, alpha):
    """V(phi) = (phi - 1)^2 / (4*alpha).  Starobinsky potential in Einstein frame."""
    phi = np.asarray(phi, dtype=float)
    return (phi - 1.0)**2 / (4.0 * alpha + EPS)


def dV_dphi(phi, alpha):
    """dV/dphi = (phi - 1) / (2*alpha)."""
    phi = np.asarray(phi, dtype=float)
    return (phi - 1.0) / (2.0 * alpha + EPS)


# ── Effective stress-energy ────────────────────────────────────────────────────

def fR_effective_stress_energy(r, r0, alpha, shape):
    """Effective stress-energy from f(R) corrections to the MT wormhole.

    Returns dict: rho_eff, p_r_eff, nec_eff, R, fR.
    Approximate: treat f(R) extra terms as effective fluid on top of GR matter.
    Sotiriou & Faraoni (2010) Eq 8.
    """
    from physics.morris_thorne import rho_GR, p_r_GR

    r = np.asarray(r, dtype=float)
    R = ricci_scalar_MT(r, r0, shape)
    fR_val = f_prime(R, alpha)

    rho_base = rho_GR(r, r0, shape)
    p_r_base = p_r_GR(r, r0, shape)

    correction = (fR_val - 1.0) / (2.0 * np.abs(alpha) + EPS)

    rho_eff = rho_base + correction * np.abs(R) / (8.0 * np.pi + EPS)
    p_r_eff = p_r_base - correction * np.abs(R) / (8.0 * np.pi + EPS)
    nec_eff = rho_eff + p_r_eff

    return {
        "rho_eff": rho_eff,
        "p_r_eff": p_r_eff,
        "nec_eff": nec_eff,
        "R": R,
        "fR": fR_val,
    }


# ── Scalar field ODE ──────────────────────────────────────────────────────────

def scalar_field_ode_rhs(r, y, r0, alpha, shape):
    """RHS for phi'' in the MT metric (Phi=0 background).

    y = [phi, phi']
    Equation: (1 - b/r) phi'' + [2/r - (r*b' - b)/(2r(r-b))] phi'
              = (phi - 1)/(2*alpha)

    Sotiriou & Faraoni (2010) field equation in Brans-Dicke frame.
    """
    phi, dphi = y
    b = float(get_shape(shape, r, r0))
    bp = float(_b_prime(r, r0, shape))

    A = 1.0 - b / (r + EPS)
    if abs(A) < EPS:
        return [dphi, 0.0]

    coeff = 2.0 / r - (r * bp - b) / (2.0 * r * (r - b + EPS) + EPS)
    source = (phi - 1.0) / (2.0 * alpha + EPS)

    d2phi = (source - coeff * dphi) / A
    return [dphi, d2phi]


# ── Series expansion at the throat (Method 1, Sec 5B.3) ──────────────────────

def _series_ic_at_throat(r0, alpha, phi0, shape, delta=None):
    """Compute analytic initial conditions at r0+delta via Taylor expansion.

    Near r = r0, the regular (finite-phi'') solution satisfies:
        phi'(r0)  = 0               (regularity / symmetry condition)
        phi''(r0) = (phi0 - 1) / alpha   [derived from series: A2 = (phi0-1)/(2*alpha)]

    Series to second order:
        phi(r0 + delta)  ~= phi0 + A2 * delta^2
        phi'(r0 + delta) ~= 2*A2 * delta
    where A2 = (phi0 - 1) / (2 * alpha).

    Returns (delta_used, phi_ic, dphi_ic, A2).
    """
    if delta is None:
        delta = 1e-3 * r0

    A2 = (phi0 - 1.0) / (2.0 * alpha + EPS)
    phi_ic = phi0 + A2 * delta**2
    dphi_ic = 2.0 * A2 * delta
    return delta, phi_ic, dphi_ic, A2


def solve_scalar_field(r0, r_max, alpha, phi0, phi_prime0=None,
                       shape="power", n_points=500, solver=None,
                       use_series_ic=True):
    """Integrate the scalar field ODE from r0 to r_max.

    v2: if use_series_ic=True (default), starts from r0+delta using the
    analytic series expansion at the throat (Sec 5B.3) rather than the
    singular point r0 itself.  Issues a warning if the ODE's next-step
    derivative disagrees with the series by more than 1%.

    phi_prime0 is ignored when use_series_ic=True (series sets it).

    Returns: r_arr, phi_arr, phi_prime_arr, sol.
    """
    from physics.solvers import AutoSolver
    if solver is None:
        solver = AutoSolver()

    if use_series_ic:
        delta, phi_ic, dphi_ic, A2 = _series_ic_at_throat(r0, alpha, phi0, shape)
        r_start = r0 + delta
        y0 = [phi_ic, dphi_ic]
    else:
        r_start = r0 + 10.0 * EPS
        y0 = [phi0, phi_prime0 if phi_prime0 is not None else 0.0]

    r_eval = np.linspace(r_start, r_max, n_points)

    sol = solve_ivp(
        scalar_field_ode_rhs,
        (r_start, r_max),
        y0,
        args=(r0, alpha, shape),
        method=solver.method if solver.method != "Auto" else "RK45",
        t_eval=r_eval,
        rtol=getattr(solver, "rtol", 1e-8),
        atol=getattr(solver, "atol", 1e-10),
        dense_output=True,
    )

    # Verify series vs ODE agreement at the second evaluation point (5B.3 check).
    # Only meaningful when that point is genuinely close to the throat (within
    # 10*delta), so the O(delta^2) truncation error is negligible.
    if use_series_ic and sol.success and len(sol.t) > 1:
        delta2 = sol.t[1] - r0            # distance of second ODE point from throat
        if delta2 < 10.0 * delta:         # only check near-throat points
            dphi_series = 2.0 * A2 * delta2
            dphi_ode = float(sol.y[1][1])
            if abs(dphi_series) > 1e-12:
                rel_diff = abs(dphi_ode - dphi_series) / abs(dphi_series)
                if rel_diff > 0.01:
                    warnings.warn(
                        f"Series IC vs ODE phi' mismatch at r0+2delta: "
                        f"series={dphi_series:.4e}, ODE={dphi_ode:.4e}, "
                        f"rel_diff={rel_diff:.2%}",
                        stacklevel=2,
                    )

    return sol.t, sol.y[0], sol.y[1], sol


def shoot_phi0(r0, r_max, alpha, phi0_range=(0.9, 1.5), shape="power", n_shots=20):
    """Shooting method: find phi(r0) such that phi(r_max) -> 1.

    Returns: phi0_best, residual, r_arr, phi_arr, phi_prime_arr.
    """
    phi0_values = np.linspace(phi0_range[0], phi0_range[1], n_shots)
    best_phi0 = phi0_values[0]
    best_residual = float('inf')
    best_r = best_phi = best_dph = None

    for phi0 in phi0_values:
        try:
            r_arr, phi_arr, dphi_arr, sol = solve_scalar_field(
                r0, r_max, alpha, phi0, shape=shape
            )
            if not sol.success or len(phi_arr) == 0:
                continue
            residual = abs(float(phi_arr[-1]) - 1.0)
            if residual < best_residual:
                best_residual = residual
                best_phi0 = phi0
                best_r = r_arr
                best_phi = phi_arr
                best_dph = dphi_arr
        except Exception:
            continue

    return best_phi0, best_residual, best_r, best_phi, best_dph


# ── Asymptotic convergence test (Sec 5B.4) ────────────────────────────────────

def shoot_phi0_with_convergence_test(r0, alpha, phi0_range=(0.9, 1.5),
                                     r_max_values=(20, 40, 80), shape="power",
                                     n_shots=20):
    """Run shooting at multiple r_max values to test asymptotic convergence.

    Convergence criterion: phi0_best must not drift more than 1% across
    successive r_max doublings.

    Returns dict:
        phi0_best, converged, r_max_used, phi0_drift, all_results.
    """
    results = {}
    for rmax in r_max_values:
        try:
            phi0_b, res, r_arr, phi_arr, dphi_arr = shoot_phi0(
                r0, rmax, alpha, phi0_range=phi0_range, shape=shape, n_shots=n_shots
            )
            results[rmax] = {
                "phi0_best": phi0_b,
                "residual": res,
                "r_arr": r_arr,
                "phi_arr": phi_arr,
            }
        except Exception as exc:
            results[rmax] = {"phi0_best": None, "residual": None, "error": str(exc)}

    phi0_vals = [v["phi0_best"] for v in results.values() if v.get("phi0_best") is not None]
    if len(phi0_vals) < 2:
        return {
            "phi0_best": phi0_vals[0] if phi0_vals else None,
            "converged": False,
            "r_max_used": list(r_max_values)[-1],
            "phi0_drift": None,
            "all_results": results,
        }

    drift = max(phi0_vals) - min(phi0_vals)
    ref = abs(phi0_vals[0]) if abs(phi0_vals[0]) > 1e-10 else 1.0
    converged = drift < 0.01 * ref

    return {
        "phi0_best": phi0_vals[-1],
        "converged": converged,
        "r_max_used": list(r_max_values)[-1],
        "phi0_drift": drift,
        "all_results": results,
    }
