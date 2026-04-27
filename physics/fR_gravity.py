"""f(R) = R + alpha*R^2 gravity scalar-field analysis in Morris-Thorne background.

Reference: Sotiriou & Faraoni (2010) Rev. Mod. Phys. 82, 451.
Scalar-tensor equivalence: phi = df/dR = 1 + 2*alpha*R.
"""

import numpy as np
from scipy.integrate import solve_ivp
from physics.constants import EPS
from physics.morris_thorne import get_shape, b_prime as _b_prime


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


def solve_scalar_field(r0, r_max, alpha, phi0, phi_prime0=0.0,
                       shape="power", n_points=500):
    """RK45 integrate phi ODE from r0+eps to r_max.

    Returns: r_arr, phi_arr, phi_prime_arr, sol.
    """
    r_start = r0 + 10.0 * EPS
    r_eval = np.linspace(r_start, r_max, n_points)

    sol = solve_ivp(
        scalar_field_ode_rhs,
        (r_start, r_max),
        [phi0, phi_prime0],
        args=(r0, alpha, shape),
        method="RK45",
        t_eval=r_eval,
        rtol=1e-8,
        atol=1e-10,
        dense_output=True,
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
