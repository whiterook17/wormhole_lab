"""Null Energy Condition checks for wormhole matter content.

Reference: Visser (1995) Lorentzian Wormholes; Morris & Thorne (1988).
"""

import numpy as np
from physics.constants import EPS
from physics.morris_thorne import rho_GR, p_r_GR, p_t_GR, get_shape, b_prime
from physics.kerr import kerr_suppression, tau_static, tau_kerr


# ── Basic NEC combiners ────────────────────────────────────────────────────────

def NEC_radial(rho, p_r):
    """NEC radial: rho + p_r >= 0 for normal matter.  Visser (1995) Eq 4.1."""
    return np.asarray(rho) + np.asarray(p_r)


def NEC_transverse(rho, p_t):
    """NEC transverse: rho + p_t >= 0 for normal matter."""
    return np.asarray(rho) + np.asarray(p_t)


# ── Full NEC profile for a GR wormhole ────────────────────────────────────────

def NEC_GR_wormhole(r, r0, shape):
    """Compute NEC components across radial profile.

    Returns dict: r, rho, p_r, p_t, nec_r, nec_t, violated_r, violated_t.
    MT (1988) Eq 7 — NEC is always violated radially at the throat.
    """
    r = np.asarray(r, dtype=float)
    rho = rho_GR(r, r0, shape)
    p_r = p_r_GR(r, r0, shape)
    p_t = p_t_GR(r, r0, shape)
    nec_r = NEC_radial(rho, p_r)
    nec_t = NEC_transverse(rho, p_t)
    return {
        "r": r,
        "rho": rho,
        "p_r": p_r,
        "p_t": p_t,
        "nec_r": nec_r,
        "nec_t": nec_t,
        "violated_r": nec_r < 0,
        "violated_t": nec_t < 0,
    }


# ── Kerr-modified exotic matter budget ────────────────────────────────────────

def exotic_NEC_with_kerr(r0, M, a):
    """Exotic matter density including Kerr frame-dragging suppression.

    Returns dict: tau_required, kerr_factor, nec_violation, reduction_pct.
    tau_required = tau_kerr(r0, M, a)  [geometric units].
    """
    tau_req = tau_kerr(r0, M, a)
    k_factor = kerr_suppression(a, M)
    tau_base = tau_static(r0)
    nec_violation = -tau_req           # exotic => rho+p_r = -tau < 0
    reduction_pct = (1.0 - k_factor) * 100.0

    return {
        "tau_required": tau_req,
        "tau_static": tau_base,
        "kerr_factor": k_factor,
        "nec_violation": nec_violation,
        "reduction_pct": reduction_pct,
    }


# ── f(R) effective NEC ────────────────────────────────────────────────────────

def fR_effective_NEC(r, r0, alpha, shape):
    """Approximate effective NEC from f(R) = R + alpha*R^2 extra terms.

    Imports are done locally to avoid circular dependency with fR_gravity.
    Returns nec_eff array.
    """
    from physics.fR_gravity import fR_effective_stress_energy
    result = fR_effective_stress_energy(r, r0, alpha, shape)
    return result["nec_eff"]
