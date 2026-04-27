"""Morris-Thorne traversable wormhole shape functions and stress-energy.

Reference: Morris & Thorne (1988), Am. J. Phys. 56, 395. Zero-tidal (Phi=0).
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from physics.constants import EPS


# ── Shape functions ────────────────────────────────────────────────────────────

def b_power(r, r0):
    """b(r) = r0^2 / r.  Morris-Thorne (1988) Eq 10, simplest case."""
    return r0**2 / np.asarray(r, dtype=float)


def b_constant(r, r0):
    """b(r) = r0.  Constant-b shape; asymptotically flat if only near throat."""
    r = np.asarray(r, dtype=float)
    return np.full_like(r, float(r0))


def b_power_law(r, r0, gamma=0.5):
    """b(r) = r0 * (r0/r)^gamma.  Generalised power-law, MT variant."""
    r = np.asarray(r, dtype=float)
    return r0 * (r0 / r)**gamma


def b_visser(r, r0, lam=0.1):
    """b(r) = r0 - lam*(r - r0).  Visser (1995) linearised shape."""
    r = np.asarray(r, dtype=float)
    return r0 - lam * (r - r0)


SHAPE_FUNCTIONS = {
    "power": b_power,
    "constant": b_constant,
    "power_law": b_power_law,
    "visser": b_visser,
}


def get_shape(name, r, r0, **kwargs):
    """Dispatch to named shape function with extra kwargs."""
    fn = SHAPE_FUNCTIONS[name]
    try:
        return fn(r, r0, **kwargs)
    except TypeError:
        return fn(r, r0)


def b_prime(r, r0, shape, dr=1e-6):
    """Numerical derivative db/dr via central difference."""
    r = float(r)
    b_plus = get_shape(shape, r + dr, r0)
    b_minus = get_shape(shape, r - dr, r0)
    return float((b_plus - b_minus) / (2.0 * dr))


# ── Wormhole validity conditions ───────────────────────────────────────────────

def wormhole_conditions(r0, shape, verbose=False):
    """Check MT throat conditions (Morris & Thorne 1988, Eq 11-13).

    Returns dict: throat, flare_out, asymptotic, valid, b_r0, b_prime_r0.
    """
    b_r0 = float(get_shape(shape, r0, r0))
    bp_r0 = b_prime(r0, r0, shape)
    b_large = float(get_shape(shape, 1000.0 * r0, r0))
    asym_ratio = b_large / (1000.0 * r0)

    throat = abs(b_r0 - r0) < 1e-6
    flare_out = bp_r0 < 1.0
    asymptotic = asym_ratio < 1e-4
    valid = throat and flare_out and asymptotic

    return {
        "throat": throat,
        "flare_out": flare_out,
        "asymptotic": asymptotic,
        "valid": valid,
        "b_r0": b_r0,
        "b_prime_r0": bp_r0,
    }


# ── Embedding diagram ──────────────────────────────────────────────────────────

def embedding_height(r, r0, shape):
    """Flamm paraboloid z(r) for upper sheet via cumulative_trapezoid.

    dz/dr = 1/sqrt(r/b(r) - 1)  (MT 1988 Eq 5).
    """
    r = np.asarray(r, dtype=float)
    b = get_shape(shape, r, r0)
    ratio = r / (b + EPS)
    denom = np.sqrt(np.maximum(ratio - 1.0, 0.0))
    integrand = np.where(denom > EPS, 1.0 / denom, 0.0)
    z = cumulative_trapezoid(integrand, r, initial=0.0)
    return z


# ── Stress-energy components (Phi=0) ──────────────────────────────────────────

def rho_GR(r, r0, shape):
    """Energy density rho = b'/(8 pi r^2).  MT (1988) Eq 6a with Phi=0."""
    r = np.asarray(r, dtype=float)
    bp = np.array([b_prime(ri, r0, shape) for ri in r.flat]).reshape(r.shape)
    return bp / (8.0 * np.pi * r**2)


def p_r_GR(r, r0, shape):
    """Radial pressure p_r = -b/(8 pi r^3).  MT (1988) Eq 6b with Phi=0."""
    r = np.asarray(r, dtype=float)
    b = get_shape(shape, r, r0)
    return -b / (8.0 * np.pi * r**3)


def p_t_GR(r, r0, shape):
    """Transverse pressure p_t = b/(16 pi r^3) - b'/(16 pi r^2).  MT (1988) Eq 6c."""
    r = np.asarray(r, dtype=float)
    b = get_shape(shape, r, r0)
    bp = np.array([b_prime(ri, r0, shape) for ri in r.flat]).reshape(r.shape)
    return b / (16.0 * np.pi * r**3) - bp / (16.0 * np.pi * r**2)
