"""Kerr metric functions in Boyer-Lindquist coordinates (geometric units G=c=1).

References: Bardeen, Press & Teukolsky (1972); Misner, Thorne & Wheeler Ch. 33.
"""

import math
import numpy as np
from physics.constants import EPS


def sigma_BL(r, theta, a):
    """Σ = r² + a²cos²θ  (Boyer-Lindquist, MTW 33.2)."""
    return r**2 + a**2 * np.cos(theta)**2


def delta(r, M, a):
    """Δ = r² − 2Mr + a²  (MTW 33.2)."""
    return r**2 - 2.0 * M * r + a**2


def frame_dragging(r, theta, M, a):
    """ω(r,θ) = 2Mar / [(r²+a²)Σ + 2Ma²r sin²θ].

    Bardeen (1970) Eq 2. Limits: a→0 → 0, r→∞ → 0.
    """
    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    sig = sigma_BL(r, theta, a)
    numerator = 2.0 * M * a * r
    denominator = (r**2 + a**2) * sig + 2.0 * M * a**2 * r * np.sin(theta)**2
    return np.where(np.abs(denominator) < EPS, 0.0, numerator / denominator)


def ergosphere_radius(theta, M, a):
    """r_erg = M + √(M² − a²cos²θ).

    Outer static limit surface; MTW §33.4.
    """
    theta = np.asarray(theta, dtype=float)
    discriminant = M**2 - a**2 * np.cos(theta)**2
    return M + np.sqrt(np.maximum(discriminant, 0.0))


def horizon_radii(M, a):
    """r± = M ± √(M²−a²).

    Returns (nan, nan) if a > M (naked singularity). MTW §33.2.
    """
    discriminant = M**2 - a**2
    if discriminant < 0:
        return (float('nan'), float('nan'))
    sqrt_d = math.sqrt(discriminant)
    return (M + sqrt_d, M - sqrt_d)


def isco_radius(M, a, prograde=True):
    """Innermost stable circular orbit — Bardeen-Press-Teukolsky (1972) Eq 2.21.

    Returns r_ISCO in geometric units.
    """
    a_clip = np.clip(a / M, -1.0 + EPS, 1.0 - EPS) * M
    sign = -1.0 if prograde else 1.0

    Z1 = 1.0 + (1.0 - (a_clip / M)**2)**(1.0 / 3.0) * (
        (1.0 + a_clip / M)**(1.0 / 3.0) + (1.0 - a_clip / M)**(1.0 / 3.0)
    )
    Z2 = np.sqrt(3.0 * (a_clip / M)**2 + Z1**2)
    return M * (3.0 + Z2 + sign * np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)))


def kerr_suppression(a, M):
    """√(1 − a²/M²).

    Dimensionless factor reducing exotic-matter density at throat.
    Returns 1 if M=0. Clips a/M to [0, 1−ε].
    """
    if M == 0:
        return 1.0
    ratio = np.clip(np.abs(a) / np.abs(M), 0.0, 1.0 - EPS)
    return np.sqrt(1.0 - ratio**2)


def tau_static(r0):
    """τ_static = r0^{-2}.  Baseline exotic stress-energy at throat (Φ=0)."""
    return r0**(-2)


def tau_kerr(r0, M, a):
    """τ_Kerr = τ_static × kerr_suppression(a, M).

    Kerr-frame exotic matter requirement at throat.
    """
    return tau_static(r0) * kerr_suppression(a, M)


def tau_reduction_percent(a, M):
    """Percentage reduction in exotic matter from Kerr frame-dragging."""
    return (1.0 - kerr_suppression(a, M)) * 100.0


def verify_limits(M=1.0):
    """Run 8 sanity checks on Kerr functions.

    Returns list of (name, expected, got, passed).
    """
    results = []

    def chk(name, expected, got, tol=1e-10, is_nan=False):
        if is_nan:
            ok = math.isnan(got)
        else:
            ok = abs(float(got) - float(expected)) < tol
        results.append((name, expected, got, ok))

    a = M  # spin parameter
    chk("kerr_suppression(0,1)==1", 1.0, kerr_suppression(0.0, 1.0))
    chk("kerr_suppression(0.9999,1)~0", 0.0, kerr_suppression(0.9999, 1.0), tol=0.02)
    chk("frame_dragging(1e5,pi/2,1,1)~0", 0.0,
        float(frame_dragging(1e5, math.pi / 2, 1.0, 1.0)), tol=1e-4)
    chk("frame_dragging(5,pi/2,1,0)==0", 0.0,
        float(frame_dragging(5.0, math.pi / 2, 1.0, 0.0)))
    chk("ergosphere(pi/2,1,0)==2", 2.0,
        float(ergosphere_radius(math.pi / 2, 1.0, 0.0)))
    r_plus, r_minus = horizon_radii(1.0, 0.0)
    chk("horizon(1,0) r+=2", 2.0, r_plus)
    r_plus2, r_minus2 = horizon_radii(1.0, 1.0)
    chk("horizon(1,1) r+=1", 1.0, r_plus2)
    r_plus3, _ = horizon_radii(1.0, 1.1)
    chk("horizon(1,1.1) is nan", float('nan'), r_plus3, is_nan=True)

    return results
