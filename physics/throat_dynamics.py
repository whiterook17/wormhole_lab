"""Dynamic throat oscillations and gravitational-wave echo spectrum.

Reference equations:
  Eq 6.3 — damped harmonic oscillator for throat displacement delta_a
  Eq 9b  — echo transfer function H(f)
"""

import math
import numpy as np
from scipy.integrate import solve_ivp


# ── Natural frequency ──────────────────────────────────────────────────────────

def natural_frequency(sigma_throat, a0):
    """omega_0 = sqrt(sigma / a0^2).  Eq 6.3 restoring coefficient."""
    return math.sqrt(sigma_throat / a0**2)


# ── Damping classification ─────────────────────────────────────────────────────

def damping_regime(sigma_throat, a0, eta_s):
    """Classify oscillator regime and return (regime_str, omega_d).

    UNDERDAMPED: omega_0 > eta_s  -> omega_d = sqrt(omega_0^2 - eta_s^2)
    CRITICAL:    omega_0 == eta_s -> omega_d = 0
    OVERDAMPED:  omega_0 < eta_s  -> omega_d conventionally 0 (gamma used instead)
    """
    omega0 = natural_frequency(sigma_throat, a0)
    disc = omega0**2 - eta_s**2
    if disc > 1e-10:
        regime = "UNDERDAMPED"
        omega_d = math.sqrt(disc)
    elif abs(disc) <= 1e-10:
        regime = "CRITICAL"
        omega_d = 0.0
    else:
        regime = "OVERDAMPED"
        omega_d = 0.0
    return regime, omega_d


# ── Analytic solution ──────────────────────────────────────────────────────────

def throat_displacement_analytic(t, delta_a0, sigma_throat, a0, eta_s, da_dot0=0.0):
    """Exact closed-form solution of Eq 6.3 for all three damping regimes.

    Eq 6.3: delta_a'' + 2*eta_s*delta_a' + (sigma/a0^2)*delta_a = 0

    Initial conditions: delta_a(0) = delta_a0, delta_a'(0) = da_dot0

    UNDERDAMPED:  delta_a = e^{-eta_s t} [C1 cos(omega_d t) + C2 sin(omega_d t)]
                  C1 = delta_a0
                  C2 = (da_dot0 + eta_s * delta_a0) / omega_d
    OVERDAMPED:   gamma = sqrt(eta_s^2 - omega_0^2)
                  delta_a = e^{-eta_s t} [C1 cosh(gamma t) + C2 sinh(gamma t)]
                  C1 = delta_a0
                  C2 = (da_dot0 + eta_s * delta_a0) / gamma
    CRITICAL:
                  delta_a = e^{-eta_s t} [C1 + C2 t]
                  C1 = delta_a0
                  C2 = da_dot0 + eta_s * delta_a0
    """
    t = np.asarray(t, dtype=float)
    omega0 = natural_frequency(sigma_throat, a0)
    regime, omega_d = damping_regime(sigma_throat, a0, eta_s)

    envelope = np.exp(-eta_s * t)

    if regime == "UNDERDAMPED":
        C1 = delta_a0
        C2 = (da_dot0 + eta_s * delta_a0) / omega_d
        return envelope * (C1 * np.cos(omega_d * t) + C2 * np.sin(omega_d * t))

    elif regime == "OVERDAMPED":
        gamma = math.sqrt(eta_s**2 - omega0**2)
        C1 = delta_a0
        C2 = (da_dot0 + eta_s * delta_a0) / gamma
        return envelope * (C1 * np.cosh(gamma * t) + C2 * np.sinh(gamma * t))

    else:  # CRITICAL
        C1 = delta_a0
        C2 = da_dot0 + eta_s * delta_a0
        return envelope * (C1 + C2 * t)


# ── ODE solver ────────────────────────────────────────────────────────────────

def _ode_rhs(t, y, sigma_throat, a0, eta_s):
    """RHS of Eq 6.3: y = [delta_a, delta_a_dot]."""
    omega0_sq = sigma_throat / a0**2
    return [y[1], -2.0 * eta_s * y[1] - omega0_sq * y[0]]


def solve_throat_ode(sigma_throat, a0, eta_s, delta_a0, t_max=50.0,
                     n_points=1000, da_dot0=0.0):
    """RK45 integration of Eq 6.3.

    Returns dict: t, da_numeric, da_dot, da_analytic, max_residual, regime, success.
    """
    t_eval = np.linspace(0.0, t_max, n_points)
    sol = solve_ivp(
        _ode_rhs,
        (0.0, t_max),
        [delta_a0, da_dot0],
        args=(sigma_throat, a0, eta_s),
        method="RK45",
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )
    regime, _ = damping_regime(sigma_throat, a0, eta_s)
    da_analytic = throat_displacement_analytic(
        sol.t, delta_a0, sigma_throat, a0, eta_s, da_dot0
    )
    residual = np.abs(sol.y[0] - da_analytic)
    return {
        "t": sol.t,
        "da_numeric": sol.y[0],
        "da_dot": sol.y[1],
        "da_analytic": da_analytic,
        "max_residual": float(np.max(residual)),
        "regime": regime,
        "success": sol.success,
    }


# ── Echo properties ───────────────────────────────────────────────────────────

def echo_frequency(sigma_throat, a0):
    """f0 = omega_0 / (2*pi).  Fundamental echo repetition rate."""
    return natural_frequency(sigma_throat, a0) / (2.0 * math.pi)


def echo_interval(sigma_throat, a0, eta_s):
    """Delta_t = pi / omega_d.  Returns inf if not underdamped."""
    regime, omega_d = damping_regime(sigma_throat, a0, eta_s)
    if regime != "UNDERDAMPED" or omega_d < 1e-12:
        return float('inf')
    return math.pi / omega_d


def echo_count(sigma_throat, a0, eta_s, threshold=0.01):
    """Number of echoes above threshold before amplitude decays below threshold."""
    regime, omega_d = damping_regime(sigma_throat, a0, eta_s)
    if regime != "UNDERDAMPED":
        return 0
    if eta_s < 1e-12:
        return 999
    dt = echo_interval(sigma_throat, a0, eta_s)
    if math.isinf(dt):
        return 0
    t_decay = -math.log(threshold) / eta_s
    return max(0, int(t_decay / dt))


# ── Echo spectrum ──────────────────────────────────────────────────────────────

def echo_spectrum(f, A0, f0, eta_s):
    """Transfer function H(f) from Eq (9b).

    H(f) = A0 * (f/f0)^2 * exp(-eta_s * f / f0^2) / (1 + (f/f0)^2)
    """
    f = np.asarray(f, dtype=float)
    x = f / (f0 + 1e-30)
    return A0 * x**2 * np.exp(-eta_s * f / (f0**2 + 1e-30)) / (1.0 + x**2)


def echo_spectrum_array(f_min, f_max, n, A0, f0, eta_s):
    """Sample echo spectrum over [f_min, f_max] with n points.

    Returns (freqs, amps).
    """
    freqs = np.linspace(f_min, f_max, n)
    amps = echo_spectrum(freqs, A0, f0, eta_s)
    return freqs, amps


# ── Stability index ───────────────────────────────────────────────────────────

def stability_index(sigma_throat, a0, eta_s, kerr_factor=None):
    """Dimensionless stability measure in [0, 1].

    Higher = more stable. Based on damping ratio and optional Kerr suppression.
    """
    omega0 = natural_frequency(sigma_throat, a0)
    zeta = eta_s / (omega0 + 1e-30)   # damping ratio
    base = math.exp(-max(zeta - 1.0, 0.0))
    if zeta < 1.0:
        decay_score = zeta
    else:
        decay_score = 1.0 / zeta
    index = 0.5 * (base + decay_score)
    if kerr_factor is not None:
        index *= (1.0 - kerr_factor * 0.2)
    return float(np.clip(index, 0.0, 1.0))


# ── Israel junction report ────────────────────────────────────────────────────

def israel_junction_report(a0, sigma_throat, eta_s):
    """Summary of junction condition dynamics at the throat shell.

    Returns dict: omega0, omega_d, regime, f0, dt_echo, n_echoes.
    Israel (1966) thin-shell formalism applied to wormhole throat.
    """
    omega0 = natural_frequency(sigma_throat, a0)
    regime, omega_d = damping_regime(sigma_throat, a0, eta_s)
    f0 = echo_frequency(sigma_throat, a0)
    dt = echo_interval(sigma_throat, a0, eta_s)
    n = echo_count(sigma_throat, a0, eta_s)
    return {
        "omega0": omega0,
        "omega_d": omega_d,
        "regime": regime,
        "f0": f0,
        "dt_echo": dt,
        "n_echoes": n,
    }
