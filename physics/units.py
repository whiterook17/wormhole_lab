"""Conversion between geometric units (G=c=1) and SI units.

All conversions assume a central mass of M_solar solar masses unless specified.
"""

from __future__ import annotations

# SI reference values
_R_G    = 1477.0          # metres per gravitational radius (1 solar mass)
_T_G    = 4.93e-6         # seconds per r_g/c (1 solar mass)
_M_KG   = 1.989e30        # kg per solar mass
_E_J    = 1.8e47          # J per M_sun c^2
_TAU_J  = 1.21e44         # J/m^2 per geometric tension unit (1 solar mass)


def to_metric(value: float, kind: str, M_solar: float = 1.0) -> tuple[float, str]:
    """Convert a geometric-unit value to SI.

    kind must be one of: 'length', 'time', 'mass', 'energy', 'tension', 'frequency'.
    Returns (converted_value, unit_label).
    """
    kind = kind.lower()
    scale = max(M_solar, 1e-30)

    if kind == "length":
        v = value * _R_G * scale
        if abs(v) >= 1e6:
            return v / 1e3, "km"
        if abs(v) >= 1.0:
            return v, "m"
        return v * 1e2, "cm"

    elif kind == "time":
        v = value * _T_G * scale
        if abs(v) < 1e-3:
            return v * 1e6, "µs"
        if abs(v) < 1.0:
            return v * 1e3, "ms"
        return v, "s"

    elif kind == "mass":
        v = value * _M_KG * scale
        return v, "kg"

    elif kind == "energy":
        v = value * _E_J * scale**2
        return v, "J"

    elif kind == "tension":
        v = value * _TAU_J * scale**2
        return v, "J/m²"

    elif kind == "frequency":
        # Geometric frequency (c/r_g units) → Hz
        if scale > 0 and _T_G > 0:
            v = value / (_T_G * scale)
        else:
            v = value
        return v, "Hz"

    else:
        return value, "(geom)"


def fmt(value: float, kind: str, mode: str, M_solar: float = 1.0,
        geom_unit: str = "") -> str:
    """Format a value according to the current display-units mode.

    mode: 'Geometric', 'Both', or 'Metric'.
    """
    geom_str = f"{value:.4g}"
    if geom_unit:
        geom_str += f" {geom_unit}"

    if mode == "Geometric":
        return geom_str

    v_si, u_si = to_metric(value, kind, M_solar)
    si_str = f"{v_si:.4g} {u_si}"

    if mode == "Metric":
        return si_str
    # Both
    return f"{geom_str}  ({si_str})"
