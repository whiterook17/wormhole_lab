"""Physical constants in geometric units (G = c = 1) and SI conversions."""

# Geometric units
G = 1.0
c = 1.0
k_B = 1.0

# SI values
G_SI = 6.674e-11          # m³ kg⁻¹ s⁻²
c_SI = 2.998e8            # m/s
M_sun = 1.989e30          # kg

# Derived SI
r_g_SI = G_SI * M_sun / c_SI**2          # ≈ 1477 m, solar gravitational radius
TAU_CONV = c_SI**4 / G_SI                # J/m² per geometric unit of stress-energy

# Observational / lab limits
CASIMIR_LAB_MAX = 1e-3                   # J/m², 2026 best Casimir lab density

# Numerical
EPS = 1e-10

# Default parameter set used across the app
DEFAULT_PARAMS = {
    "M": 1.0,
    "a_over_M": 0.85,
    "r0": 1.2,
    "sigma_throat": 0.40,
    "eta_s": 0.15,
    "alpha_fR": 0.15,
    "phi0": 1.05,
}
