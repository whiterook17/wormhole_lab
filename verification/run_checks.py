"""Verification suite: 40 self-consistency checks across all physics modules."""

import math
import numpy as np

from physics.kerr import (
    kerr_suppression, frame_dragging, ergosphere_radius,
    horizon_radii, tau_static, tau_kerr, tau_reduction_percent,
)
from physics.morris_thorne import (
    b_power, b_prime, get_shape, wormhole_conditions,
)
from physics.energy_conditions import NEC_GR_wormhole
from physics.throat_dynamics import (
    natural_frequency, damping_regime, solve_throat_ode,
    echo_frequency, echo_spectrum,
)
from physics.fR_gravity import (
    phi_from_R, R_from_phi, scalar_potential, f_prime, f_R,
)


class CheckRunner:
    """Accumulate named physics checks and summarise results."""

    def __init__(self):
        self.results = []

    def check(self, name, expected, got, tol=1e-6, condition=None, section=""):
        """Register one check.

        condition options: None (abs diff), 'positive', 'negative',
        'less_than', 'greater_than', 'bool', 'nan'.
        """
        try:
            got_f = float(got) if not isinstance(got, bool) else got
        except Exception:
            got_f = got

        if condition == "nan":
            ok = math.isnan(float(got_f))
        elif condition == "bool":
            ok = bool(got_f) == bool(expected)
        elif condition == "positive":
            ok = float(got_f) > 0
        elif condition == "negative":
            ok = float(got_f) < 0
        elif condition == "less_than":
            ok = float(got_f) < float(expected)
        elif condition == "greater_than":
            ok = float(got_f) > float(expected)
        else:
            ok = abs(float(got_f) - float(expected)) < tol

        self.results.append({
            "name": name,
            "section": section,
            "expected": expected,
            "got": got_f,
            "ok": ok,
        })

    def summary(self):
        n_pass = sum(1 for r in self.results if r["ok"])
        n_fail = len(self.results) - n_pass
        return {
            "n_pass": n_pass,
            "n_fail": n_fail,
            "total": len(self.results),
            "all_passed": n_fail == 0,
        }

    def to_dataframe(self):
        rows = []
        for r in self.results:
            got_str = str(r["got"])
            if isinstance(r["got"], float):
                got_str = f"{r['got']:.6g}"
            rows.append({
                "Check": r["name"],
                "Section": r["section"],
                "Expected": str(r["expected"]),
                "Got": got_str,
                "Status": "PASS" if r["ok"] else "FAIL",
            })
        return rows


def run_all(M=1.0):
    """Execute all 40 physics verification checks."""
    cr = CheckRunner()
    SEC = "section"

    # ── KERR ──────────────────────────────────────────────────────────────────
    s = "Kerr"
    cr.check("ks(0,1)==1", 1.0, kerr_suppression(0.0, 1.0), tol=1e-10, section=s)
    cr.check("ks(0.9999,1)~0", 0.02, kerr_suppression(0.9999, 1.0),
             condition="less_than", section=s)
    cr.check("ks(0.5,1)<1", 1.0, kerr_suppression(0.5, 1.0),
             condition="less_than", section=s)
    cr.check("ks(0.8)<ks(0.4)", kerr_suppression(0.4, 1.0), kerr_suppression(0.8, 1.0),
             condition="less_than", section=s)
    cr.check("fd(1e5,pi/2,1,1)~0", 0.0,
             float(frame_dragging(1e5, math.pi / 2, 1.0, 1.0)), tol=1e-4, section=s)
    cr.check("fd(5,pi/2,1,0)==0", 0.0,
             float(frame_dragging(5.0, math.pi / 2, 1.0, 0.0)), section=s)
    cr.check("fd(5,pi/2,1,0.5)>0", 0.0,
             float(frame_dragging(5.0, math.pi / 2, 1.0, 0.5)),
             condition="greater_than", section=s)
    rp0, rm0 = horizon_radii(1.0, 0.0)
    cr.check("horizon(1,0) r+=2", 2.0, rp0, section=s)
    cr.check("horizon(1,0) r-=0", 0.0, rm0, section=s)
    rp1, rm1 = horizon_radii(1.0, 1.0)
    cr.check("horizon(1,1) r+=1", 1.0, rp1, section=s)
    cr.check("horizon(1,1) r-=1", 1.0, rm1, section=s)
    rp_naked, _ = horizon_radii(1.0, 1.1)
    cr.check("horizon(1,1.1) is nan", True, math.isnan(rp_naked),
             condition="bool", section=s)
    cr.check("erg(pi/2,1,0)==2", 2.0,
             float(ergosphere_radius(math.pi / 2, 1.0, 0.0)), section=s)
    rp05, _ = horizon_radii(1.0, 0.5)
    cr.check("erg(pi/2,1,0.5)>r+", rp05,
             float(ergosphere_radius(math.pi / 2, 1.0, 0.5)),
             condition="greater_than", section=s)
    cr.check("tau_kerr(1.2,1,0)==tau_static(1.2)", tau_static(1.2),
             tau_kerr(1.2, 1.0, 0.0), section=s)
    cr.check("tau_kerr(r0,1,0.99)<tau_kerr(r0,1,0)", tau_kerr(1.2, 1.0, 0.0),
             tau_kerr(1.2, 1.0, 0.99), condition="less_than", section=s)
    cr.check("reduction(0.99M)>85%", 85.0, tau_reduction_percent(0.99, 1.0),
             condition="greater_than", section=s)

    # ── MORRIS-THORNE ─────────────────────────────────────────────────────────
    s = "Morris-Thorne"
    r0 = 1.2
    cr.check("b_power(r0,r0)==r0", r0, float(b_power(r0, r0)), section=s)
    cr.check("b'(r0,'power')<1", 1.0, b_prime(r0, r0, "power"),
             condition="less_than", section=s)
    cr.check("b_power(1000*r0,r0)/(1000*r0)<1e-4", 1e-4,
             float(b_power(1000 * r0, r0)) / (1000 * r0),
             condition="less_than", section=s)

    # ── NEC ───────────────────────────────────────────────────────────────────
    s = "NEC"
    r_arr = np.array([r0])
    nec = NEC_GR_wormhole(r_arr, r0, "power")
    cr.check("NEC_radial at throat <0", 0.0, float(nec["nec_r"][0]),
             condition="negative", section=s)
    cr.check("rho_GR at throat <0", 0.0, float(nec["rho"][0]),
             condition="negative", section=s)

    # ── THROAT DYNAMICS ───────────────────────────────────────────────────────
    s = "Throat Dynamics"
    sig, a0, eta = 0.4, 1.2, 0.15
    omega0 = natural_frequency(sig, a0)
    expected_omega0 = math.sqrt(0.4 / 1.44)
    cr.check("omega0==sqrt(0.4/1.44)", expected_omega0, omega0, section=s)

    regime, omega_d = damping_regime(sig, a0, eta)
    cr.check("regime==UNDERDAMPED", True, regime == "UNDERDAMPED",
             condition="bool", section=s)
    cr.check("omega_d < omega0", omega0, omega_d,
             condition="less_than", section=s)

    result = solve_throat_ode(sig, a0, eta, 0.1, t_max=50.0, n_points=1000)
    cr.check("ODE vs analytic residual<1e-6", 1e-6, result["max_residual"],
             condition="less_than", section=s)

    da_final = float(result["da_numeric"][-1])
    cr.check("displacement(t=50)~0", 0.0, da_final, tol=0.05, section=s)

    regime_od, _ = damping_regime(0.1, 1.2, 0.8)
    cr.check("regime(sig=0.1,eta=0.8)==OVERDAMPED", True,
             regime_od == "OVERDAMPED", condition="bool", section=s)

    # ── ECHO SPECTRUM ─────────────────────────────────────────────────────────
    s = "Echo Spectrum"
    f0 = echo_frequency(sig, a0)
    cr.check("echo_frequency>0", 0.0, f0, condition="greater_than", section=s)
    cr.check("echo_spectrum(0,...)==0", 0.0, float(echo_spectrum(0.0, 1.0, f0, eta)),
             section=s)
    h_peak = float(echo_spectrum(f0, 1.0, f0, eta))
    h_10 = float(echo_spectrum(10.0 * f0, 1.0, f0, eta))
    cr.check("H(10*f0)<H(f0)", h_peak, h_10, condition="less_than", section=s)

    # ── f(R) ALGEBRA ──────────────────────────────────────────────────────────
    s = "f(R) Gravity"
    alpha = 0.2
    cr.check("phi_from_R(0.5,0.2)==1.2", 1.2, float(phi_from_R(0.5, alpha)), section=s)
    cr.check("R_from_phi(1.2,0.2)==0.5", 0.5, float(R_from_phi(1.2, alpha)), section=s)
    cr.check("scalar_potential(1.0,0.2)==0", 0.0, float(scalar_potential(1.0, alpha)),
             section=s)
    cr.check("scalar_potential(1.5,0.2)>0", 0.0, float(scalar_potential(1.5, alpha)),
             condition="greater_than", section=s)
    cr.check("f_prime(0,0.2)==1", 1.0, float(f_prime(0.0, alpha)), section=s)
    cr.check("f_R(0,0.2)==0", 0.0, float(f_R(0.0, alpha)), section=s)

    # ── SELF-CONSISTENCY ──────────────────────────────────────────────────────
    s = "Self-Consistency"
    r0_sc = 1.2
    for a_frac in [0.0, 0.5, 0.9, 0.99]:
        a_val = a_frac * M
        ratio = tau_kerr(r0_sc, M, a_val) / (tau_static(r0_sc) + 1e-30)
        expected_ratio = kerr_suppression(a_val, M)
        cr.check(f"tau_ratio matches suppression a={a_frac}M",
                 float(expected_ratio), float(ratio), tol=1e-8, section=s)

    return cr
