"""Wormhole Math Checker — Streamlit entry point (v2.1 UX overhaul)."""

from __future__ import annotations

import json
import math
import pathlib
import warnings
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Physics
from physics.constants import DEFAULT_PARAMS, CASIMIR_LAB_MAX, EPS
from physics.kerr import (
    kerr_suppression, frame_dragging, ergosphere_radius,
    horizon_radii, tau_static, tau_kerr, tau_reduction_percent,
)
from physics.morris_thorne import (
    get_shape, b_prime, wormhole_conditions, embedding_height,
    rho_GR, p_r_GR, p_t_GR,
)
from physics.energy_conditions import NEC_GR_wormhole, exotic_NEC_with_kerr
from physics.throat_dynamics import (
    natural_frequency, damping_regime, solve_throat_ode,
    echo_frequency, echo_interval, echo_count,
    echo_spectrum, echo_spectrum_array, stability_index,
    israel_junction_report,
)
from physics.fR_gravity import (
    ricci_scalar_MT, f_R, f_prime, phi_from_R, R_from_phi,
    scalar_potential, fR_effective_stress_energy,
    solve_scalar_field, shoot_phi0,
)
from physics.units import to_metric, fmt
from verification.run_checks import run_all
from ui.guide import render_guide

# Storage (never crashes the app)
try:
    from storage.run import Run
    from storage import save_run, load_all_runs
    _STORAGE_OK = True
except Exception:
    _STORAGE_OK = False
    def save_run(run): pass          # noqa: E731
    def load_all_runs(): return []   # noqa: E731


# ── Colour palette ────────────────────────────────────────────────────────────
TEAL   = "#00c8c8"
GOLD   = "#e8a020"
RED    = "#e05050"
PURPLE = "#a070e0"
GREEN  = "#40d080"
DIM    = "#2a3f5a"
BG     = "#0b1120"
BG2    = "#111e36"
TEXT   = "#c8daea"


# ── Auto-save ─────────────────────────────────────────────────────────────────

def _auto_save(model_name: str, params: dict, results: dict,
               convergence: dict | None = None) -> None:
    if not _STORAGE_OK:
        return
    try:
        run = Run.create(model_name, params, results, convergence)
        save_run(run)
    except Exception:
        pass


# ── Usage logging ─────────────────────────────────────────────────────────────

def _log_usage(page: str, control: str, value) -> None:
    try:
        p = pathlib.Path(".streamlit/usage.json")
        p.parent.mkdir(exist_ok=True)
        events: list = json.loads(p.read_text()) if p.exists() else []
        events.append({
            "ts": datetime.datetime.utcnow().isoformat(),
            "page": page,
            "control": control,
            "value": value,
        })
        p.write_text(json.dumps(events, indent=2))
    except Exception:
        pass


# ── Units helpers ─────────────────────────────────────────────────────────────

def _du() -> str:
    """Current display-units mode."""
    return st.session_state.get("display_units", "Geometric")


def _si_hint(value: float, kind: str, M_solar: float = 1.0) -> str | None:
    """Return SI hint string when mode is Both or Metric, else None."""
    mode = _du()
    if mode == "Geometric":
        return None
    v_si, u_si = to_metric(value, kind, M_solar)
    return f"{v_si:.4g} {u_si}"


# ── Navigation ────────────────────────────────────────────────────────────────

def navigate_to(page_name: str) -> None:
    st.session_state["_nav"] = page_name
    st.session_state["_radio_nav"] = page_name
    st.rerun()


def _help_banner(page_key: str) -> None:
    """Show a Guide button at top-right of every page."""
    _, col_btn = st.columns([9, 1])
    with col_btn:
        if st.button("\U0001f4d6 Guide", key=f"_hbanner_{page_key}",
                     help="Open the plain-English guide"):
            navigate_to("\U0001f4d6 Guide")


# ── Numeric stepper component ─────────────────────────────────────────────────

def numeric_input_with_steppers(
    label: str,
    key: str,
    min_val: float,
    max_val: float,
    default: float,
    step: float,
    format_str: str = "%.3f",
    help_text: str | None = None,
    unit_geom: str = "",
    unit_si_fn=None,
) -> float:
    """Number input with flanking − / + stepper buttons.

    key is a page-local identifier; session state is stored under _stepper_{key}.
    Returns the current float value.
    """
    skey = f"_stepper_{key}"
    if skey not in st.session_state:
        st.session_state[skey] = float(default)

    def _dec():
        st.session_state[skey] = float(
            max(min_val, round(st.session_state[skey] - step, 10))
        )

    def _inc():
        st.session_state[skey] = float(
            min(max_val, round(st.session_state[skey] + step, 10))
        )

    c_m, c_i, c_p = st.columns([1, 8, 1])
    c_m.button("−", key=f"_bm_{key}", on_click=_dec)
    with c_i:
        val = st.number_input(
            label,
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(st.session_state[skey]),
            step=float(step),
            format=format_str,
            help=help_text,
            key=skey,
        )
    c_p.button("+", key=f"_bp_{key}", on_click=_inc)

    if unit_si_fn is not None and _du() != "Geometric":
        try:
            hint = unit_si_fn(val)
            if hint:
                st.caption(f"≈ {hint}")
        except Exception:
            pass

    return float(st.session_state[skey])


# ── Plot styling helpers ──────────────────────────────────────────────────────

def _style_ax(ax, title=""):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEAL)
    if title:
        ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_color(DIM)
    ax.grid(True, color=DIM, linewidth=0.4, linestyle="--")


def _new_fig(nrows=1, ncols=1, figsize=(7, 4)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(BG)
    return fig, axes


# ── Plot builders ─────────────────────────────────────────────────────────────

def plot_kerr_suppression(M, a, r0):
    fig, ax = _new_fig(figsize=(6, 4))
    x = np.linspace(0.0, 0.999, 400)
    y = np.array([kerr_suppression(xi * M, M) for xi in x])
    ax.plot(x, y, color=TEAL, lw=2)
    ax_frac = a / M
    sup = kerr_suppression(a, M)
    ax.axvline(ax_frac, color=GOLD, lw=1.4, linestyle="--",
               label=f"a/M={ax_frac:.2f}")
    ax.annotate(f"{sup:.3f}", xy=(ax_frac, sup),
                xytext=(ax_frac + 0.05, sup + 0.05), color=GOLD, fontsize=9)
    pct = tau_reduction_percent(a, M)
    ax.annotate(f"-{pct:.1f}% exotic matter at a=0.99M",
                xy=(0.99, kerr_suppression(0.99 * M, M)),
                xytext=(0.6, 0.4), color=RED, fontsize=8,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    _style_ax(ax, "Kerr Suppression Factor")
    ax.set_xlabel("a / M (spin parameter)")
    ax.set_ylabel("sqrt(1 - a^2/M^2)")
    ax.legend(facecolor=BG2, labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    return fig


def plot_frame_dragging(M, a, r0):
    fig, ax = _new_fig(figsize=(6, 4))
    r_arr = np.linspace(r0, 20.0, 400)
    omega = frame_dragging(r_arr, math.pi / 2, M, a)
    ax.plot(r_arr, omega, color=PURPLE, lw=2)
    ax.axvline(r0, color=GOLD, lw=1.2, linestyle="--", label=f"r0={r0:.2f}")
    _style_ax(ax, "Frame Dragging omega(r, pi/2)")
    ax.set_xlabel("r (geometric units)")
    ax.set_ylabel("omega(r)")
    ax.legend(facecolor=BG2, labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    return fig


def plot_nec(r0, shape, M, a):
    fig, axes = _new_fig(1, 2, figsize=(12, 4))
    r_arr = np.linspace(r0, 6.0 * r0, 300)
    nec_data = NEC_GR_wormhole(r_arr, r0, shape)
    nec_r = nec_data["nec_r"]

    ax = axes[0]
    ax.plot(r_arr, nec_r, color=RED, lw=2, label="rho + p_r")
    ax.axhline(0, color=DIM, lw=1)
    ax.fill_between(r_arr, nec_r, 0, where=(nec_r < 0),
                    color=RED, alpha=0.25, label="NEC violated")
    _style_ax(ax, "NEC Violated = Wormhole Supported")
    ax.set_xlabel("r")
    ax.set_ylabel("rho + p_r")
    ax.legend(facecolor=BG2, labelcolor=TEXT, fontsize=8)

    ax2 = axes[1]
    ax2.plot(r_arr, nec_data["rho"], color=TEAL, lw=1.8, label="rho")
    ax2.plot(r_arr, nec_data["p_r"], color=GOLD, lw=1.8, label="p_r")
    ax2.plot(r_arr, nec_data["p_t"], color=PURPLE, lw=1.8, label="p_t")
    ax2.axhline(0, color=DIM, lw=1)
    _style_ax(ax2, "Stress-Energy Components")
    ax2.set_xlabel("r")
    ax2.set_ylabel("value")
    ax2.legend(facecolor=BG2, labelcolor=TEXT, fontsize=8)

    fig.tight_layout()
    return fig


def plot_flamm(r0, shape):
    fig, ax = _new_fig(figsize=(8, 5))
    r_arr = np.linspace(r0, 6.0 * r0, 400)
    z = embedding_height(r_arr, r0, shape)
    ax.plot(r_arr,  z, color=TEAL,   lw=2, label="Upper sheet z(r)")
    ax.plot(r_arr, -z, color=PURPLE, lw=2, linestyle="--", label="Lower sheet -z(r)")
    ax.axvline(r0, color=GOLD, lw=1.4, linestyle="--", label=f"Throat r0={r0:.2f}")
    _style_ax(ax, "Flamm Paraboloid — Wormhole Embedding")
    ax.set_xlabel("r")
    ax.set_ylabel("z (embedding height)")
    ax.legend(facecolor=BG2, labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    return fig


def plot_throat(sigma_throat, a0, eta_s, delta_a0, t_max):
    regime, omega_d = damping_regime(sigma_throat, a0, eta_s)
    color_map = {"UNDERDAMPED": TEAL, "CRITICAL": GOLD, "OVERDAMPED": RED}
    clr = color_map.get(regime, TEAL)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solve_throat_ode(sigma_throat, a0, eta_s, delta_a0, t_max=t_max)

    fig, ax = _new_fig(figsize=(7, 4))
    ax.plot(result["t"], result["da_numeric"], color=clr, lw=2, label="Numeric (RK45)")
    ax.plot(result["t"], result["da_analytic"], color=GOLD, lw=1.5,
            linestyle="--", label="Analytic")
    ax.text(0.02, 0.95, f"max|residual|={result['max_residual']:.2e}",
            transform=ax.transAxes, color=TEXT, fontsize=8, va="top")
    _style_ax(ax, f"Throat Displacement — {regime}")
    ax.set_xlabel("t")
    ax.set_ylabel("delta_a(t)")
    ax.legend(facecolor=BG2, labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    return fig


def plot_echo(sigma_throat, a0, eta_s):
    f0 = echo_frequency(sigma_throat, a0)
    f_max = max(10.0 * f0, 1.0)
    freqs, amps = echo_spectrum_array(0.0, f_max, 500, 1.0, f0, eta_s)

    fig, ax = _new_fig(figsize=(7, 4))
    ax.plot(freqs, amps, color=TEAL, lw=2)
    ax.axvline(f0, color=GOLD, lw=1.2, linestyle="--", label=f"f0={f0:.4f}")
    _style_ax(ax, "GW Echo Spectrum H(f)  [Eq 9b]")
    ax.set_xlabel("f")
    ax.set_ylabel("H(f)")
    ax.legend(facecolor=BG2, labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    return fig


def plot_regime_map(a0_cur, sigma_cur):
    a0_vals = np.linspace(0.3, 4.0, 60)
    sig_vals = np.linspace(0.01, 2.0, 60)
    Z = np.zeros((len(sig_vals), len(a0_vals)))
    for i, sv in enumerate(sig_vals):
        for j, av in enumerate(a0_vals):
            reg, _ = damping_regime(sv, av, 0.15)
            Z[i, j] = 0 if reg == "UNDERDAMPED" else (1 if reg == "CRITICAL" else 2)

    fig, ax = _new_fig(figsize=(7, 5))
    cmap = matplotlib.colors.ListedColormap([GREEN, GOLD, RED])
    ax.pcolormesh(a0_vals, sig_vals, Z, cmap=cmap, shading="auto", vmin=0, vmax=2)
    ax.scatter([a0_cur], [sigma_cur], color="white", s=100, zorder=5, marker="*")
    _style_ax(ax, "Regime Map (eta_s=0.15 fixed)")
    ax.set_xlabel("a0")
    ax.set_ylabel("sigma_throat")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GREEN,   label="Underdamped"),
        Patch(facecolor=GOLD,    label="Critical"),
        Patch(facecolor=RED,     label="Overdamped"),
        Patch(facecolor="white", label="Current"),
    ]
    ax.legend(handles=legend_elements, facecolor=BG2, labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    return fig


def plot_fR_phi(r_arr, phi_arr, r0, alpha):
    fig, axes = _new_fig(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(r_arr, phi_arr, color=PURPLE, lw=2)
    ax.axhline(1.0, color=DIM, lw=1, linestyle="--", label="phi=1 (GR limit)")
    ax.axvline(r0, color=GOLD, lw=1.2, linestyle="--", label=f"r0={r0:.2f}")
    _style_ax(ax, "Scalar Field phi(r)")
    ax.set_xlabel("r")
    ax.set_ylabel("phi(r)")
    ax.legend(facecolor=BG2, labelcolor=TEXT, fontsize=8)

    r_np = np.asarray(r_arr)
    R_arr = ricci_scalar_MT(r_np, r0, "power")
    ax2 = axes[1]
    ax2.plot(r_np, R_arr, color=GOLD, lw=2)
    ax2.axhline(0, color=DIM, lw=1)
    _style_ax(ax2, "Ricci Scalar R(r)")
    ax2.set_xlabel("r")
    ax2.set_ylabel("R(r)")

    fig.tight_layout()
    return fig


def plot_fR_nec(r0, alpha, shape):
    r_arr = np.linspace(r0, 10.0 * r0, 300)
    result = fR_effective_stress_energy(r_arr, r0, alpha, shape)
    nec = result["nec_eff"]

    fig, ax = _new_fig(figsize=(6, 4))
    ax.plot(r_arr, nec, color=PURPLE, lw=2)
    ax.axhline(0, color=DIM, lw=1)
    ax.fill_between(r_arr, nec, 0, where=(nec < 0),
                    color=RED, alpha=0.3, label="NEC violated")
    _style_ax(ax, "Effective NEC (f(R) corrected)")
    ax.set_xlabel("r")
    ax.set_ylabel("rho_eff + p_r_eff")
    ax.legend(facecolor=BG2, labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    return fig


def plot_potential(alpha, phi0_mark=None):
    phi_arr = np.linspace(0.5, 1.8, 300)
    V = scalar_potential(phi_arr, alpha)

    fig, ax = _new_fig(figsize=(6, 4))
    ax.plot(phi_arr, V, color=GOLD, lw=2)
    if phi0_mark is not None:
        V_mark = float(scalar_potential(phi0_mark, alpha))
        ax.scatter([phi0_mark], [V_mark], color=TEAL, s=60, zorder=5,
                   label=f"phi(r0)={phi0_mark:.3f}")
    _style_ax(ax, "Scalar Potential V(phi)")
    ax.set_xlabel("phi")
    ax.set_ylabel("V(phi)")
    ax.legend(facecolor=BG2, labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    return fig


def plot_alpha_sweep(r0, shape, current_alpha):
    alphas = np.linspace(0.01, 1.5, 40)
    min_nec = []
    for al in alphas:
        try:
            r_arr = np.linspace(r0, 8.0 * r0, 120)
            res = fR_effective_stress_energy(r_arr, r0, al, shape)
            min_nec.append(float(np.min(res["nec_eff"])))
        except Exception:
            min_nec.append(0.0)
    min_nec = np.array(min_nec)

    fig, ax = _new_fig(figsize=(8, 4))
    colors = [RED if v < 0 else DIM for v in min_nec]
    ax.scatter(alphas, min_nec, c=colors, s=30)
    ax.axhline(0, color=DIM, lw=1)
    ax.axvline(current_alpha, color=GOLD, lw=1.2, linestyle="--",
               label=f"alpha={current_alpha:.2f}")
    _style_ax(ax, "Min Effective NEC vs alpha")
    ax.set_xlabel("alpha")
    ax.set_ylabel("min(rho_eff + p_r_eff)")
    ax.legend(facecolor=BG2, labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    return fig


# ── Session state initialisation ──────────────────────────────────────────────

def _init_state():
    dp = DEFAULT_PARAMS
    for k, v in dp.items():
        if k not in st.session_state:
            st.session_state[k] = v
    defaults = {
        "fR_solution": None,
        "last_checks": None,
        "display_units": "Geometric",
        "_nav": "\U0001f4d6 Guide",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Tier badges ───────────────────────────────────────────────────────────────

TIER_COLORS = {"I": TEAL, "II": GOLD, "III": PURPLE, "IV": RED}
TIER_LABELS = {
    "I":   "TIER I — Confirmed",
    "II":  "TIER II — Well-motivated",
    "III": "TIER III — Conjectural",
    "IV":  "TIER IV — Explicit gap",
}


def tier_badge(tier):
    color = TIER_COLORS.get(tier, TEAL)
    label = TIER_LABELS.get(tier, "")
    st.markdown(
        f'<span style="background:{color};color:#0b1120;border-radius:4px;'
        f'padding:2px 8px;font-size:0.75em;font-weight:bold;">{label}</span>',
        unsafe_allow_html=True,
    )


# ── Footer ────────────────────────────────────────────────────────────────────

FOOTER = (
    "**A note on feasibility:** Exotic matter requirements shown are physically real "
    "constraints.  At current technology, a macroscopic traversable wormhole is not "
    "buildable.  The tool demonstrates how parameters relate, not that the object "
    "is constructible."
)


def show_footer():
    st.divider()
    st.caption(FOOTER)


# ── Page: Guide ───────────────────────────────────────────────────────────────

def page_guide():
    render_guide(navigate_fn=navigate_to)


# ── Page: Overview ────────────────────────────────────────────────────────────

def page_overview():
    _help_banner("overview")

    st.markdown(
        "<h1 style='font-size:2.4em;margin-bottom:0;'>Wormhole Math Checker</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "#### Verify the mathematics of hypothetical traversable wormholes "
        "— no physics background required"
    )

    col_cta, col_snap = st.columns([3, 1])
    with col_cta:
        st.markdown(
            "This tool runs **automated checks** across 6 physics modules.  "
            "Every number comes from a named equation; every equation is "
            "cross-checked against a known analytic limit.  If all checks pass, "
            "the equations are mutually consistent."
        )
        if st.button("\U0001f4d6 Read the Guide first →", type="primary",
                     key="ov_guide_cta"):
            navigate_to("\U0001f4d6 Guide")

    M_cur  = float(st.session_state.get("M", 1.0))
    a_cur  = float(st.session_state.get("a_over_M", 0.85))
    r0_cur = float(st.session_state.get("r0", 1.2))
    sup    = float(kerr_suppression(a_cur * M_cur, M_cur))

    with col_snap:
        r0_si = _si_hint(r0_cur, "length", M_cur)
        st.metric("Kerr suppression", f"{sup:.4f}",
                  help="√(1 − a²/M²): fraction of exotic matter "
                       "needed vs non-rotating case")
        st.metric("Throat r₀", f"{r0_cur:.2f} r_g",
                  help=f"Wormhole throat radius.{' ≈ ' + r0_si if r0_si else ''}")

    st.divider()

    # Navigation cards
    st.markdown("### Where to go next")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("#### \U0001f52c Check the physics")
            st.markdown(
                "Run the automated verification suite and confirm the equations "
                "are self-consistent."
            )
            if st.button("Verification Suite →", key="ov_nav_verify"):
                navigate_to("\U0001f52c Verification Suite")
    with c2:
        with st.container(border=True):
            st.markdown("#### \U0001f300 Explore parameters")
            st.markdown(
                "Tune mass, spin, and throat radius.  See how exotic matter "
                "requirements change in real time."
            )
            if st.button("Kerr Explorer →", key="ov_nav_kerr"):
                navigate_to("\U0001f300 Kerr Explorer")
    with c3:
        with st.container(border=True):
            st.markdown("#### \U0001f4e1 Watch it oscillate")
            st.markdown(
                "Perturb the throat and watch it ring down.  Inspect the "
                "gravitational-wave echo spectrum."
            )
            if st.button("Throat & Echo →", key="ov_nav_echo"):
                navigate_to("\U0001f4e1 Throat & Echo")

    st.divider()
    st.markdown("### Current parameter snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mass M", f"{M_cur:.1f} M☉",
              help="Central mass in solar masses.  Scales all length and time units.")
    c2.metric("Spin a/M", f"{a_cur:.2f}",
              help="Dimensionless spin.  0 = Schwarzschild, 1 = extremal Kerr.")
    c3.metric("Throat r₀", f"{r0_cur:.2f} r_g",
              help="Wormhole throat radius in gravitational radii.")
    tau_k = float(tau_kerr(r0_cur, M_cur, a_cur * M_cur))
    c4.metric("τ_Kerr", f"{tau_k:.4e}",
              help="Exotic matter tension required, reduced by Kerr frame-dragging.")

    show_footer()


# ── Page: Verification Suite ──────────────────────────────────────────────────

def page_verification():
    _help_banner("verification")
    st.header("Verification Suite")

    with st.expander("\U0001f4cc What does this page do?"):
        st.markdown(
            "Runs every automated check across all physics modules.  "
            "Each check computes a quantity two independent ways and asserts they match.  "
            "A FAIL here means a software inconsistency, not a violation of physics.  "
            "All checks should be green before trusting numbers on other pages."
        )

    if st.button("Run All Checks", type="primary"):
        with st.spinner("Running checks…"):
            try:
                cr = run_all()
                st.session_state["last_checks"] = cr
                s = cr.summary()
                _auto_save("Verification", {},
                           {"n_pass": s["n_pass"], "n_fail": s["n_fail"],
                            "total": s["total"], "all_passed": s["all_passed"]})
            except Exception as e:
                st.error(f"Error running checks: {e}")
                return

    cr = st.session_state.get("last_checks")
    if cr is None:
        st.info("Press ‘Run All Checks’ to begin.")
        show_footer()
        return

    s = cr.summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total checks", s["total"],
              help="Number of mathematical assertions evaluated")
    c2.metric("Passed", s["n_pass"],
              help="Checks whose computed values agreed to required precision")
    c3.metric("Failed", s["n_fail"],
              help="Checks that found a discrepancy — investigate these first")

    import pandas as pd
    df = pd.DataFrame(cr.to_dataframe())

    def color_status(val):
        return f"color: {GREEN}" if val == "PASS" else f"color: {RED}"

    st.dataframe(df.style.map(color_status, subset=["Status"]),
                 use_container_width=True)

    if s["all_passed"]:
        st.success(
            f"All {s['total']} checks passed — physics is self-consistent"
        )
    else:
        st.error(f"{s['n_fail']} check(s) failed")

    with st.expander("What each group tests"):
        st.markdown(
            "**Kerr checks:** Horizon, ergosphere, ISCO, and frame-dragging formulae "
            "against textbook limits (a = 0 Schwarzschild, r → ∞).\n\n"
            "**Morris–Thorne checks:** Shape function satisfies the three canonical "
            "throat conditions: b(r₀) = r₀, b′ < 1, and b/r → 0.\n\n"
            "**NEC checks:** Null Energy Condition is violated at the throat — "
            "a *required* feature, not a bug, for wormhole support.\n\n"
            "**Throat Dynamics checks:** Analytic solution of the damped oscillator "
            "matches RK45 to 10⁻⁶ precision; regime classifier is correct.\n\n"
            "**Echo Spectrum checks:** H(0) = 0, peak at f₀, power-law falloff.\n\n"
            "**f(R) Algebra checks:** φ↔R round-trip, potential minimum at "
            "φ = 1, f′(0) = 1 (GR recovery).\n\n"
            "**Model Protocol checks:** GR and f(R) model objects implement "
            "`GravityModel` protocol with finite-valued outputs."
        )

    show_footer()


# ── Page: Kerr Explorer ───────────────────────────────────────────────────────

def page_kerr():
    _help_banner("kerr")
    st.header("Kerr Metric Explorer")
    tier_badge("I")

    with st.expander("\U0001f4cc Kerr metric — what you’re seeing"):
        st.markdown(
            "The Kerr metric describes a rotating black hole.  Spin introduces "
            "**frame-dragging** (nearby spacetime is dragged along) and an "
            "**ergosphere** (region where corotation is unavoidable).  "
            "Higher spin reduces the exotic matter needed to keep the wormhole "
            "open — the **suppression factor** √(1 − a²/M²) "
            "quantifies this reduction."
        )

    c_sl1, c_sl2 = st.columns(2)
    with c_sl1:
        M = numeric_input_with_steppers(
            "Mass M (M☉)", "ke_M", 0.1, 10.0,
            float(st.session_state.get("M", 1.0)), 0.1,
            format_str="%.1f",
            help_text="Central mass in solar masses.  Scales all geometric lengths and times.",
            unit_si_fn=lambda v: _si_hint(v, "mass", v),
        )
        a_frac = numeric_input_with_steppers(
            "Spin a/M", "ke_a", 0.0, 0.99,
            float(st.session_state.get("a_over_M", 0.85)), 0.01,
            format_str="%.2f",
            help_text="Dimensionless spin parameter.  0 = Schwarzschild, 0.99 = near-extremal.",
        )

    with c_sl2:
        r0 = numeric_input_with_steppers(
            "Throat radius r₀ (r_g)", "ke_r0", 0.5, 5.0,
            float(st.session_state.get("r0", 1.2)), 0.05,
            format_str="%.2f",
            help_text="Wormhole throat radius in gravitational radii.",
            unit_si_fn=lambda v: _si_hint(v, "length", M),
        )
        theta = numeric_input_with_steppers(
            "Polar angle θ (rad)", "ke_theta", 0.1, math.pi / 2,
            math.pi / 2, 0.05,
            format_str="%.2f",
            help_text="Polar angle for frame-dragging.  π/2 = equatorial plane (maximum effect).",
        )

    st.session_state["M"] = M
    st.session_state["a_over_M"] = a_frac
    st.session_state["r0"] = r0

    a = a_frac * M
    if a >= M:
        st.error("Spin must be < M to avoid a naked singularity")
        return

    try:
        fd    = float(frame_dragging(r0, theta, M, a))
        sup   = float(kerr_suppression(a, M))
        erg   = float(ergosphere_radius(theta, M, a))
        tau_k = float(tau_kerr(r0, M, a))
        tau_s = float(tau_static(r0))
        ratio = tau_k / (tau_s + EPS)

        fd_hint  = _si_hint(fd,    "frequency", M)
        erg_hint = _si_hint(erg,   "length",    M)
        tau_hint = _si_hint(tau_k, "tension",   M)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frame dragging ω", f"{fd:.5f}",
                  help=f"Angular velocity of spacetime at (r₀, θ).  "
                       f"{'SI: ' + fd_hint if fd_hint else 'Geometric units.'}")
        c2.metric("Kerr suppression", f"{sup:.4f}",
                  help="√(1 − a²/M²).  Fraction of GR exotic matter "
                       "still required; 1 = Schwarzschild, → 0 as a → M.")
        c3.metric("Ergosphere r_erg", f"{erg:.4f}",
                  help=f"Ergosphere radius at this θ.  "
                       f"{'SI: ' + erg_hint if erg_hint else 'Geometric units.'}")
        c4.metric("τ_Kerr / τ_static", f"{ratio:.4f}",
                  help="Should equal the suppression factor (self-consistency check).")
    except Exception as e:
        st.warning(f"Computation error: {e}")
        return

    col_l, col_r = st.columns(2)
    with col_l:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig_sup = plot_kerr_suppression(M, a, r0)
        st.pyplot(fig_sup)
        plt.close(fig_sup)
        st.caption(
            "Suppression factor vs spin.  The dashed line marks the current a/M.  "
            "Near-extremal spin (a/M → 1) reduces exotic matter to zero."
        )

    with col_r:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig_fd = plot_frame_dragging(M, a, r0)
        st.pyplot(fig_fd)
        plt.close(fig_fd)
        st.caption(
            "Frame-dragging angular velocity ω(r) in the equatorial plane.  "
            "Falls off as r⁻³ at large distances."
        )

    st.divider()
    st.markdown("### Exotic Matter Budget")

    with st.expander("What is the exotic matter budget?"):
        st.markdown(
            "The table shows how much exotic matter tension (τ_Kerr) is required "
            "at the throat for different spin values.  Higher spin → lower requirement.  "
            "The ‘Reduction %’ column shows the saving vs the Schwarzschild case."
        )

    import pandas as pd
    a_fracs = [0.0, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.99]
    budget_rows = []
    for af in a_fracs:
        av = af * M
        s_val = float(kerr_suppression(av, M))
        tk    = float(tau_kerr(r0, M, av))
        pct   = float(tau_reduction_percent(av, M))
        rp, _ = horizon_radii(M, av)
        if math.isnan(rp):
            regime = "Naked singularity"
        elif af < 0.5:
            regime = "Slow Kerr"
        elif af < 0.9:
            regime = "Fast Kerr"
        else:
            regime = "Near-extremal"
        budget_rows.append({
            "a/M": af,
            "Suppression": round(s_val, 4),
            "tau_Kerr": f"{tk:.5e}",
            "Reduction %": round(pct, 2),
            "Regime": regime,
        })
    st.dataframe(pd.DataFrame(budget_rows), use_container_width=True, hide_index=True)

    show_footer()


# ── Page: Wormhole & NEC ──────────────────────────────────────────────────────

def page_wormhole_nec():
    _help_banner("wormhole")
    st.header("Wormhole Geometry & NEC Analysis")
    tier_badge("I")

    with st.expander("\U0001f4cc What you’re seeing on this page"):
        st.markdown(
            "Checks three throat conditions on the chosen **shape function b(r)**.  "
            "Plots the NEC violation profile (exotic matter is where ρ + p_r < 0) "
            "and the Flamm embedding diagram.  "
            "The exotic matter budget compares the required tension to the Casimir lab maximum."
        )

    c_l, c_r = st.columns(2)
    with c_l:
        r0 = numeric_input_with_steppers(
            "Throat radius r₀ (r_g)", "wn_r0", 0.3, 5.0,
            float(st.session_state.get("r0", 1.2)), 0.05,
            format_str="%.2f",
            help_text="Wormhole throat radius in gravitational radii.  "
                      "Determines where exotic matter is concentrated.",
            unit_si_fn=lambda v: _si_hint(v, "length", st.session_state.get("M", 1.0)),
        )
        shape = st.selectbox(
            "Shape function b(r)",
            ["power", "constant", "power_law", "visser"],
            key="wn_shape",
            help="Mathematical form of the wormhole geometry.  "
                 "‘power’ (b = r₀²/r) is the Morris–Thorne original.",
        )

    with c_r:
        M = numeric_input_with_steppers(
            "Mass M (M☉)", "wn_M", 0.1, 10.0,
            float(st.session_state.get("M", 1.0)), 0.1,
            format_str="%.1f",
            help_text="Central mass.  Scales the gravitational radius and all SI conversions.",
            unit_si_fn=lambda v: _si_hint(v, "mass", v),
        )
        a_frac = numeric_input_with_steppers(
            "Spin a/M", "wn_a", 0.0, 0.99,
            float(st.session_state.get("a_over_M", 0.85)), 0.01,
            format_str="%.2f",
            help_text="Spin reduces the exotic matter requirement by the suppression factor.",
        )

    st.session_state["r0"] = r0
    st.session_state["M"]  = M
    st.session_state["a_over_M"] = a_frac

    if r0 <= 0:
        st.error("Throat radius must be positive")
        return

    try:
        cond = wormhole_conditions(r0, shape)
    except Exception as e:
        st.warning(f"Could not evaluate conditions: {e}")
        return

    def badge(ok, label):
        color = GREEN if ok else RED
        sym   = "PASS" if ok else "FAIL"
        st.markdown(
            f'<span style="background:{color};color:#0b1120;border-radius:4px;'
            f'padding:4px 12px;font-weight:bold;">{sym} — {label}</span>',
            unsafe_allow_html=True,
        )

    b1, b2, b3 = st.columns(3)
    with b1:
        badge(cond["throat"],     f"b(r₀) = {cond['b_r0']:.4f} = r₀")
    with b2:
        badge(cond["flare_out"],  f"b′(r₀) = {cond['b_prime_r0']:.4f} < 1")
    with b3:
        badge(cond["asymptotic"], "b/r → 0 asymptotically")

    st.caption(
        "All three conditions must be PASS for a valid traversable wormhole.  "
        "b(r₀) = r₀ sets the throat; b′ < 1 ensures the wormhole "
        "‘opens’ outward; asymptotic flatness keeps spacetime normal far away."
    )

    st.divider()
    st.markdown("### Null Energy Condition Profile")

    with st.expander("What does NEC violation mean?"):
        st.markdown(
            "The Null Energy Condition (NEC) is ρ + p_r ≥ 0 for all normal matter.  "
            "Red shading shows where NEC is **violated** — this is where exotic matter "
            "must be present to hold the wormhole open.  The violation is concentrated "
            "near the throat and falls off with distance."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig_nec = plot_nec(r0, shape, M, a_frac * M)
        st.pyplot(fig_nec)
        plt.close(fig_nec)
        st.caption(
            "Left: NEC quantity ρ + p_r.  Red fill = NEC violated = exotic matter present.  "
            "Right: individual stress-energy components (ρ, p_r, p_t)."
        )
    except Exception as e:
        st.warning(f"NEC plot error: {e}")

    st.divider()
    st.markdown("### Flamm Embedding Diagram")

    with st.expander("Reading the embedding diagram"):
        st.markdown(
            "This 2D cross-section shows the wormhole throat connecting two flat universes.  "
            "The vertical height z(r) = 2√(r − r₀) visualises curvature — "
            "the steeper the walls, the stronger the gravity near the throat.  "
            "Upper and lower sheets represent the two sides of the wormhole."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig_flamm = plot_flamm(r0, shape)
        st.pyplot(fig_flamm)
        plt.close(fig_flamm)
        st.caption(
            "Flamm paraboloid embedding of the equatorial slice.  "
            "The throat (gold dashed line) is where the two sheets meet."
        )
    except Exception as e:
        st.warning(f"Embedding plot error: {e}")

    st.divider()
    st.markdown("### Exotic Matter Budget vs Casimir Lab Limits")
    tier_badge("IV")

    with st.expander("Why compare to the Casimir effect?"):
        st.markdown(
            "The Casimir effect is the largest NEC-violating energy density "
            "produced in a laboratory (∼ 10⁻³ J/m² for closely spaced "
            "plates).  The wormhole requires orders of magnitude more.  "
            "The ‘feasibility gap’ tells you how many powers of ten separate "
            "current technology from what would be needed."
        )

    try:
        a_val = a_frac * M
        ex = exotic_NEC_with_kerr(r0, M, a_val)
        tau_req = ex["tau_required"]
        orders = math.log10(max(tau_req / CASIMIR_LAB_MAX, 1e-20))
        tau_hint = _si_hint(tau_req, "tension", M)

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "τ_required (geom)", f"{tau_req:.4e}",
            help=f"Exotic matter tension at throat.  "
                 f"{'SI: ' + tau_hint if tau_hint else 'Geometric units.'}",
        )
        c2.metric("Kerr reduction", f"{ex['reduction_pct']:.1f}%",
                  help="Reduction in exotic matter requirement due to spin.")
        c3.metric(
            "Feasibility gap (decades)",
            f"{orders:.1f}" if tau_req > CASIMIR_LAB_MAX else "Within lab range!",
            help="Powers of 10 between the required tension and the Casimir lab maximum.",
        )
    except Exception as e:
        st.warning(f"Budget computation error: {e}")

    show_footer()


# ── Page: Throat & Echo ───────────────────────────────────────────────────────

def page_throat_echo():
    _help_banner("echo")
    st.header("Dynamic Throat (Eq 6.3) & GW Echo Spectrum (Eq 9b)")
    tier_badge("II")

    st.latex(r"\ddot{\delta a} + 2\eta_s \dot{\delta a} + \frac{\sigma}{a_0^2}\delta a = 0")
    st.latex(
        r"\hat{H}(f) = A_0 \cdot \left(\frac{f}{f_0}\right)^2 \cdot "
        r"e^{-\eta_s f/f_0^2} \cdot \left[1+\left(\frac{f}{f_0}\right)^2\right]^{-1}"
    )

    with st.expander("\U0001f4cc Throat oscillation and GW echoes — background"):
        st.markdown(
            "A small perturbation δa to the throat radius obeys a **damped harmonic "
            "oscillator** equation (Eq 6.3).  The stiffness σ and damping η_s "
            "determine whether the throat **rings** (underdamped), **creeps back** "
            "(overdamped), or follows the fastest no-ring return (critical).  "
            "Each bounce inside the wormhole produces a gravitational-wave echo whose "
            "spectrum H(f) peaks at f₀ = ω₀ / 2π (Eq 9b)."
        )

    c_l, c_r = st.columns(2)
    with c_l:
        sigma_throat = numeric_input_with_steppers(
            "σ_throat (stiffness)", "te_sig", 0.01, 2.0,
            float(st.session_state.get("sigma_throat", 0.40)), 0.01,
            format_str="%.2f",
            help_text="Throat spring constant.  Higher σ → faster oscillation frequency.",
        )
        a0 = numeric_input_with_steppers(
            "a₀ (equilibrium throat radius)", "te_a0", 0.3, 4.0, 1.2, 0.05,
            format_str="%.2f",
            help_text="Equilibrium throat radius.  Larger a₀ → lower natural frequency.",
            unit_si_fn=lambda v: _si_hint(v, "length", st.session_state.get("M", 1.0)),
        )
        eta_s = numeric_input_with_steppers(
            "η_s (damping coefficient)", "te_eta", 0.01, 1.0,
            float(st.session_state.get("eta_s", 0.15)), 0.01,
            format_str="%.2f",
            help_text="Damping strength.  η_s < ω₀ = underdamped; "
                      "η_s > ω₀ = overdamped.",
        )

    with c_r:
        delta_a0 = numeric_input_with_steppers(
            "δa₀ (initial displacement)", "te_da", 0.01, 0.5, 0.10, 0.01,
            format_str="%.2f",
            help_text="Size of the initial throat perturbation.  "
                      "Linear regime: keep this small (< 0.2).",
        )
        t_max = numeric_input_with_steppers(
            "t_max (integration time)", "te_tmax", 10, 100, 50, 5,
            format_str="%.0f",
            help_text="How long to integrate the ODE.  Increase to see late-time decay.",
        )

    st.session_state["sigma_throat"] = sigma_throat
    st.session_state["eta_s"] = eta_s

    try:
        omega0 = natural_frequency(sigma_throat, a0)
        regime, omega_d = damping_regime(sigma_throat, a0, eta_s)
        f0_hz = _si_hint(omega0 / (2 * math.pi), "frequency",
                         st.session_state.get("M", 1.0))

        regime_colors = {"UNDERDAMPED": TEAL, "CRITICAL": GOLD, "OVERDAMPED": RED}
        rclr = regime_colors.get(regime, TEAL)

        rm1, rm2, rm3 = st.columns(3)
        rm1.metric("ω₀ (natural freq)", f"{omega0:.4f}",
                   help=f"Natural angular frequency of throat oscillation.  "
                        f"{'SI: ' + f0_hz if f0_hz else ''}")
        rm2.metric("ω_d (damped freq)",
                   f"{omega_d:.4f}" if omega_d > 0 else "N/A",
                   help="Actual oscillation frequency accounting for damping.  "
                        "N/A for overdamped.")
        rm3.markdown(
            f'<span style="background:{rclr};color:#0b1120;border-radius:4px;'
            f'padding:4px 12px;font-weight:bold;font-size:1.1em;">{regime}</span>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Frequency computation error: {e}")
        show_footer()
        return

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig_t = plot_throat(sigma_throat, a0, eta_s, delta_a0, int(t_max))
            st.pyplot(fig_t)
            plt.close(fig_t)
            st.caption(
                "Throat displacement δa(t).  Solid: RK45 numerical; "
                "dashed: analytic solution.  The residual (top-left) should be < 10⁻⁶."
            )
        except Exception as e:
            st.warning(f"Throat plot error: {e}")

    with col_r:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig_e = plot_echo(sigma_throat, a0, eta_s)
            st.pyplot(fig_e)
            plt.close(fig_e)
            st.caption(
                "GW echo spectrum H(f).  The peak at f₀ (gold dashed) is the "
                "characteristic echo frequency.  Wider peaks = more damping."
            )
        except Exception as e:
            st.warning(f"Echo plot error: {e}")

    st.divider()
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig_rm = plot_regime_map(a0, sigma_throat)
            st.pyplot(fig_rm)
            plt.close(fig_rm)
            st.caption(
                "Regime map over (a₀, σ) parameter space with η_s = 0.15 fixed.  "
                "The white star marks the current parameters."
            )
        except Exception as e:
            st.warning(f"Regime map error: {e}")

    with col_r2:
        st.markdown("### Echo Properties")
        with st.expander("What do these numbers mean?"):
            st.markdown(
                "**f₀**: fundamental echo frequency (Hz in SI).  "
                "**Δt echo**: time between successive echo pulses.  "
                "**n_echoes**: estimated number of detectable echoes before damping kills the signal.  "
                "**stability_index**: ratio quantifying how quickly the throat returns to equilibrium."
            )
        try:
            report = israel_junction_report(a0, sigma_throat, eta_s)
            si = stability_index(sigma_throat, a0, eta_s)
            dt = report["dt_echo"]
            f0_hint = _si_hint(report["f0"], "frequency",
                               st.session_state.get("M", 1.0))
            st.table({
                "Property":  ["f₀", "Δt echo", "n_echoes", "stability_index"],
                "Value (geom)": [
                    f"{report['f0']:.5f}",
                    f"{dt:.4f}" if not math.isinf(dt) else "∞",
                    str(report["n_echoes"]),
                    f"{si:.3f}",
                ],
                "SI": [
                    f0_hint or "—",
                    _si_hint(dt, "time", st.session_state.get("M", 1.0)) or "—"
                    if not math.isinf(dt) else "—",
                    "—", "—",
                ],
            })
        except Exception as e:
            st.warning(f"Echo table error: {e}")

        st.markdown("**η_s sensitivity**")
        try:
            f0 = echo_frequency(sigma_throat, a0)
            f_arr = np.linspace(0, max(8 * f0, 0.1), 200)
            fig_eta, ax_eta = _new_fig(figsize=(5, 3))
            for eta_v in [0.05, 0.15, 0.30, 0.50, 0.80]:
                amps = echo_spectrum(f_arr, 1.0, f0, eta_v)
                ax_eta.plot(f_arr, amps, lw=1.2, label=f"η={eta_v}")
            _style_ax(ax_eta, "Echo Spectrum — η_s sweep")
            ax_eta.set_xlabel("f")
            ax_eta.set_ylabel("H(f)")
            ax_eta.legend(facecolor=BG2, labelcolor=TEXT, fontsize=7)
            fig_eta.tight_layout()
            st.pyplot(fig_eta)
            plt.close(fig_eta)
            st.caption("All five η_s values share the same f₀.  "
                       "Higher damping broadens the peak and reduces amplitude.")
        except Exception as e:
            st.warning(f"η_s sweep error: {e}")

    show_footer()


# ── Page: f(R) Gravity ────────────────────────────────────────────────────────

def page_fR():
    _help_banner("fr")
    st.header("f(R) = R + αR² Gravity")
    tier_badge("III")

    st.latex(r"f(R) = R + \alpha R^2")
    st.latex(r"\phi = \frac{df}{dR} = 1 + 2\alpha R")
    st.latex(r"V(\phi) = \frac{(\phi-1)^2}{4\alpha}")

    with st.expander("\U0001f4cc f(R) gravity — background"):
        st.markdown(
            "In Starobinsky’s f(R) model, the Ricci scalar R in the action is "
            "replaced by R + αR².  This introduces an extra scalar degree "
            "of freedom φ = df/dR = 1 + 2αR that propagates through the "
            "Morris–Thorne background and modifies the effective stress-energy.  "
            "At α = 0 the theory reduces exactly to GR (φ = 1 everywhere).  "
            "Larger α enhances the effective NEC violation, potentially reducing "
            "the exotic matter requirement."
        )

    c_l, c_r = st.columns(2)
    with c_l:
        alpha = numeric_input_with_steppers(
            "α (Starobinsky parameter)", "fr_alpha", 0.01, 2.0,
            float(st.session_state.get("alpha_fR", 0.15)), 0.01,
            format_str="%.2f",
            help_text="f(R) coupling constant.  α = 0 recovers GR.  "
                      "Larger α → stronger scalar field effect.",
        )
        phi0 = numeric_input_with_steppers(
            "φ₀ (field value at throat)", "fr_phi0", 0.8, 1.5,
            float(st.session_state.get("phi0", 1.05)), 0.01,
            format_str="%.2f",
            help_text="φ(r₀): initial scalar field value.  "
                      "φ = 1 is the GR fixed point.  "
                      "Use the Shooting Method to find the φ₀ that gives "
                      "φ → 1 at large r.",
        )

    with c_r:
        r0_fr = numeric_input_with_steppers(
            "Throat radius r₀ (r_g)", "fr_r0", 0.5, 3.0,
            float(st.session_state.get("r0", 1.2)), 0.05,
            format_str="%.2f",
            help_text="Wormhole throat radius.  ODE integration starts here + small offset.",
            unit_si_fn=lambda v: _si_hint(v, "length", st.session_state.get("M", 1.0)),
        )
        r_max = numeric_input_with_steppers(
            "r_max (outer boundary)", "fr_rmax", 10, 50, 30, 1,
            format_str="%.0f",
            help_text="Radial extent of the ODE integration.  "
                      "Larger r_max gives a stricter asymptotic-flatness test.",
        )

    if alpha <= 0:
        st.error("alpha must be positive for f(R) = R + αR²")
        return
    if r0_fr <= 0:
        st.error("Throat radius must be positive")
        return

    st.session_state["alpha_fR"] = alpha
    st.session_state["phi0"]     = phi0

    if st.button("Solve Scalar Field ODE", type="primary"):
        with st.spinner("Integrating scalar field ODE…"):
            try:
                r_arr, phi_arr, dphi_arr, sol = solve_scalar_field(
                    r0_fr, int(r_max), alpha, phi0, shape="power"
                )
                st.session_state["fR_solution"] = {
                    "r": r_arr, "phi": phi_arr, "dphi": dphi_arr,
                    "success": sol.success,
                    "r0": r0_fr, "alpha": alpha,
                }
                if sol.success:
                    st.success("ODE solved successfully.")
                else:
                    st.warning("ODE solver did not fully converge.")
                _auto_save(
                    "fR",
                    {"r0": r0_fr, "r_max": int(r_max), "alpha": alpha, "phi0": phi0},
                    {"phi_final": float(phi_arr[-1]) if len(phi_arr) > 0 else None,
                     "success": sol.success},
                    {"n_steps": len(r_arr), "method": "RK45"},
                )
            except Exception as e:
                st.error(f"ODE solve failed: {e}")

    sol_data = st.session_state.get("fR_solution")
    if sol_data is not None:
        r_arr  = sol_data["r"]
        phi_arr = sol_data["phi"]

        col_l, col_r = st.columns(2)
        with col_l:
            try:
                fig_phi = plot_fR_phi(r_arr, phi_arr, sol_data["r0"], sol_data["alpha"])
                st.pyplot(fig_phi)
                plt.close(fig_phi)
                st.caption(
                    "Left: scalar field φ(r).  Should approach 1 (GR limit) at large r.  "
                    "Right: Ricci scalar R(r) that sources φ."
                )
            except Exception as e:
                st.warning(f"φ plot error: {e}")

        with col_r:
            try:
                phi_r0 = float(phi_arr[0]) if len(phi_arr) > 0 else phi0
                fig_pot = plot_potential(sol_data["alpha"], phi0_mark=phi_r0)
                st.pyplot(fig_pot)
                plt.close(fig_pot)
                st.caption(
                    "Scalar potential V(φ) = (φ−1)² / (4α).  "
                    "The teal dot marks the field value at the throat."
                )
            except Exception as e:
                st.warning(f"Potential plot error: {e}")

        st.divider()
        col_l2, col_r2 = st.columns(2)
        with col_l2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fig_nec = plot_fR_nec(sol_data["r0"], sol_data["alpha"], "power")
                st.pyplot(fig_nec)
                plt.close(fig_nec)
                st.caption(
                    "Effective NEC with f(R) corrections: ρ_eff + p_r_eff.  "
                    "Red fill = exotic matter region.  Compare to GR: if the red region "
                    "shrinks, f(R) has reduced the exotic matter requirement."
                )
            except Exception as e:
                st.warning(f"Effective NEC plot error: {e}")

        with col_r2:
            phi_r0_val = float(phi_arr[0]) if len(phi_arr) > 0 else phi0
            phi_inf_val = float(phi_arr[-1]) if len(phi_arr) > 0 else None
            phi_hint = _si_hint(phi_r0_val - 1.0, "energy", 1.0)

            c1, c2 = st.columns(2)
            c1.metric("φ(r₀)", f"{phi_r0_val:.5f}",
                      help="Scalar field at the throat.  Deviations from 1 drive the f(R) correction.")
            if phi_inf_val is not None:
                c2.metric("φ(r_max)", f"{phi_inf_val:.5f}",
                          help="Scalar field at the outer boundary.  Should be close to 1 "
                               "(GR limit) for a valid solution.")
    else:
        st.info("Press ‘Solve Scalar Field ODE’ to compute the scalar field.")

    st.divider()
    st.markdown("### Shooting Method — Find φ₀ such that φ(r_max) → 1")

    with st.expander("What is the shooting method doing?"):
        st.markdown(
            "The ODE is an initial-value problem but the *boundary condition* is at r_max "
            "(φ → 1).  The shooting method guesses φ₀, integrates the "
            "ODE, checks how far φ(r_max) is from 1, and iterates (using Brent’s "
            "method) until the residual is small.  The result is the ‘correct’ "
            "initial condition for the asymptotically flat solution."
        )

    if st.button("Find φ₀ s.t. φ(∞) → 1"):
        with st.spinner("Running shooting method…"):
            try:
                phi0_best, residual, r_s, phi_s, _ = shoot_phi0(
                    r0_fr, int(r_max), alpha, shape="power"
                )
                converged = residual < 0.05
                color = GREEN if converged else GOLD
                label = "Converged" if converged else "Best approximation"
                st.markdown(
                    f"**φ₀_best** = {phi0_best:.5f} &nbsp;&nbsp; "
                    f"**residual** = {residual:.4e} &nbsp;&nbsp; "
                    f'<span style="background:{color};color:#0b1120;border-radius:4px;'
                    f'padding:2px 8px;font-size:0.85em;font-weight:bold;">{label}</span>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Shooting failed: {e}")

    st.divider()
    st.markdown("### α Sweep — Min Effective NEC vs α")

    with st.expander("What does this sweep show?"):
        st.markdown(
            "For each α from 0.01 to 1.5, the effective NEC is computed over "
            "r ∈ [r₀, 8r₀] and the minimum value is plotted.  "
            "Red dots are α values where NEC is violated (exotic matter present).  "
            "The gold line marks the current α."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig_sweep = plot_alpha_sweep(r0_fr, "power", alpha)
        st.pyplot(fig_sweep)
        plt.close(fig_sweep)
        st.caption(
            "Red dots: NEC violated (α values that support a wormhole in f(R) gravity).  "
            "Dim dots: NEC satisfied (no exotic matter support).  "
            "Gold line: currently selected α."
        )
    except Exception as e:
        st.warning(f"Alpha sweep error: {e}")

    show_footer()


# ── Page: History ─────────────────────────────────────────────────────────────

def page_history():
    _help_banner("history")
    st.header("Run History")

    if not _STORAGE_OK:
        st.warning("Storage module unavailable — history disabled.")
        show_footer()
        return

    runs = load_all_runs()

    with st.sidebar:
        st.markdown("### Filters")
        all_models = sorted({r.model for r in runs}) if runs else []
        model_filter = st.selectbox("Model", ["All"] + all_models)
        date_from   = st.text_input("Date from (YYYY-MM-DD)", "")
        date_to     = st.text_input("Date to   (YYYY-MM-DD)", "")
        tag_filter  = st.text_input("Tag contains", "")
        nec_only    = st.checkbox("NEC violated runs only")

    filtered = runs
    if model_filter != "All":
        filtered = [r for r in filtered if r.model == model_filter]
    if date_from:
        filtered = [r for r in filtered if r.timestamp >= date_from]
    if date_to:
        filtered = [r for r in filtered if r.timestamp <= date_to]
    if tag_filter:
        filtered = [r for r in filtered if tag_filter in r.tags]
    if nec_only:
        filtered = [r for r in filtered if r.results.get("nec_at_throat", 0) < 0
                    or r.results.get("nec_violation", 0) < 0]

    if not filtered:
        st.info("No runs recorded yet.  Use Solve / Run Checks on any page to auto-save.")
        show_footer()
        return

    import pandas as pd
    rows = [r.flat_dict() for r in filtered]
    df = pd.DataFrame(rows)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total runs", len(filtered))
    c2.metric("Models", len({r.model for r in filtered}))
    c3.metric("Latest", filtered[0].timestamp[:16] if filtered else "—")

    st.divider()
    st.dataframe(df, use_container_width=True)

    st.divider()
    st.markdown("### Diff — compare two runs")
    run_labels = [f"{r.run_id} | {r.model} | {r.timestamp[:16]}" for r in filtered]
    run_ids    = [r.run_id for r in filtered]

    if len(run_ids) >= 2:
        col_l, col_r = st.columns(2)
        with col_l:
            pick_a = st.selectbox("Run A", run_labels, index=0, key="hist_a")
        with col_r:
            pick_b = st.selectbox("Run B", run_labels,
                                   index=min(1, len(run_labels) - 1), key="hist_b")

        idx_a = run_labels.index(pick_a)
        idx_b = run_labels.index(pick_b)
        ra, rb = filtered[idx_a], filtered[idx_b]

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**{ra.run_id}** — {ra.model} — {ra.timestamp[:16]}")
            st.json({"params": ra.params, "results": ra.results}, expanded=False)
        with col_r:
            st.markdown(f"**{rb.run_id}** — {rb.model} — {rb.timestamp[:16]}")
            st.json({"params": rb.params, "results": rb.results}, expanded=False)

    st.divider()
    st.markdown("### Export")
    col_l, col_r = st.columns(2)
    with col_l:
        try:
            from storage.json_backend import JSONBackend
            backend = JSONBackend()
            st.download_button(
                "Download selected as JSON",
                data=backend.export_json([r.run_id for r in filtered]),
                file_name="wormhole_runs.json",
                mime="application/json",
            )
        except Exception as e:
            st.warning(f"JSON export unavailable: {e}")
    with col_r:
        try:
            from storage.json_backend import JSONBackend
            backend = JSONBackend()
            st.download_button(
                "Download selected as CSV",
                data=backend.export_csv([r.run_id for r in filtered]),
                file_name="wormhole_runs.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.warning(f"CSV export unavailable: {e}")

    st.divider()
    st.markdown("### Delete a run")
    del_id = st.selectbox("Run to delete", ["— select —"] + run_ids,
                           key="hist_del")
    if del_id != "— select —":
        if st.button(f"Delete {del_id}", type="secondary"):
            try:
                from storage.json_backend import JSONBackend
                JSONBackend().delete(del_id)
                st.success(f"Deleted run {del_id}")
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")

    show_footer()


# ── App entry point ───────────────────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="Wormhole Math Checker",
    page_icon="\U0001f300",
)

_init_state()

PAGES = {
    "\U0001f4d6 Guide":               page_guide,
    "\U0001f3e0 Overview":            page_overview,
    "\U0001f52c Verification Suite":  page_verification,
    "\U0001f300 Kerr Explorer":       page_kerr,
    "\U0001f4d0 Wormhole & NEC":      page_wormhole_nec,
    "\U0001f4e1 Throat & Echo":       page_throat_echo,
    "\U0001f52d f(R) Gravity":        page_fR,
    "\U0001f4da History":             page_history,
}

page_names = list(PAGES.keys())

# Keep radio in sync with _nav (navigate_to sets both)
if st.session_state.get("_nav") in page_names:
    st.session_state["_radio_nav"] = st.session_state["_nav"]

with st.sidebar:
    # Units toggle — top of sidebar
    st.markdown("### \U0001f4d0 Display units")
    du_options = ["Geometric", "Both", "Metric"]
    du_idx = du_options.index(st.session_state.get("display_units", "Geometric"))
    selected_du = st.radio(
        "units",
        du_options,
        index=du_idx,
        key="_du_radio",
        label_visibility="collapsed",
        help="Geometric: G=c=1 units.  Metric: SI equivalents.  Both: side-by-side.",
    )
    st.session_state["display_units"] = selected_du

    st.divider()
    st.markdown("### Navigation")
    selected = st.radio(
        "",
        page_names,
        label_visibility="collapsed",
        key="_radio_nav",
    )
    st.session_state["_nav"] = selected

    st.divider()
    st.caption("Wormhole Math Checker v2.1")
    st.caption("Against Chronology Protection")

PAGES[selected]()
