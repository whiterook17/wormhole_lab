"""Wormhole Math Checker — Streamlit entry point."""

import math
import warnings

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
from verification.run_checks import run_all

# Storage (history system — never crashes the app on failure)
try:
    from storage.run import Run
    from storage import save_run, load_all_runs
    _STORAGE_OK = True
except Exception:
    _STORAGE_OK = False
    def save_run(run): pass
    def load_all_runs(): return []

# ── Auto-save helper ──────────────────────────────────────────────────────────

def _auto_save(model_name: str, params: dict, results: dict,
               convergence: dict | None = None) -> None:
    """Create and persist a Run record.  Silent on failure."""
    if not _STORAGE_OK:
        return
    try:
        run = Run.create(model_name, params, results, convergence)
        save_run(run)
    except Exception:
        pass


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
    ax.axvline(ax_frac, color=GOLD, lw=1.4, linestyle="--", label=f"a/M={ax_frac:.2f}")
    ax.annotate(f"{sup:.3f}", xy=(ax_frac, sup), xytext=(ax_frac + 0.05, sup + 0.05),
                color=GOLD, fontsize=9)
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
    ax.plot(r_arr, z, color=TEAL, lw=2, label="Upper sheet z(r)")
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
    im = ax.pcolormesh(a0_vals, sig_vals, Z, cmap=cmap, shading="auto",
                       vmin=0, vmax=2)
    ax.scatter([a0_cur], [sigma_cur], color="white", s=100, zorder=5,
               marker="*", label="Current")
    _style_ax(ax, "Regime Map (eta_s=0.15 fixed)")
    ax.set_xlabel("a0")
    ax.set_ylabel("sigma_throat")
    ax.legend(facecolor=BG2, labelcolor=TEXT, fontsize=8)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GREEN, label="Underdamped"),
        Patch(facecolor=GOLD, label="Critical"),
        Patch(facecolor=RED, label="Overdamped"),
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
    from physics.morris_thorne import SHAPE_FUNCTIONS
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
    if "fR_solution" not in st.session_state:
        st.session_state["fR_solution"] = None
    if "last_checks" not in st.session_state:
        st.session_state["last_checks"] = None


# ── Tier badges ───────────────────────────────────────────────────────────────

TIER_COLORS = {
    "I": TEAL, "II": GOLD, "III": PURPLE, "IV": RED,
}
TIER_LABELS = {
    "I": "TIER I — Confirmed",
    "II": "TIER II — Well-motivated",
    "III": "TIER III — Conjectural",
    "IV": "TIER IV — Explicit gap",
}


def tier_badge(tier):
    color = TIER_COLORS.get(tier, TEAL)
    label = TIER_LABELS.get(tier, "")
    return st.markdown(
        f'<span style="background:{color};color:#0b1120;border-radius:4px;'
        f'padding:2px 8px;font-size:0.75em;font-weight:bold;">{label}</span>',
        unsafe_allow_html=True,
    )


# ── Footer ────────────────────────────────────────────────────────────────────

FOOTER = (
    "**A note on feasibility:** Exotic matter requirements shown are physically real "
    "constraints. At current technology, a macroscopic traversable wormhole is not "
    "buildable. The tool demonstrates how parameters relate, not that the object "
    "is constructible."
)


def show_footer():
    st.divider()
    st.caption(FOOTER)


# ── Page 0 — Overview ─────────────────────────────────────────────────────────

def page_overview():
    st.title("Wormhole Math Checker")
    st.subheader("Against Chronology Protection — Physics Verification Tool")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equations verified", "41")
    c2.metric("Physics modules", "6")
    c3.metric("Gravity models", "2 (GR + f(R))")
    c4.metric("Damping regimes", "3")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### What this tool does")
        st.markdown(
            "- Verifies Kerr metric horizon, ergosphere, and ISCO formulae\n"
            "- Checks Morris-Thorne throat conditions and NEC violation profiles\n"
            "- Quantifies exotic matter reduction from Kerr frame-dragging\n"
            "- Solves and verifies damped throat oscillation (Eq 6.3)\n"
            "- Computes GW echo spectrum (Eq 9b) and damping regimes\n"
            "- Integrates the f(R) scalar field ODE in MT background\n"
            "- Runs 41 self-consistency checks across all modules\n"
            "- Compares exotic matter density to 2026 Casimir lab limits"
        )

    with col_r:
        st.markdown("### Quick start")
        M_qs = st.slider("Mass M", 0.1, 10.0, float(st.session_state["M"]),
                         0.1, key="ov_M")
        a_qs = st.slider("Spin a/M", 0.0, 0.99,
                         float(st.session_state["a_over_M"]), 0.01, key="ov_a")
        st.session_state["M"] = M_qs
        st.session_state["a_over_M"] = a_qs
        a_val = a_qs * M_qs
        sup = kerr_suppression(a_val, M_qs)
        tau_k = tau_kerr(float(st.session_state["r0"]), M_qs, a_val)
        st.metric("Suppression factor", f"{sup:.4f}")
        st.metric("tau_Kerr", f"{tau_k:.6f}")

    st.divider()
    st.markdown("### Kerr Suppression Curve")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plot_kerr_suppression(
            st.session_state["M"],
            a_qs * st.session_state["M"],
            float(st.session_state["r0"]),
        )
    st.pyplot(fig)
    plt.close(fig)

    show_footer()


# ── Page 1 — Verification Suite ───────────────────────────────────────────────

def page_verification():
    st.header("Verification Suite — 41 Checks")

    if st.button("Run All Checks", type="primary"):
        with st.spinner("Running checks..."):
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
        st.info("Press 'Run All Checks' to begin.")
        show_footer()
        return

    s = cr.summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total checks", s["total"])
    c2.metric("Passed", s["n_pass"], delta=None)
    c3.metric("Failed", s["n_fail"], delta=None)

    rows = cr.to_dataframe()

    import pandas as pd
    df = pd.DataFrame(rows)

    def color_status(val):
        if val == "PASS":
            return f"color: {GREEN}"
        return f"color: {RED}"

    styled = df.style.applymap(color_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True)

    if s["all_passed"]:
        st.success(f"All {s['total']} checks passed — physics is self-consistent")
    else:
        st.error(f"{s['n_fail']} checks failed")

    with st.expander("What each check means"):
        st.markdown(
            "**Kerr (15 checks):** Confirms horizon, ergosphere, ISCO, and "
            "frame-dragging formulae match textbook limits (a=0 Schwarzschild, r→∞). "
            "Verifies the suppression factor reduces exotic matter at high spin.\n\n"
            "**Morris-Thorne (3 checks):** Confirms the shape function satisfies the "
            "three canonical throat conditions: b(r0)=r0, b'<1, and b/r→0.\n\n"
            "**NEC (2 checks):** Confirms the Null Energy Condition is violated at "
            "the throat — this is a *required* feature, not a bug, for wormhole support.\n\n"
            "**Throat Dynamics (6 checks):** Verifies the analytic solution of the "
            "damped oscillator matches RK45 numerical integration to 1e-6 precision. "
            "Checks regime classification and decay to zero.\n\n"
            "**Echo Spectrum (3 checks):** Confirms H(0)=0 (no DC component), "
            "the peak at f0, and falloff at high frequencies.\n\n"
            "**f(R) Algebra (6 checks):** Round-trips phi<->R, checks potential "
            "minimum at phi=1, and confirms f'(0)=1 (GR recovery).\n\n"
            "**Self-Consistency (4 checks):** Confirms tau_kerr/tau_static == "
            "kerr_suppression at a=0, 0.5M, 0.9M, 0.99M."
        )

    show_footer()


# ── Page 2 — Kerr Explorer ────────────────────────────────────────────────────

def page_kerr():
    st.header("Kerr Metric Explorer")
    tier_badge("I")

    c_sl1, c_sl2 = st.columns(2)
    with c_sl1:
        M = st.slider("Mass M", 0.1, 10.0, float(st.session_state["M"]),
                      0.1, key="ke_M")
        a_frac = st.slider("Spin a/M", 0.0, 0.99,
                           float(st.session_state["a_over_M"]), 0.01, key="ke_a")
    with c_sl2:
        r0 = st.slider("Throat radius r0", 0.5, 5.0,
                       float(st.session_state["r0"]), 0.05, key="ke_r0")
        theta = st.slider("Polar angle theta (rad)", 0.1, math.pi / 2,
                           math.pi / 2, 0.05, key="ke_theta")

    st.session_state["M"] = M
    st.session_state["a_over_M"] = a_frac
    st.session_state["r0"] = r0

    a = a_frac * M
    if a >= M:
        st.error("Spin must be < M to avoid naked singularity")
        return

    try:
        fd = float(frame_dragging(r0, theta, M, a))
        sup = float(kerr_suppression(a, M))
        erg = float(ergosphere_radius(theta, M, a))
        tau_k = float(tau_kerr(r0, M, a))
        tau_s = float(tau_static(r0))
        ratio = tau_k / (tau_s + EPS)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Frame dragging omega", f"{fd:.5f}")
        c2.metric("Kerr suppression", f"{sup:.4f}")
        c3.metric("Ergosphere r_erg", f"{erg:.4f}")
        c4.metric("tau_Kerr / tau_static", f"{ratio:.4f}")
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

    with col_r:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig_fd = plot_frame_dragging(M, a, r0)
        st.pyplot(fig_fd)
        plt.close(fig_fd)

    st.divider()
    st.markdown("### Exotic Matter Budget")

    import pandas as pd
    a_fracs = [0.0, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.99]
    budget_rows = []
    for af in a_fracs:
        av = af * M
        s = float(kerr_suppression(av, M))
        tk = float(tau_kerr(r0, M, av))
        pct = float(tau_reduction_percent(av, M))
        rp, rm = horizon_radii(M, av)
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
            "Suppression": round(s, 4),
            "tau_Kerr": f"{tk:.5e}",
            "Reduction %": round(pct, 2),
            "Regime": regime,
            "Current": af == round(a_frac, 2),
        })
    df_budget = pd.DataFrame(budget_rows)

    def highlight_cur(row):
        if row["Current"]:
            return [f"background-color: {DIM}"] * len(row)
        return [""] * len(row)

    styled = df_budget.drop(columns=["Current"]).style.apply(
        highlight_cur, axis=1, subset=pd.IndexSlice[:, df_budget.columns[:-1]]
    )
    st.dataframe(df_budget.drop(columns=["Current"]), use_container_width=True)

    show_footer()


# ── Page 3 — Wormhole & NEC ───────────────────────────────────────────────────

def page_wormhole_nec():
    st.header("Wormhole Geometry & NEC Analysis")
    tier_badge("I")

    c_l, c_r = st.columns(2)
    with c_l:
        r0 = st.slider("Throat radius r0", 0.3, 5.0,
                       float(st.session_state["r0"]), 0.05, key="wn_r0")
        shape = st.selectbox("Shape function", ["power", "constant", "power_law", "visser"],
                              key="wn_shape")
    with c_r:
        M = st.slider("Mass M", 0.1, 10.0, float(st.session_state["M"]),
                      0.1, key="wn_M")
        a_frac = st.slider("Spin a/M", 0.0, 0.99,
                           float(st.session_state["a_over_M"]), 0.01, key="wn_a")

    st.session_state["r0"] = r0
    st.session_state["M"] = M
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
        sym = "PASS" if ok else "FAIL"
        st.markdown(
            f'<span style="background:{color};color:#0b1120;border-radius:4px;'
            f'padding:4px 12px;font-weight:bold;">{sym} — {label}</span>',
            unsafe_allow_html=True,
        )

    b1, b2, b3 = st.columns(3)
    with b1:
        badge(cond["throat"], f"b(r0)={cond['b_r0']:.4f}=r0")
    with b2:
        badge(cond["flare_out"], f"b'(r0)={cond['b_prime_r0']:.4f}<1")
    with b3:
        badge(cond["asymptotic"], "b/r->0 asymptotically")

    st.divider()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig_nec = plot_nec(r0, shape, M, a_frac * M)
        st.pyplot(fig_nec)
        plt.close(fig_nec)
    except Exception as e:
        st.warning(f"NEC plot error: {e}")

    st.divider()
    st.markdown("### Flamm Embedding Diagram")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig_flamm = plot_flamm(r0, shape)
        st.pyplot(fig_flamm)
        plt.close(fig_flamm)
    except Exception as e:
        st.warning(f"Embedding plot error: {e}")

    st.divider()
    st.markdown("### Exotic Matter Budget vs Casimir Lab Limits")
    tier_badge("IV")
    try:
        a_val = a_frac * M
        ex = exotic_NEC_with_kerr(r0, M, a_val)
        tau_req = ex["tau_required"]
        orders = math.log10(max(tau_req / CASIMIR_LAB_MAX, 1e-20))
        c1, c2, c3 = st.columns(3)
        c1.metric("tau_required (geom)", f"{tau_req:.4e}")
        c2.metric("Kerr reduction", f"{ex['reduction_pct']:.1f}%")
        c3.metric("Feasibility gap (orders of magnitude)",
                  f"{orders:.1f}" if tau_req > CASIMIR_LAB_MAX else "Within lab range!")
    except Exception as e:
        st.warning(f"Budget computation error: {e}")

    show_footer()


# ── Page 4 — Throat & Echo ────────────────────────────────────────────────────

def page_throat_echo():
    st.header("Dynamic Throat (Eq 6.3) & GW Echo Spectrum (Eq 9b)")
    tier_badge("II")

    st.latex(r"\ddot{\delta a} + 2\eta_s \dot{\delta a} + \frac{\sigma}{a_0^2}\delta a = 0")
    st.latex(
        r"\hat{H}(f) = A_0 \cdot \left(\frac{f}{f_0}\right)^2 \cdot "
        r"e^{-\eta_s f/f_0^2} \cdot \left[1+\left(\frac{f}{f_0}\right)^2\right]^{-1}"
    )

    c_l, c_r = st.columns(2)
    with c_l:
        sigma_throat = st.slider("sigma_throat", 0.01, 2.0,
                                  float(st.session_state["sigma_throat"]),
                                  0.01, key="te_sig")
        a0 = st.slider("a0 (throat radius)", 0.3, 4.0, 1.2, 0.05, key="te_a0")
        eta_s = st.slider("eta_s (damping)", 0.01, 1.0,
                           float(st.session_state["eta_s"]), 0.01, key="te_eta")
    with c_r:
        delta_a0 = st.slider("delta_a0 (initial displacement)", 0.01, 0.5,
                              0.10, 0.01, key="te_da")
        t_max = st.slider("t_max", 10, 100, 50, 5, key="te_tmax")

    st.session_state["sigma_throat"] = sigma_throat
    st.session_state["eta_s"] = eta_s

    try:
        omega0 = natural_frequency(sigma_throat, a0)
        regime, omega_d = damping_regime(sigma_throat, a0, eta_s)

        regime_colors = {"UNDERDAMPED": TEAL, "CRITICAL": GOLD, "OVERDAMPED": RED}
        rclr = regime_colors.get(regime, TEAL)

        rm1, rm2, rm3 = st.columns(3)
        rm1.metric("omega_0 (natural freq)", f"{omega0:.4f}")
        rm2.metric("omega_d (damped freq)", f"{omega_d:.4f}" if omega_d > 0 else "N/A")
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
                fig_t = plot_throat(sigma_throat, a0, eta_s, delta_a0, t_max)
            st.pyplot(fig_t)
            plt.close(fig_t)
        except Exception as e:
            st.warning(f"Throat plot error: {e}")

    with col_r:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig_e = plot_echo(sigma_throat, a0, eta_s)
            st.pyplot(fig_e)
            plt.close(fig_e)
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
        except Exception as e:
            st.warning(f"Regime map error: {e}")

    with col_r2:
        st.markdown("### Echo Properties")
        try:
            report = israel_junction_report(a0, sigma_throat, eta_s)
            si = stability_index(sigma_throat, a0, eta_s)
            dt = report["dt_echo"]
            st.table({
                "Property": ["f0", "Delta_t echo", "n_echoes", "stability_index"],
                "Value": [
                    f"{report['f0']:.5f}",
                    f"{dt:.4f}" if not math.isinf(dt) else "inf",
                    str(report["n_echoes"]),
                    f"{si:.3f}",
                ],
            })
        except Exception as e:
            st.warning(f"Echo table error: {e}")

        st.markdown("**eta_s sensitivity**")
        try:
            f0 = echo_frequency(sigma_throat, a0)
            f_arr = np.linspace(0, max(8 * f0, 0.1), 200)
            fig_eta, ax_eta = _new_fig(figsize=(5, 3))
            for eta_v in [0.05, 0.15, 0.30, 0.50, 0.80]:
                amps = echo_spectrum(f_arr, 1.0, f0, eta_v)
                ax_eta.plot(f_arr, amps, lw=1.2, label=f"eta={eta_v}")
            _style_ax(ax_eta, "Echo Spectrum — eta_s sweep")
            ax_eta.set_xlabel("f")
            ax_eta.set_ylabel("H(f)")
            ax_eta.legend(facecolor=BG2, labelcolor=TEXT, fontsize=7)
            fig_eta.tight_layout()
            st.pyplot(fig_eta)
            plt.close(fig_eta)
        except Exception as e:
            st.warning(f"Eta sweep error: {e}")

    show_footer()


# ── Page 5 — f(R) Gravity ─────────────────────────────────────────────────────

def page_fR():
    st.header("f(R) = R + alphaR^2 Gravity")
    tier_badge("III")

    st.latex(r"f(R) = R + \alpha R^2")
    st.latex(r"\phi = \frac{df}{dR} = 1 + 2\alpha R")
    st.latex(r"V(\phi) = \frac{(\phi-1)^2}{4\alpha}")

    c_l, c_r = st.columns(2)
    with c_l:
        alpha = st.slider("alpha", 0.01, 2.0,
                          float(st.session_state["alpha_fR"]), 0.01, key="fr_alpha")
        phi0 = st.slider("phi0 (initial field value)", 0.8, 1.5,
                          float(st.session_state["phi0"]), 0.01, key="fr_phi0")
    with c_r:
        r0_fr = st.slider("r0", 0.5, 3.0, float(st.session_state["r0"]),
                           0.05, key="fr_r0")
        r_max = st.slider("r_max", 10, 50, 30, 1, key="fr_rmax")

    if alpha <= 0:
        st.error("alpha must be positive for f(R) = R + alpha*R^2")
        return
    if r0_fr <= 0:
        st.error("Throat radius must be positive")
        return

    st.session_state["alpha_fR"] = alpha
    st.session_state["phi0"] = phi0

    if st.button("Solve Scalar Field ODE", type="primary"):
        with st.spinner("Integrating scalar field ODE..."):
            try:
                r_arr, phi_arr, dphi_arr, sol = solve_scalar_field(
                    r0_fr, r_max, alpha, phi0, shape="power"
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
                    {"r0": r0_fr, "r_max": r_max, "alpha": alpha, "phi0": phi0},
                    {"phi_final": float(phi_arr[-1]) if len(phi_arr) > 0 else None,
                     "success": sol.success},
                    {"n_steps": len(r_arr), "method": "RK45"},
                )
            except Exception as e:
                st.error(f"ODE solve failed: {e}")

    sol_data = st.session_state.get("fR_solution")
    if sol_data is not None:
        r_arr = sol_data["r"]
        phi_arr = sol_data["phi"]

        col_l, col_r = st.columns(2)
        with col_l:
            try:
                fig_phi = plot_fR_phi(r_arr, phi_arr, sol_data["r0"], sol_data["alpha"])
                st.pyplot(fig_phi)
                plt.close(fig_phi)
            except Exception as e:
                st.warning(f"phi plot error: {e}")

        with col_r:
            try:
                phi_r0 = float(phi_arr[0]) if len(phi_arr) > 0 else phi0
                fig_pot = plot_potential(sol_data["alpha"], phi0_mark=phi_r0)
                st.pyplot(fig_pot)
                plt.close(fig_pot)
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
            except Exception as e:
                st.warning(f"Effective NEC plot error: {e}")

        with col_r2:
            try:
                phi_r0_val = float(phi_arr[0]) if len(phi_arr) > 0 else phi0
                fig_pot2 = plot_potential(sol_data["alpha"], phi0_mark=phi_r0_val)
                st.pyplot(fig_pot2)
                plt.close(fig_pot2)
            except Exception as e:
                st.warning(f"Potential plot error: {e}")
    else:
        st.info("Press 'Solve Scalar Field ODE' to compute the scalar field solution.")

    st.divider()
    st.markdown("### Shooting Method — Find phi(r0) such that phi(r_max) -> 1")
    if st.button("Find phi0 s.t. phi(inf)->1"):
        with st.spinner("Running shooting method..."):
            try:
                phi0_best, residual, r_s, phi_s, _ = shoot_phi0(
                    r0_fr, r_max, alpha, shape="power"
                )
                converged = residual < 0.05
                color = GREEN if converged else GOLD
                label = "Converged" if converged else "Best approximation"
                st.markdown(
                    f"**phi0_best** = {phi0_best:.5f} &nbsp;&nbsp; "
                    f"**residual** = {residual:.4e} &nbsp;&nbsp; "
                    f'<span style="background:{color};color:#0b1120;border-radius:4px;'
                    f'padding:2px 8px;font-size:0.85em;font-weight:bold;">{label}</span>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Shooting failed: {e}")

    st.divider()
    st.markdown("### alpha Sweep — Min Effective NEC vs alpha")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig_sweep = plot_alpha_sweep(r0_fr, "power", alpha)
        st.pyplot(fig_sweep)
        plt.close(fig_sweep)
    except Exception as e:
        st.warning(f"Alpha sweep error: {e}")

    show_footer()


# ── Page 6 — History ─────────────────────────────────────────────────────────

def page_history():
    st.header("Run History")

    if not _STORAGE_OK:
        st.warning("Storage module unavailable — history disabled.")
        show_footer()
        return

    runs = load_all_runs()

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Filters")
        all_models = sorted({r.model for r in runs}) if runs else []
        model_filter = st.selectbox("Model", ["All"] + all_models)
        date_from = st.text_input("Date from (YYYY-MM-DD)", "")
        date_to   = st.text_input("Date to   (YYYY-MM-DD)", "")
        tag_filter = st.text_input("Tag contains", "")
        nec_only = st.checkbox("NEC violated runs only")

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
        st.info("No runs recorded yet. Use Solve / Run Checks on any page to auto-save.")
        show_footer()
        return

    import pandas as pd
    rows = [r.flat_dict() for r in filtered]
    df = pd.DataFrame(rows)

    # Show summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total runs", len(filtered))
    c2.metric("Models", len({r.model for r in filtered}))
    c3.metric("Latest", filtered[0].timestamp[:16] if filtered else "—")

    st.divider()
    st.dataframe(df, use_container_width=True)

    # ── Compare two runs side-by-side ─────────────────────────────────────────
    st.divider()
    st.markdown("### Diff — compare two runs")
    run_ids = [r.run_id for r in filtered]
    run_labels = [f"{r.run_id} | {r.model} | {r.timestamp[:16]}" for r in filtered]

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

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Export")
    col_l, col_r = st.columns(2)
    with col_l:
        try:
            from storage.json_backend import JSONBackend
            backend = JSONBackend()
            all_ids = [r.run_id for r in filtered]
            st.download_button(
                "Download selected as JSON",
                data=backend.export_json(all_ids),
                file_name="wormhole_runs.json",
                mime="application/json",
            )
        except Exception as e:
            st.warning(f"JSON export unavailable: {e}")
    with col_r:
        try:
            from storage.json_backend import JSONBackend
            backend = JSONBackend()
            csv_data = backend.export_csv([r.run_id for r in filtered])
            st.download_button(
                "Download selected as CSV",
                data=csv_data,
                file_name="wormhole_runs.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.warning(f"CSV export unavailable: {e}")

    # ── Delete ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Delete a run")
    del_id = st.selectbox("Run to delete", ["— select —"] + run_ids, key="hist_del")
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


# ── Main ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="Wormhole Math Checker",
    page_icon="🌀",
)

_init_state()

PAGES = {
    "🏠 Overview": page_overview,
    "🔬 Verification Suite": page_verification,
    "🌀 Kerr Explorer": page_kerr,
    "📐 Wormhole & NEC": page_wormhole_nec,
    "📡 Throat & Echo": page_throat_echo,
    "🔭 f(R) Gravity": page_fR,
    "📚 History": page_history,
}

with st.sidebar:
    st.markdown("## Navigation")
    selected = st.radio("", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption("Wormhole Math Checker v2.0")
    st.caption("Against Chronology Protection")

PAGES[selected]()
