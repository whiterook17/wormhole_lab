"""Guide page — plain-English introduction to the Wormhole Math Checker."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


# ── Embedding diagram ─────────────────────────────────────────────────────────

def _wormhole_diagram():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#111e36")

    r = np.linspace(1.0, 5.0, 200)
    z = 2.0 * np.sqrt(np.maximum(r - 1.0, 0.0))

    # Universe A (right)
    ax.plot(r, z,   color="#00c8c8", lw=2)
    ax.plot(r, -z,  color="#00c8c8", lw=2)
    # Universe B (left, mirrored)
    ax.plot(-r, z,  color="#a070e0", lw=2)
    ax.plot(-r, -z, color="#a070e0", lw=2)

    # Throat arc
    th = np.linspace(-np.pi / 2, np.pi / 2, 80)
    ax.plot(np.cos(th) * 0.85, np.sin(th) * 0.9, color="#e8a020", lw=3)

    # Labels
    ax.text(3.5,  2.5,  "Universe A", color="#00c8c8", fontsize=9, ha="center")
    ax.text(-3.5, 2.5,  "Universe B", color="#a070e0", fontsize=9, ha="center")
    ax.text(0.0,  -1.2, "Throat  r₀",  color="#e8a020", fontsize=9, ha="center")
    ax.annotate("", xy=(0.9, 0.0), xytext=(2.5, 0.0),
                arrowprops=dict(arrowstyle="->", color="#c8daea", lw=1.0))
    ax.annotate("", xy=(-0.9, 0.0), xytext=(-2.5, 0.0),
                arrowprops=dict(arrowstyle="->", color="#c8daea", lw=1.0))
    ax.text(0.0, 0.15, "passage", color="#c8daea", fontsize=8, ha="center")

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-3.2, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Flamm Embedding — Wormhole Cross-Section",
                 color="#00c8c8", fontsize=10)
    fig.tight_layout()
    return fig


# ── Main render function ──────────────────────────────────────────────────────

def render_guide(navigate_fn=None):
    """Render the 8-section guide page.

    navigate_fn: callable(page_name) that switches the app to another page,
                 or None to skip navigation buttons.
    """
    st.title("\U0001f4d6 Guide — How to Use This Tool")
    st.caption("Start here if this is your first visit.")

    # ── 1. What this tool is ─────────────────────────────────────────────────
    st.markdown("## What is this tool?")
    st.markdown(
        """
**Wormhole Math Checker** is a physics verification tool that checks the mathematics
behind hypothetical traversable wormholes.  It does **not** claim wormholes exist or
are buildable — it verifies that the equations used in theoretical papers are
internally self-consistent.

Think of it as a *calculator with receipts*: every number it shows you comes from a
named equation, and every equation is cross-checked against a known analytic limit or
numerical solution.
        """
    )
    if navigate_fn:
        if st.button("▶ Jump to Verification Suite", key="guide_cta_verify"):
            navigate_fn("\U0001f52c Verification Suite")

    st.divider()

    # ── 2. Wormhole explainer ────────────────────────────────────────────────
    st.markdown("## What is a traversable wormhole?")
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown(
            """
A **traversable wormhole** is a hypothetical tunnel through spacetime connecting two
distant regions (or two different universes).  Unlike a black hole it has two mouths and
a **throat** — the narrowest point — that a traveller could pass through without
being crushed.

The mathematics come from General Relativity.  Einstein’s field equations allow such
solutions **if** a special kind of matter is present at the throat.  This matter must have
**negative energy density** — it violates what physicists call the
**Null Energy Condition (NEC)**.  No known macroscopic substance does this naturally,
though quantum effects (like the Casimir effect) produce tiny NEC violations in a lab.

This tool implements the **Morris–Thorne** (1988) wormhole model — the simplest
self-consistent traversable wormhole — extended with Kerr (rotating) geometry and
f(R) modified-gravity corrections.
            """
        )
    with col_r:
        try:
            fig = _wormhole_diagram()
            st.pyplot(fig)
            plt.close(fig)
        except Exception:
            st.caption("[Embedding diagram unavailable]")

    st.divider()

    # ── 3. Four parameter cards ──────────────────────────────────────────────
    st.markdown("## The four key parameters")
    st.caption(
        "These controls appear on every physics page. Here is what they mean in plain English."
    )

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### M — Central Mass")
            st.markdown(
                """
The mass of the central gravitating body, in **solar masses** (geometric units where
G = c = 1).

- M = 1 → one solar mass ≈ 1.989 × 10³⁰ kg
- Larger M → bigger gravitational radii → more exotic matter needed overall
- The Schwarzschild radius is r_s = 2M (in geometric units)
                """
            )
        with st.container(border=True):
            st.markdown("#### r₀ — Throat Radius")
            st.markdown(
                """
The radius of the wormhole’s narrowest point, in **gravitational radii** (r_g).

- r₀ must be outside the horizon to avoid being swallowed by the black hole
- r₀ = 1.2 means the throat is 1.2 × 1 477 m ≈ 1.77 km for a solar-mass body
- Smaller r₀ → stronger curvature → more exotic matter required
                """
            )
    with c2:
        with st.container(border=True):
            st.markdown("#### a/M — Spin Parameter")
            st.markdown(
                """
How fast the central mass rotates, as a fraction of its maximum (extremal) spin.

- a/M = 0 → non-rotating (Schwarzschild)
- a/M = 1 → maximally rotating (extremal Kerr) — theoretical limit
- Higher spin **reduces** the exotic matter required by a factor √(1 − a²/M²)
- This is the **Kerr suppression** effect: frame-dragging partly substitutes for exotic matter
                """
            )
        with st.container(border=True):
            st.markdown("#### σ_throat — Stiffness")
            st.markdown(
                """
How elastically the wormhole throat responds to perturbations.

- High σ → throat snaps back quickly (stiff spring)
- Low σ → throat responds sluggishly (weak spring)
- Together with η_s (damping) it sets the oscillation regime:
  **underdamped** (rings), **critically damped** (returns fastest), or **overdamped** (creeps back)
- The governing equation is identical in form to a spring–mass–damper:
  δä + 2η_s δą + (σ/a₀²)δa = 0
                """
            )

    st.divider()

    # ── 4. Geometric units table ─────────────────────────────────────────────
    st.markdown("## Geometric units — what the numbers mean")
    st.markdown(
        """
All physics is computed in **geometric units** (G = c = 1).  These simplify the equations
but can be disorienting.  The table below shows how to interpret typical values in SI
for a **1 solar-mass** central body.
        """
    )

    import pandas as pd
    tbl = {
        "Quantity": [
            "Length  (1 r_g)",
            "Time    (1 t_g)",
            "Mass    (1 M)",
            "Energy  (1 M c²)",
            "Tension (1 τ_unit)",
            "Frequency (1 f_g)",
        ],
        "Geometric value": [
            "1 r_g", "1 t_g", "1 M", "1", "1", "1 f_g",
        ],
        "SI equivalent": [
            "1 477 m ≈ 1.48 km",
            "4.93 × 10⁻⁶ s ≈ 4.93 µs",
            "1.989 × 10³⁰ kg",
            "1.8 × 10⁴⁷ J",
            "1.21 × 10⁴⁴ J/m²",
            "≈ 203 000 Hz",
        ],
        "Intuition": [
            "Half a Schwarzschild radius",
            "Light crossing time of r_g",
            "One solar mass",
            "Solar mass–energy",
            "Casimir lab max ≈ 10⁻³ J/m²",
            "Radio band",
        ],
    }
    st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
    st.info(
        "\U0001f4d0 **Units toggle:** Use the **\U0001f4d0 Display units** control in the "
        "sidebar to switch between Geometric only, Metric (SI), or Both at once."
    )

    st.divider()

    # ── 5. Verification checks ───────────────────────────────────────────────
    st.markdown("## The verification checks — what are they testing?")
    st.markdown(
        """
The **Verification Suite** runs every check automatically.  Each check is a mathematical
assertion: it computes something two different ways (or against a known analytic limit)
and confirms the answers match to high precision.  A failing check means a bug in the
software, not a violation of physics.
        """
    )

    with st.expander("Kerr checks — rotating black hole geometry"):
        st.markdown(
            """
- Horizon radii r± = M ± √(M²−a²) match the Schwarzschild limit at a = 0
- Ergosphere r_erg = M + √(M²−a²cos²θ) → r_s at θ = π/2
- Frame-dragging ω(r) → 0 as r → ∞
- Suppression factor √(1−a²/M²) reduces exotic matter at high spin
- τ_Kerr/τ_static = kerr_suppression at four reference spin values
            """
        )

    with st.expander("Morris–Thorne checks — wormhole shape conditions"):
        st.markdown(
            """
- **b(r₀) = r₀**: the shape function equals the throat radius at the throat
- **b′(r₀) < 1**: the flare-out condition — the wormhole must open outward
- **b(r)/r → 0 as r→∞**: asymptotic flatness — spacetime is normal far away
            """
        )

    with st.expander("NEC checks — exotic matter requirement"):
        st.markdown(
            """
The Null Energy Condition (NEC) states ρ + p_r ≥ 0 for normal matter.
A traversable wormhole *requires* NEC violation at the throat.  It is not an error —
it is the mathematical signature of exotic matter keeping the wormhole open.
These checks confirm the NEC **is** violated, as required.
            """
        )

    with st.expander("Throat dynamics checks — oscillation physics"):
        st.markdown(
            """
- The analytic solution δa(t) for each damping regime matches RK45 integration to < 10⁻⁶
- The regime classifier assigns the correct label based on η_s vs ω₀
- Displacement decays to zero for overdamped and critically damped cases
            """
        )

    with st.expander("Echo spectrum checks — gravitational wave signal"):
        st.markdown(
            """
- H(0) = 0: no DC component in the echo signal
- Peak at f₀: the spectrum peaks at the natural frequency
- High-frequency falloff: the envelope decays as a power law at large f
            """
        )

    with st.expander("f(R) algebra checks — modified gravity"):
        st.markdown(
            """
- φ(R) = 1 + 2αR and R(φ) = (φ−1)/(2α) are mutual inverses
- V(φ = 1) is a minimum — GR is a fixed point of the theory
- f′(0) = 1 — the theory reduces to GR when R = 0
- ODE solution remains finite and → 1 at large r (asymptotic flatness)
            """
        )

    with st.expander("Model protocol checks — software integrity"):
        st.markdown(
            """
These confirm that GR and f(R) model objects implement the `GravityModel` protocol
correctly: `stress_energy()`, `nec_at_throat()`, and `is_traversable()` all return
valid, finite values.
            """
        )

    st.divider()

    # ── 6. Suggested journey ─────────────────────────────────────────────────
    st.markdown("## Suggested journey")

    journey = [
        ("\U0001f52c Verification Suite", "verify",
         "Run all checks first.  If all pass the physics is self-consistent and you can "
         "trust every number on the other pages."),
        ("\U0001f300 Kerr Explorer", "kerr",
         "Set M = 1 and sweep a/M from 0 to 0.99.  Watch the suppression factor fall and "
         "the exotic matter requirement shrink."),
        ("\U0001f4d0 Wormhole & NEC", "nec",
         "Try different shape functions.  Notice that the NEC is always violated at the "
         "throat — that is the exotic matter signature, not an error."),
        ("\U0001f4e1 Throat & Echo", "echo",
         "Increase σ_throat until you cross from underdamped into overdamped.  Watch the "
         "echo spectrum peak sharpen."),
        ("\U0001f52d f(R) Gravity", "fr",
         "Solve the scalar field ODE with α = 0.15.  Notice how the effective NEC "
         "changes compared to pure GR."),
    ]

    page_map = {
        "verify": "\U0001f52c Verification Suite",
        "kerr":   "\U0001f300 Kerr Explorer",
        "nec":    "\U0001f4d0 Wormhole & NEC",
        "echo":   "\U0001f4e1 Throat & Echo",
        "fr":     "\U0001f52d f(R) Gravity",
    }

    for idx, (page_label, key_suffix, description) in enumerate(journey, start=1):
        cn, cd = st.columns([1, 9])
        cn.markdown(f"### {idx}")
        with cd:
            st.markdown(f"**{page_label}**")
            st.markdown(description)
            if navigate_fn:
                if st.button("Go →", key=f"guide_nav_{key_suffix}"):
                    navigate_fn(page_map[key_suffix])
        if idx < len(journey):
            st.divider()

    st.divider()

    # ── 7. What this tool is NOT ─────────────────────────────────────────────
    st.markdown("## What this tool is NOT")

    col1, col2 = st.columns(2)
    with col1:
        st.error(
            "**NOT a claim that wormholes exist**\n\n"
            "The tool verifies equations.  Whether the exotic matter they require can be "
            "produced in nature is an open question.  Current best answer: no, not at "
            "macroscopic scales."
        )
        st.error(
            "**NOT a design tool**\n\n"
            "You cannot use these calculations to build a real wormhole.  The exotic matter "
            "required exceeds what any known or proposed technology could produce by many "
            "orders of magnitude."
        )
    with col2:
        st.warning(
            "**NOT a full quantum gravity treatment**\n\n"
            "The physics here is classical GR plus simple extensions.  Quantum gravity "
            "(loop quantum gravity, string theory) may qualitatively change results at "
            "Planck scales."
        )
        st.warning(
            "**NOT peer-reviewed research**\n\n"
            "This is an educational tool.  The equations implement published results "
            "(Morris & Thorne 1988, Visser 1995, Starobinsky 1980) but the software "
            "itself has not been peer-reviewed."
        )

    st.divider()

    # ── 8. Glossary ──────────────────────────────────────────────────────────
    st.markdown("## Glossary")

    terms = [
        ("Exotic matter",
         "Matter with negative energy density, required at the wormhole throat.  "
         "Quantum systems like the Casimir effect produce tiny amounts (~10⁻³ J/m²), "
         "far below what a macroscopic wormhole would need."),
        ("Null Energy Condition (NEC)",
         "ρ + p ≥ 0 for all null vectors.  Normal matter satisfies this.  "
         "Traversable wormholes require NEC violation (ρ + p_r < 0) at the throat."),
        ("Kerr metric",
         "The exact GR solution for a rotating, uncharged black hole, characterised by "
         "mass M and spin a.  Introduces frame-dragging of nearby spacetime."),
        ("Frame dragging",
         "The rotating spacetime drags nearby objects along with it.  "
         "The angular velocity is ω(r,θ) = 2Mar / (r² + a²cos²θ)²."),
        ("Ergosphere",
         "Region outside the horizon where corotation with the black hole is unavoidable.  "
         "Radius r_erg = M + √(M²−a²cos²θ) in Boyer–Lindquist coordinates."),
        ("Morris–Thorne wormhole",
         "The 1988 Thorne–Morris model of a static, spherically symmetric traversable "
         "wormhole.  Defined by a shape function b(r) and redshift function Φ(r)."),
        ("Shape function b(r)",
         "Encodes the wormhole geometry.  Must satisfy b(r₀) = r₀ (throat), "
         "b′(r₀) < 1 (flare-out), and b/r → 0 (asymptotic flatness)."),
        ("Flamm paraboloid",
         "The 2D embedding diagram of the wormhole’s equatorial slice.  "
         "Height z(r) = 2√(r − r₀) for the power shape function b(r) = r₀²/r."),
        ("f(R) gravity",
         "Modification of GR replacing the Ricci scalar R in the action with f(R).  "
         "Starobinsky’s choice f(R) = R + αR² is used here."),
        ("Scalar field φ",
         "In f(R) gravity, the extra degree of freedom df/dR = φ = 1 + 2αR acts as "
         "a scalar field satisfying its own ODE in the Morris–Thorne background."),
        ("Shooting method",
         "A numerical technique for solving boundary value problems: guess an initial "
         "condition, integrate the ODE, and iterate until the far-field boundary "
         "condition is satisfied."),
        ("GW echoes",
         "Repeated pulses of gravitational radiation bouncing between the wormhole mouth "
         "and an effective potential barrier.  Their spacing Δt ≈ 1/f₀ carries "
         "information about the throat geometry."),
        ("Casimir effect",
         "A quantum vacuum effect producing an attractive force between parallel "
         "conducting plates.  The resulting energy density is negative (NEC violating) "
         "but ~10⁻³ J/m², many orders of magnitude below what a macroscopic wormhole "
         "requires."),
        ("Geometric units",
         "Unit system where G = c = 1.  Lengths, times, and masses all become "
         "proportional.  1 r_g (gravitational radius) = GM/c² ≈ 1 477 m for M = M_sun."),
    ]

    for term, definition in terms:
        with st.expander(term):
            st.markdown(definition)

    st.divider()
    st.caption(
        "Wormhole Math Checker v2.0 — educational physics tool. "
        "Equations from Morris & Thorne (1988), Visser (1995), Starobinsky (1980)."
    )
