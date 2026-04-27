# Wormhole Math Checker

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

A physics verification and exploration tool for traversable wormhole mathematics. Built to accompany the research at [againstcpc.com](https://againstcpc.com), this app lets you interactively verify the equations behind Kerr-metric wormholes, Morris-Thorne geometry, null energy condition analysis, dynamic throat oscillations, gravitational-wave echo spectra, and f(R) = R + αR² scalar-gravity coupling — all in geometric units (G = c = 1).

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11 recommended. No API keys or secrets required.

## Physics modules

| Module | Content |
|---|---|
| `physics/constants.py` | Geometric units, SI conversions, lab limits |
| `physics/kerr.py` | Frame dragging, horizons, ergosphere, ISCO, suppression factor |
| `physics/morris_thorne.py` | Shape functions, throat conditions, Flamm embedding, stress-energy |
| `physics/energy_conditions.py` | NEC radial/transverse, Kerr-modified exotic budget |
| `physics/throat_dynamics.py` | Damped oscillator (Eq 6.3), echo spectrum (Eq 9b), stability |
| `physics/fR_gravity.py` | Ricci scalar, scalar field ODE, shooting method, alpha sweep |

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo
3. Set **Main file path** to `app.py`, **Python version** to `3.11`
4. Click **Deploy** — no secrets needed

## Links

- Research context: [againstcpc.com](https://againstcpc.com)
