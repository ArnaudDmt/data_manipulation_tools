"""Shared setup for regenerating the paper's data figures.

Four things have to be settled before any of these scripts runs: the default plotly renderer
looks for a Chrome that is not installed and blocks forever, kaleido stamps a MathJax banner into
every export, the colours must be the shared palette rather than each script's own defaults, and
fig.show() has to become a no-op -- these scripts all call it, and a detached rebuild has no
browser to open them in.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("BROWSER", "firefox")
import plotly.io as pio

pio.renderers.default = "firefox"
pio.kaleido.scope.mathjax = None

ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, f"{ROOT}/scripts/paper_results_scripts")
sys.path.insert(0, f"{ROOT}/scripts/plots_scripts")

from generate_metrics_plots import generate_turbo_subset_colors

# Order fixes each hue; keep it stable across every figure. RI-EKF sits fifth, as in the
# published figures, where it reads olive-yellow; moving it earlier turned it green.
ESTIMATORS = ["KO", "KO_ZPC", "KO_WWS", "Tilt", "Hartley", "Control", "Mocap"]
COLORS = generate_turbo_subset_colors(ESTIMATORS)


def also_write_pdf():
    """Mirror every SVG export into a PDF; some scripts only emit SVG."""
    import plotly.graph_objects as go
    original = go.Figure.write_image

    def write_image(self, file, *args, **kwargs):
        original(self, file, *args, **kwargs)
        if str(file).endswith(".svg"):
            original(self, str(file)[:-4] + ".pdf", *args, **kwargs)

    go.Figure.write_image = write_image


def _redirect():
    """Send every export into PAPER_FIG_OUT instead of /tmp.

    The plotting scripts under scripts/paper_results_scripts write to hardcoded /tmp paths. Rather
    than edit five of them, rewrite the destination centrally so a rebuild keeps its own outputs
    together and two rebuilds cannot overwrite each other.
    """
    target = os.environ.get("PAPER_FIG_OUT")
    if not target:
        return
    Path(target).mkdir(parents=True, exist_ok=True)
    import plotly.graph_objects as go
    original = go.Figure.write_image

    def write_image(self, file, *args, **kwargs):
        path = Path(str(file))
        if path.is_absolute():
            path = Path(target) / path.name
        return original(self, str(path), *args, **kwargs)

    go.Figure.write_image = write_image


_redirect()


def headless():
    """Silence fig.show(): the rebuild runs detached and must not wait on a browser."""
    import plotly.graph_objects as go
    go.Figure.show = lambda self, *args, **kwargs: None
