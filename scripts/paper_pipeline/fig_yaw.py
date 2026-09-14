import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from figure_harness import ROOT, COLORS, headless
headless()
import plotContactPoses

plotContactPoses.plotContactPoses(
    # Drawn back to front: the KO is plotted last so it stays visible where the curves overlap.
    # The legend is put back in reading order by traceorder="reversed" in plotContactPoses.
    estimators_to_plot=["Mocap", "Hartley", "KO_ZPC", "KineticsObserver"], colors=COLORS,
    path=f"{ROOT}/Projects/HRP5_MultiContact_1")
