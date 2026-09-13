import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from figure_harness import ROOT, COLORS, headless
headless()
import plotContactPoses

plotContactPoses.plotContactPoses(
    estimators_to_plot=["KineticsObserver", "KO_ZPC", "Hartley", "Mocap"], colors=COLORS,
    path=f"{ROOT}/Projects/HRP5_MultiContact_1")
