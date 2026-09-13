import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from figure_harness import ROOT, COLORS, headless
headless()
import plotExternalForceAndBias

plotExternalForceAndBias.plotExtWrench(
    colors=COLORS, path=f"{ROOT}/Projects/HRP5_MultiContact_1_WO_LeftHand")
