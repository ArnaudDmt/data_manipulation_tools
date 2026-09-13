import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from figure_harness import ROOT, COLORS, also_write_pdf, headless
also_write_pdf()
headless()
import plotPoseAndVelocity

plotPoseAndVelocity.plotPoseVel(
    ["KO", "Hartley", "Mocap"], f"{ROOT}/Projects/KO_TRO2024_RHPS1_1", COLORS)
