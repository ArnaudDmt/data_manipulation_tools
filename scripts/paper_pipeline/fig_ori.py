"""Contact rest orientation for the initial-error experiment (RightFootRoll).

plotContactRestPoses writes SVG and exits before its own PDF line, and no svg-to-pdf converter is
installed here; also_write_pdf mirrors each SVG into a PDF through kaleido instead.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_harness import ROOT, COLORS, also_write_pdf, headless
also_write_pdf()
headless()
import plotContactPoses

plotContactPoses.plotContactRestPoses(
    path=f"{ROOT}/Projects/HRP5_MultiContact_ContactInitOriError")
