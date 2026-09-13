import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from figure_harness import ROOT, COLORS, headless
headless()
# generate_metrics_plots builds these from observersInfos.yaml (name, lineWidth) and hands them
# over; plotMultipleTrajs then merges in its own group and column_names.
from generate_metrics_plots import estimator_plot_args
import plotMultipleTrajs

project = sys.argv[1]
# The curve list comes from the manifest: not every trajectory figure carries KO-ZPC, and drawing
# it where the dataset was not ticked with the second observer instance fails on a missing column.
wanted = sys.argv[2].split(",") if len(sys.argv) > 2 else ["KO", "Hartley", "Control", "Mocap"]
plotMultipleTrajs.plot_multiple_trajs(
    wanted, [project], COLORS,
    {name: dict(estimator_plot_args[name]) for name in wanted},
    path=f"{ROOT}/Projects/")
