"""LongWalk top view, first 500 s only.

The full 1946 s superimposes twenty-five passes over the same 6x5 m patch and reads as a tangle;
the opening window keeps the trajectory followable while still covering about 90 m of walking.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from figure_harness import ROOT, COLORS, headless
headless()
import numpy as np, pandas as pd
import plotly.graph_objects as go
sys.path.insert(0, f"{ROOT}/scripts/paper_results_scripts")
import paper_colors
from generate_metrics_plots import estimator_plot_args

SERIES = [("KO", "Kinetics Observer"), ("Hartley", "RI-EKF"), ("Mocap", "Ground truth")]
UNTIL = 500.0

columns = ["t"] + [f"{e}_position_{a}" for e, _ in SERIES for a in "xy"]
frame = pd.read_csv(f"{ROOT}/Projects/HRP5P_LongWalk/output_data/finalDataCSV.csv",
                    sep=";", usecols=lambda c: c in columns)
window = frame["t"].to_numpy() <= UNTIL

fig = go.Figure()
for name, label in reversed(SERIES):          # drawn back to front, ground truth underneath
    r, g, b = paper_colors.resolve(COLORS, name)
    fig.add_trace(go.Scatter(
        x=frame[f"{name}_position_x"].to_numpy()[window],
        y=frame[f"{name}_position_y"].to_numpy()[window],
        # plotMultipleTrajs draws its main traces at lineWidth + 2, lineWidth coming from
        # observersInfos.yaml. Trimmed by half a point here: three passes overlap on this figure
        # and at the full stroke the RI-EKF vanishes under the Kinetics Observer.
        mode="lines",
        line=dict(color=f"rgb({r}, {g}, {b})",
                  width=estimator_plot_args[name]["lineWidth"] + 1.5),
        name=label))

fig.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    # Same canvas and type size as the other trajectory figures, which take plotly's 700x500
    # default and export at 525.12 x 375.12 pts.
    font=dict(family="Times New Roman", size=22, color="black"),
    legend=dict(yanchor="bottom", y=1.01, xanchor="left", x=0.0, orientation="h",
                bgcolor="rgba(0,0,0,0)", traceorder="reversed"),
    margin=dict(l=0, r=0, b=0, t=34), width=700, height=500,
    xaxis=dict(title="X Position (m)", gridcolor="lightgrey", gridwidth=1.5,
               zerolinecolor="lightgrey", scaleanchor="y", scaleratio=1),
    yaxis=dict(title="Y Position (m)", gridcolor="lightgrey", gridwidth=1.5,
               zerolinecolor="lightgrey"))
fig.write_image("/tmp/traj_hrp5_long.pdf")
fig.show()
print(f"{window.sum()} echantillons jusqu'a {UNTIL:.0f} s -> /tmp/traj_hrp5_long.pdf")
