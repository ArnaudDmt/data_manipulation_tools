"""Estimated gyrometer bias over the long walk, against the two instants where the truth is known.

The true bias is only measurable while the robot does not turn: everywhere else the robot's own
rotation, 11588 deg accumulated over the walk, swamps it through any small frame misalignment.
HRP5P_LongWalk has exactly two such windows, 0-35.7 s and 1941-1946 s, and they say the yaw bias
is 0.0001 and 0.0003 deg/s -- both within the measurement uncertainty, i.e. no bias and no
detectable drift. The logged gyrometer is already compensated.

What the estimators put in that state is therefore an error absorber, not a bias estimate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_harness import ROOT, COLORS, headless
headless()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

AXES = ("x", "y", "z")
# Measured above; value and uncertainty in deg/s, both windows.
TRUTH = {"x": [(17.8, 0.00040, 0.00015), (1943.5, 0.00737, 0.00023)],
         "y": [(17.8, 0.00077, 0.00026), (1943.5, 0.01659, 0.00051)],
         "z": [(17.8, 0.00012, 0.00104), (1943.5, 0.00031, 0.00073)]}
KO = "Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_gyroBias_Accelerometer_"

# The three files do NOT share a clock: logReplay is the full log at 500 Hz over 2948 s,
# HartleyOutputCSV the same log at 250 Hz, and finalDataCSV only the evaluated window, 0-1946 s.
# Truncating to the shortest mixed a 500 Hz series with a 250 Hz one and ran 1000 s past the end
# of the walk. Both series are interpolated onto one grid, clipped to the evaluated window.
out = Path(ROOT) / "Projects/HRP5P_LongWalk/output_data"
ko = pd.read_csv(out / "logReplay.csv", sep=";",
                 usecols=lambda c: c in ["t"] + [f"{KO}{a}" for a in AXES])
ri = pd.read_csv(out / "HartleyOutputCSV.csv", sep=";",
                 usecols=lambda c: c in ["t"] + [f"IMU_GyroBias_{a}" for a in AXES])
evaluated = pd.read_csv(out / "finalDataCSV.csv", sep=";", usecols=["t"])["t"].to_numpy()
t = evaluated[::2]                            # ~125 Hz, far above anything a bias does
ko_t, ri_t = ko["t"].to_numpy(), ri["t"].to_numpy()
ko = {a: np.interp(t, ko_t, ko[f"{KO}{a}"].to_numpy()) for a in AXES}
ri = {a: np.interp(t, ri_t, ri[f"IMU_GyroBias_{a}"].to_numpy()) for a in AXES}
step = max(1, len(t) // 4000)                 # 4000 points is plenty at this time scale
print(f"fenetre evaluee : {t[0]:.1f} a {t[-1]:.1f} s, {len(t)} points")

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
for row, a in enumerate(AXES, start=1):
    for name, label, values in (("KO", "KO", np.degrees(ko[a])),
                                ("Hartley", "RI-EKF", np.degrees(ri[a]))):
        colour = COLORS[name]
        fig.add_trace(go.Scatter(x=t[::step], y=values[::step], mode="lines",
                                 line=dict(color=f"rgba({colour[0]}, {colour[1]}, {colour[2]}, 1)", width=1),
                                 name=label, legendgroup=label, showlegend=(row == 1)), row=row, col=1)
    if a != "z":
        # On x and y the second window is NOT a clean bias measurement: over those five seconds the
        # robot is still settling in roll and pitch, so the mocap's own rotation is not zero and the
        # residual is motion, not bias. Only the yaw axis is measured cleanly.
        continue
    when, value, sigma = zip(*TRUTH[a])
    fig.add_trace(go.Scatter(x=when, y=value, mode="markers",
                             marker=dict(size=9, color="black", symbol="diamond"),
                             error_y=dict(type="data", array=sigma, visible=True, color="black"),
                             name="Measured truth", legendgroup="truth", showlegend=(row == 1)),
                  row=row, col=1)
    fig.update_yaxes(title_text=f"bias {a} (deg/s)", row=row, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_layout(template="plotly_white", height=640, width=780,
                  font=dict(family="Times New Roman", size=18, color="black"),
                  margin=dict(l=90, r=20, t=60, b=60),
                  legend=dict(yanchor="bottom", y=1.02, xanchor="left", x=0.01, orientation="h",
                              bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Times New Roman", size=18, color="black")))
destination = Path(ROOT) / "results/paper-rebuild/figures/gyroBias_longwalk.pdf"
destination.parent.mkdir(parents=True, exist_ok=True)
fig.write_image(str(destination))
print(f"ecrit {destination}")
