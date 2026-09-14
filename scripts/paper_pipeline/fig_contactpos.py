"""Right-foot contact POSITION in the world, estimated against the mocap.

The published figure (rightFoot_yaw) shows the contact's yaw; this is its translation counterpart,
in the same format: same palette, same legend, same shaded bands for the left foot and left hand
contacts. The foot pose is obtained by forward kinematics from each estimator's floating base, so
it shows how each one places a contact it believes fixed.
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
from scipy.spatial.transform import Rotation as R

PROJECT = sys.argv[1] if len(sys.argv) > 1 else "HRP5_MultiContact_1"
CONTACT = "RightFootCenter"   # the contact state is keyed by SURFACE, not by force sensor
KINE = "RightFootCenter"
DEBUG = "Observers_MainObserverPipeline_MCKineticsObserver_debug_"
# Drawn back to front, like the yaw figure: the KO last so it stays visible where curves overlap.
SERIES = [("Mocap", "Mocap", "Ground truth"), ("Hartley", "Hartley", "RI-EKF"),
          ("KO_ZPC", "KO_ZPC", "KO-ZPC"), ("KO", "KO", "KO")]
BANDS = {"LeftFootCenter": ("Left foot", "rgba(150,190,240,0.30)"),
         "LeftHandCloseContact": ("Left hand", "rgba(240,150,160,0.30)")}
AXES = ("x", "y", "z")
# Same window as the published yaw figure (its index_range is [0, 2830] at 200 Hz).
WINDOW = 14.15

out = Path(ROOT) / "Projects" / PROJECT / "output_data"
enc = pd.read_csv(out / "logReplay.csv", sep=";", low_memory=False)
obs = pd.read_csv(out / "finalDataCSV.csv", sep=";", low_memory=False)
n = min(len(enc), len(obs))
enc, obs = enc.iloc[:n].reset_index(drop=True), obs.iloc[:n].reset_index(drop=True)
keep = obs["t"].to_numpy() <= WINDOW
enc, obs = enc[keep].reset_index(drop=True), obs[keep].reset_index(drop=True)
t = obs["t"].to_numpy()

fb = enc[[f"{DEBUG}contactKine_{KINE}_inputUserContactKine_position_{a}" for a in AXES]].to_numpy()
is_set = (enc[f"{DEBUG}contactState_isSet_{CONTACT}"] == "Set").to_numpy()

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04)
for key, prefix, label in SERIES:
    columns = [f"{prefix}_position_{a}" for a in AXES]
    quat = [f"{prefix}_orientation_{a}" for a in "xyzw"]
    if not set(columns + quat) <= set(obs.columns):
        print(f"  {label}: colonnes absentes, courbe ignoree")
        continue
    base = obs[columns].to_numpy()
    rotation = R.from_quat(obs[quat].to_numpy())
    world = base + rotation.apply(fb)
    world[~is_set] = np.nan
    world = world - np.nanmean(world[is_set][:50], axis=0)   # origine commune
    colour = COLORS[key]
    rgba = f"rgba({colour[0]}, {colour[1]}, {colour[2]}, 1)"
    for row, a in enumerate(AXES, start=1):
        fig.add_trace(go.Scatter(x=t, y=1000 * world[:, row - 1], mode="lines",
                                 line=dict(color=rgba, width=1 if key != "KO" else 1),
                                 name=label, legendgroup=label, showlegend=(row == 1)), row=row, col=1)

for contact, (label, fill) in BANDS.items():
    mask = (enc[f"{DEBUG}contactState_isSet_{contact}"] == "Set").to_numpy()
    edges = np.flatnonzero(np.diff(mask.astype(int)))
    starts = list(edges[::2] + 1) if not mask[0] else [0] + list(edges[1::2] + 1)
    ends = list(edges[1::2]) if not mask[0] else list(edges[::2])
    for start, end in zip(starts, ends):
        for row in (1, 2, 3):
            fig.add_vrect(x0=t[start], x1=t[end], fillcolor=fill, line_width=0,
                          layer="below", row=row, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(size=10, color=fill), name=label, legend="legend2"))

for row, a in enumerate(AXES, start=1):
    fig.update_yaxes(title_text=f"{a} (mm)", row=row, col=1)
fig.update_xaxes(title_text="Time (s)", row=3, col=1)
fig.update_layout(template="plotly_white", height=620, width=760,
                  font=dict(family="Times New Roman", size=18, color="black"),
                  margin=dict(l=70, r=20, t=60, b=60),
                  legend=dict(yanchor="bottom", y=1.02, xanchor="left", x=0.01,
                              orientation="h", bgcolor="rgba(0,0,0,0)", traceorder="reversed",
                              font=dict(family="Times New Roman", size=18, color="black")),
                  legend2=dict(yanchor="top", y=0.98, xanchor="left", x=0.02,
                               bgcolor="rgba(0,0,0,0)",
                               font=dict(family="Times New Roman", size=14, color="black")))
# Written under the repository, not the scratchpad: Firefox cannot read /tmp/claude-*, so a
# figure left there opens as "file not found".
destination = Path(ROOT) / "results/paper-rebuild/figures/rightFoot_position.pdf"
destination.parent.mkdir(parents=True, exist_ok=True)
fig.write_image(str(destination))
print(f"ecrit {destination}")
