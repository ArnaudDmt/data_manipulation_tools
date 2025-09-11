import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  # For color palette generation
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
from pathlib import Path
import matplotlib.image as mpimg
import plotly.io as pio
import os
from tqdm import tqdm   
from typing import Dict, List                # just for type hints

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

import plotly.io as pio
pio.kaleido.scope.mathjax = None

# Tell webbrowser to use wslview
os.environ["BROWSER"] = "wslview"
pio.renderers.default = "browser"


default_path = '.../Projects/'
 
default_exps = [
    'KO_TRO2024_RHPS1_1', 
    'KO_TRO2024_RHPS1_2', 
    'KO_TRO2024_RHPS1_3', 
    'KO_TRO2024_RHPS1_4', 
    'KO_TRO2024_RHPS1_5'
]

default_estimators = [
    'Control',
    'Vanyte',
    'Hartley',
    'KO',
    # 'KineticsObserver',
    # 'KO_APC',
    # 'KO_ASC',
    # 'KO_ZPC',
    # 'KOWithoutWrenchSensors',
    'Tilt',
    # 'Mocap',
]


# # Define columns for each estimator
estimator_plot_args_default = {
    
    'Hartley': {'group': 1, 'lineWidth': 1, 'column_names':  ['RI-EKF_Position_x', 'RI-EKF_Position_y']},
    'Control': {'group': 1, 'lineWidth': 2, 'column_names': ['Controller_tx', 'Controller_ty']},
    'Vanyte': {'group': 1, 'lineWidth': 1, 'column_names': ['Vanyte_position_x', 'Vanyte_position_y']},
    'Mocap': {'group': 0, 'lineWidth': 1, 'column_names': ['Mocap_position_x', 'Mocap_position_y']},
    'KO': {'group': 1, 'lineWidth': 1, 'column_names': ['KO_position_x', 'KO_position_y']},
    
    'Tilt': {'group': 1, 'lineWidth': 1, 'column_names': ['Tilt_position_x', 'Tilt_position_y']},
    #'KO_APC': {'group': 1, 'lineWidth': 2, 'column_names': ['KO_APC_posW_tx', 'KO_APC_posW_ty']},
    #'KO_ASC': {'group': 2, 'lineWidth': 2, 'column_names': ['KO_ASC_posW_tx', 'KO_ASC_posW_ty']},
    # 'KO-ZPC': {'group': 1, 'lineWidth': 2, 'column_names': ['KO_ZPC_posW_tx', 'KO_ZPC_posW_ty']},
    #'KOWithoutWrenchSensors': {'group': 1, 'lineWidth': 2, 'column_names': ['KOWithoutWrenchSensors_posW_tx', 'KOWithoutWrenchSensors_posW_ty']},
    
    
}
def plot_multiple_trajs(estimators, exps, colors, estimator_plot_args, path = default_path,  main_expe = 0):    
    estimators = list(set(estimators).intersection(estimator_plot_args.keys()).intersection(estimator_plot_args_default.keys())) 
        

    for estimatorName in estimators:
        estimator_plot_args[estimatorName].update(estimator_plot_args_default[estimatorName])
        
    order = list(estimator_plot_args_default.keys())
    estimators = sorted(estimators, key=order.index)
    
    xys = dict.fromkeys(estimators)

    # all_columns = []
    all_groups_keys = set()

    for estimator in estimators:
        all_groups_keys.add(estimator_plot_args[estimator]['group'])
        xys[estimator] = dict.fromkeys(range(len(exps)))
        for k in range(len(exps)):
            xys[estimator][k] = {0: [], 1:[]}
        # for col in estimator_plot_args[estimator]['column_names']:
        #     all_columns.append(col)

    all_groups = dict.fromkeys(all_groups_keys)
    
    for group in all_groups.keys():
        all_groups[group] = {'estimators': [], 'plot_lims': {'xmin': {}, 'xmax': {}, 'ymin': {}, 'ymax': {}}}
    for estimator in estimators:
        all_groups[estimator_plot_args[estimator]['group']]["estimators"].append(estimator)

    for e, exp in enumerate(exps):
        file = f'{path}{exp}/output_data/finalDataCSV.csv'
        df = pd.read_csv(file, sep=';')
        for estimator in estimators:
            xys[estimator][e][0] = df[estimator + '_position_x'].to_numpy()  # 1-D
            xys[estimator][e][1] = df[estimator + '_position_y'].to_numpy()  # 1-D

    
    for group in list(filter(lambda x: x != 0, all_groups.keys())):
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        combined_estimators = all_groups[0]["estimators"] + all_groups[group]["estimators"]
        
        for estimator in combined_estimators:
            for e in range(len(exps)):
                xmins.append(np.min(xys[estimator][e][0]))
                xmaxs.append(np.max(xys[estimator][e][0]))
                ymins.append(np.min(xys[estimator][e][1]))
                ymaxs.append(np.max(xys[estimator][e][1]))

        all_groups[group]['plot_lims']['xmin'] = np.min(xmins)
        all_groups[group]['plot_lims']['xmax'] = np.max(xmaxs)
        all_groups[group]['plot_lims']['ymin'] = np.min(ymins)
        all_groups[group]['plot_lims']['ymax'] = np.max(ymaxs)
                
    # Create a Plotly figure
    
    for group in list(filter(lambda x: x != 0, all_groups.keys())):
        fig = go.Figure()
        combined_estimators = all_groups[0]["estimators"] + all_groups[group]["estimators"]
        for estimator in combined_estimators:
            estimatorName = estimator_plot_args[estimator]["name"]
            color = colors[estimator]

            # Process each CSV for the current estimator
            for e in xys[estimator].keys():
                if(e == main_expe):
                    if(estimator == 'Mocap'):
                        transparent_color = f'rgba({color[0]}, {color[1]}, {color[2]}, 1)'
                        # Use transparent_color in the line color
                        fig.add_trace(go.Scatter(
                            x=xys[estimator][e][0], y=xys[estimator][e][1],
                            mode='lines', line=dict(color=transparent_color, width=estimator_plot_args[estimator]['lineWidth'] + 2),
                            name=f'{estimatorName}', showlegend=True))
                    else:
                        transparent_color = f'rgba({color[0]}, {color[1]}, {color[2]}, 1)'
                        fig.add_trace(go.Scatter(
                            x=xys[estimator][e][0], y=xys[estimator][e][1],
                            mode='lines', line=dict(color=transparent_color, width=estimator_plot_args[estimator]['lineWidth'] + 2),
                            name=f'{estimatorName}', showlegend=True))
                else:
                    if(estimator == 'Mocap'):
                        transparent_color = f'rgba({color[0]}, {color[1]}, {color[2]}, 0.6)'
                        fig.add_trace(go.Scatter(
                            x=xys[estimator][e][0], y=xys[estimator][e][1],
                            mode='lines', line=dict(color=transparent_color, width=1.5, dash='5px,2px'),
                            name=f'{estimatorName} - CSV {e+1} X', showlegend=False))
                    # else:
                    #     transparent_color = f'rgba({color[0]}, {color[1]}, {color[2]}, 0.6)'
                    #     fig.add_trace(go.Scatter(
                    #         x=xys[estimator][e][0], y=xys[estimator][e][1],
                    #         mode='lines', line=dict(color=transparent_color, width=1, dash='5px,2px'), visible=False,
                    #         name=f'{estimatorName} - CSV {e+1} X', showlegend=False))


        x_min = all_groups[group]['plot_lims']['xmin']
        y_min = all_groups[group]['plot_lims']['ymin']
        x_max = all_groups[group]['plot_lims']['xmax']
        y_max = all_groups[group]['plot_lims']['ymax']


        max_x_abs = max(np.abs(x_min), np.abs(x_max))
        max_y_abs = max(np.abs(y_min), np.abs(y_max))

        x_min = x_min - max_x_abs * 0.01
        x_max = x_max + max_x_abs * 0.01

        y_min = y_min - max_y_abs * 0.01
        y_max = y_max + max_y_abs * 0.01

        # Update layout
        fig.update_layout(
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            template="plotly_white",
            legend=dict(
                yanchor="bottom",
                y=1.03,
                xanchor="left",
                x=-0.1,
                orientation='h',
                bgcolor = 'rgba(0,0,0,0)',
                traceorder='reversed',
                font = dict(family = 'Times New Roman', size=22, color="black"),
                ),
            margin=dict(l=0,r=0,b=0,t=0),
            font = dict(family = 'Times New Roman', size=20, color="black")
        )

        fig.update_yaxes(
            scaleanchor = "x",
            scaleratio = 1
        )

        fig.update_xaxes(
            range=[x_min, x_max],
            autorange=False
        )

        fig.update_yaxes(
            range=[y_min, y_max],
            autorange=False
        )
        print(f'(x_min, x_max): {(x_min, x_max)}')
        print(f'(y_min, y_max): {(y_min, y_max)}')

        fig.write_image(f'/tmp/trajectories_{group}.pdf')

    
        # Show the plot
        fig.show()

def plot_multiple_trajs_video( 
    estimators, exps, colors, estimator_plot_args, path = default_path,  main_expe = 0 ):
    """
    Render an MP4 of 2-D trajectories using Matplotlib’s blitting workflow.
    Requires FFmpeg to be available on the PATH.
    """

    out_mp4="trajectories_all.mp4" 
    fps=30 
    skip_every_n=1000
    dpi=200 

    # ─── 1.  validate & merge plot-args ────────────────────────────
    estimators = list(
    set(estimators)
    .intersection(estimator_plot_args.keys())
    .intersection(estimator_plot_args_default.keys())
    )
    for est in estimators:
        # user-provided args override defaults, but fill in anything missing
        merged = estimator_plot_args_default[est] | estimator_plot_args[est]
        estimator_plot_args[est] = merged

    order = list(estimator_plot_args_default.keys())
    estimators.sort(key=order.index)

    # ─── 2.  load all CSVs into memory ─────────────────────────────
    xys = {est: {k: {0: [], 1: []} for k in range(len(exps))} for est in estimators}
    all_groups = {}

    for k, exp in enumerate(exps):
        df = pd.read_csv(
            os.path.join(path, exp, "output_data", "finalDataCSV.csv"),
            sep=";",
        )
        for est in estimators:
            grp = estimator_plot_args[est]["group"]
            all_groups.setdefault(grp, {"estimators": []})
            if est not in all_groups[grp]["estimators"]:
                all_groups[grp]["estimators"].append(est)

            for axis, suffix in enumerate(["_position_x", "_position_y"]):
                series = df[f"{est}{suffix}"].to_numpy()
                mask = ~np.isnan(series)
                xys[est][k][axis] = series[mask]

    # determine drawing order: group 0 first, then the rest
    ref_grp = 0 if 0 in all_groups else list(all_groups)[0]
    groups = [g for g in all_groups if g != ref_grp]
    combined_estimators = (
        all_groups[ref_grp]["estimators"]
        + [est for g in groups for est in all_groups[g]["estimators"]]
    )

    # ─── 3.  global axis limits ───────────────────────────────────
    def _lim(func, axis):
        vals = [
            func(xys[est][k][axis])
            for est in combined_estimators
            for k in xys[est]
            if xys[est][k][axis].size > 0
        ]
        return func(vals) * 1.1

    x_min, x_max = _lim(np.min, 0), _lim(np.max, 0)
    y_min, y_max = _lim(np.min, 1), _lim(np.max, 1)

    total_samples = max(
        len(xys[est][k][0]) for est in combined_estimators for k in xys[est]
    )
    frame_idx = range(0, total_samples, skip_every_n)
    if len(frame_idx) < 2:
        raise RuntimeError("Need at least two frames – lower skip_every_n.")

    # ─── 4.  initialise Matplotlib figure & line artists ───────────
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X Position [m]")
    ax.set_ylabel("Y Position [m]")
    ax.set_title("Trajectories")

    line_handles = {}                      # (estimator, exp_idx) → Line2D

    for est in combined_estimators:
        r, g, b, _ = colors[est]
        for k in xys[est]:
            is_main = (k == main_expe)
            alpha   = 1.0 if is_main else 0.6
            width   = estimator_plot_args[est]["lineWidth"] + 2 if is_main else 1.5

            # decide on the line style
            if is_main:
                linestyle = "-"           # primary trace → solid
            elif est == "Mocap":
                linestyle = "--"          # non-main Mocap → dashed
            else:
                linestyle = "-"           # all others → solid

            name  = estimator_plot_args[est]["name"].replace("Vanyte", "WAIKO")
            label = name if is_main else f"{name} – CSV {k + 1}"

            (ln,) = ax.plot(
                [], [],
                lw=width,
                ls=linestyle,              # use linestyle instead of dashes
                label=label,
                color=np.array([r, g, b]) / 255,
                alpha=alpha,
            )
            line_handles[(est, k)] = ln

    # legend only for primary trajectories
    main_handles = [line_handles[(est, main_expe)] for est in combined_estimators]
    main_labels  = [
        estimator_plot_args[est]["name"].replace("Vanyte", "WAIKO")
        for est in combined_estimators
    ]
    ax.legend(
        handles=main_handles,
        labels=main_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    # ─── 5.  animate with FFMpegWriter (uses blitting internally) ──
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=[
            "-pix_fmt", "yuv420p",       # 8-bit 4:2:0 = universally supported
            "-profile:v", "high",        # optional; keeps quality high
            "-movflags", "faststart"     # web-friendly “moov” atom at the front
        ],
    )

    
    def _to_mpl_rgb(c):
        """
        Accepts any of the formats you used in Plotly:
        • (R,G,B) ints 0-255 or floats 0-1
        • (R,G,B,A) same ranges
        • '#RRGGBB' or '#RRGGBBAA'

        Returns a plain `(r, g, b)` tuple of floats in 0-1.
        Alpha is ignored because you pass it separately.
        """
        if isinstance(c, (str, np.str_)):
            r, g, b, _ = mcolors.to_rgba(c)
            return (r, g, b)

        # sequence: strip any alpha channel
        if len(c) == 4:
            r, g, b, _ = c
        else:
            r, g, b = c

        r = float(r); g = float(g); b = float(b)
        if max(r, g, b) > 1:            # 0-255 ints → scale down
            return (r / 255., g / 255., b / 255.)
        return (r, g, b)                # already 0-1 floats

    for est in combined_estimators:
        rgb = _to_mpl_rgb(colors[est])        #  <-- single call

        for k in xys[est]:
            is_main = (k == main_expe)
            alpha   = 1.0 if is_main else 0.6
            width   = estimator_plot_args[est]["lineWidth"] + 2 if is_main else 1.5
            linestyle = "--" if (not is_main and est == "Mocap") else "-"

            base_name = estimator_plot_args[est]["name"].replace("Vanyte", "WAIKO")
            label     = base_name if is_main else f"{base_name} – CSV {k + 1}"

            (ln,) = ax.plot(
                [], [],
                lw=width,
                ls=linestyle,
                color=rgb,                 # <-- exact colour preserved
                alpha=alpha,
                label=label if is_main else "_nolegend_",
            )
            line_handles[(est, k)] = ln

    print(f"Encoding {len(frame_idx)} frames → {out_mp4} …")
    with writer.saving(fig, out_mp4, dpi=dpi):
        for t in frame_idx:
            for est in combined_estimators:
                for k in xys[est]:
                    ln = line_handles[(est, k)]
                    history = 10000                       # plot window

                    start = max(0, t + 1 - history)     # inclusive slice start
                    ln.set_data(
                        xys[est][k][0][start : t + 1],
                        xys[est][k][1][start : t + 1],
                    )
            writer.grab_frame()

    plt.close(fig)
    dur = len(frame_idx) / fps
    print(f"Done   ({len(frame_idx)} frames, {dur:.2f}s @ {fps} fps)")
    print(f"(x_min, x_max): {(x_min, x_max)}")
    print(f"(y_min, y_max): {(y_min, y_max)}")



import math


def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Return the yaw angle (rotation around Z) from a unit quaternion."""
    # atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def _to_mpl_rgb(c):
    """Accept hex string or tuple/list in 0-1 or 0-255."""
    try:
        return mcolors.to_rgb(c)
    except Exception:
        c = tuple(c)
        if max(c) > 1.0:
            return tuple(v / 255.0 for v in c)
        return c

def quat_to_yaw(x, y, z, w):
    """Yaw (rotation about Z) from quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def _ema_forward(x: np.ndarray, lam: float) -> np.ndarray:
    """Causal first-order low-pass (EMA): y[n] = (1-lam)*y[n-1] + lam*x[n]."""
    if x.size == 0:
        return x
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    one_minus = 1.0 - lam
    for i in range(1, x.size):
        y[i] = one_minus * y[i-1] + lam * x[i]
    return y

def _lowpass_zero_phase_ema(x: np.ndarray, sample_rate_hz: float, cutoff_hz: float) -> np.ndarray:
    """
    Zero-phase first-order low-pass using forward-backward EMA.
    lam = 1 - exp(-2*pi*fc*dt).  fc in Hz, dt = 1/fs.
    """
    if x.size == 0 or cutoff_hz <= 0 or sample_rate_hz <= 0:
        return x.astype(float).copy()
    dt = 1.0 / sample_rate_hz
    lam = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz * dt)
    y = _ema_forward(x.astype(float), lam)              # forward
    y_rev = _ema_forward(y[::-1], lam)[::-1]            # backward (zero-phase)
    return y_rev

def _cum_arclen(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative arclength S[i] (world frame) with S[0]=0, computed on FILTERED positions."""
    if x.size == 0:
        return np.array([], dtype=float)
    dx = np.diff(x, prepend=x[:1])
    dy = np.diff(y, prepend=y[:1])
    seg = np.hypot(dx, dy)
    seg[0] = 0.0
    return np.cumsum(seg)

def _tail_slice_from_filtered_cumlen(S: np.ndarray, t: int, dist: float, direction: str = "past") -> slice:
    """
    Returns a slice covering the tail around time t:
      - direction="past":   slice(i_best, t+1) with S[t]-S[i_best] ~ dist
      - direction="future": slice(t, j_best+1) with S[j_best]-S[t] ~ dist
    """
    if S.size == 0:
        return slice(t, t + 1)

    t = max(0, min(int(t), len(S) - 1))

    if direction == "past":
        if t == 0:
            return slice(0, 1)
        s_t = S[t]
        target = max(0.0, s_t - dist)
        St = S[:t + 1]

        i_low = int(np.searchsorted(St, target, side="right") - 1)
        i_low = max(0, i_low)
        i_high = min(t, i_low + 1)

        d_low = abs((s_t - St[i_low]) - dist)
        d_high = abs((s_t - St[i_high]) - dist)
        i_best = i_low if d_low <= d_high else i_high
        return slice(i_best, t + 1)

    # direction == "future"
    if t == len(S) - 1:
        return slice(t, t + 1)
    s_t = S[t]
    target = s_t + max(0.0, dist)

    j_high = int(np.searchsorted(S, target, side="left"))
    j_high = max(t, min(j_high, len(S) - 1))
    j_low = max(t, j_high - 1)

    d_low = abs((S[j_low]  - s_t) - dist)
    d_high = abs((S[j_high] - s_t) - dist)
    j_best = j_low if d_low <= d_high else j_high
    return slice(t, j_best + 1)

# ----------------------------
# Main function
# ----------------------------
def plot_relative_trajs_video_distance(
    estimators, exps, colors, estimator_plot_args,
    path=".", main_expe=0,
    sample_rate_hz: float = 250.0,
    cutoff_hz: float = 0.02,
    trail_dist: float = 4.0,
    tail_mode: str = "future",           # "past" or "future"
    out_mp4: str = "trajectories_relative.mp4",
    fps: int = 60,
    skip_every_n: int = 1000,
    dpi: int = 200,
    slow_x: float = 4.0,               # slow down by this factor (≥1)
    slow_mode: str = "duplicate",      # "duplicate" or "fps"
):
    """
    At each frame t:
      • Choose tail length (in meters) from FILTERED Mocap cumulative distance.
      • Plot the tail using RAW positions for every estimator in the robot frame at time t.
      • tail_mode="future": plot indices [t .. j_best]  (or "past" for [i_best .. t])
      • Display a plain speedup label 'x<value>' that accounts for both skip_every_n and the slow-down factor.

    Speedup label:
      speedup = skip_every_n / effective_slow
      where effective_slow = dup (duplicate mode) OR (fps / fps_out) (fps mode).
    """

    # --- 1) Normalize estimators & defaults ---
    estimators = list(dict.fromkeys(estimators))  # preserve order, unique
    for est in estimators:
        estimator_plot_args.setdefault(est, {})
        estimator_plot_args[est].setdefault("group", 0)
        estimator_plot_args[est].setdefault("lineWidth", 2.0)
        estimator_plot_args[est].setdefault("name", est)

    # --- 2) Load & align data; keep RAW for plotting; FILTERED only for Mocap distance ---
    xys_raw = {est: {k: {0: np.array([]), 1: np.array([])} for k in range(len(exps))} for est in estimators}
    yaws    = {est: {k: np.array([]) for k in range(len(exps))} for est in estimators}

    # cumulative length ONLY for Mocap (filtered)
    cumS_mocap = {k: np.array([]) for k in range(len(exps))}
    groups: dict[int, set[str]] = {}

    for k, exp in enumerate(exps):
        df_path = os.path.join(path, exp, "output_data", "finalDataCSV.csv")
        df = pd.read_csv(df_path, sep=";")

        for est in estimators:
            grp = estimator_plot_args[est]["group"]
            groups.setdefault(grp, set()).add(est)

            px = df.get(f"{est}_position_x", pd.Series([np.nan]*len(df))).to_numpy()
            py = df.get(f"{est}_position_y", pd.Series([np.nan]*len(df))).to_numpy()
            qx = df.get(f"{est}_orientation_x", pd.Series([np.nan]*len(df))).to_numpy()
            qy = df.get(f"{est}_orientation_y", pd.Series([np.nan]*len(df))).to_numpy()
            qz = df.get(f"{est}_orientation_z", pd.Series([np.nan]*len(df))).to_numpy()
            qw = df.get(f"{est}_orientation_w", pd.Series([np.nan]*len(df))).to_numpy()

            # Single common mask across all streams for this estimator to keep alignment
            mask = ~(np.isnan(px) | np.isnan(py) | np.isnan(qx) | np.isnan(qy) | np.isnan(qz) | np.isnan(qw))
            if not np.any(mask):
                continue

            rx = px[mask]  # RAW aligned
            ry = py[mask]
            traj_yaw = np.array([quat_to_yaw(a, b, c, d) for a, b, c, d in zip(qx[mask], qy[mask], qz[mask], qw[mask])])

            # Store RAW for plotting
            xys_raw[est][k][0] = rx
            xys_raw[est][k][1] = ry
            yaws[est][k]       = traj_yaw

            # For Mocap ONLY: build filtered cumulative distance (used to choose tail length)
            if est == "Mocap":
                fx = _lowpass_zero_phase_ema(rx, sample_rate_hz, cutoff_hz)
                fy = _lowpass_zero_phase_ema(ry, sample_rate_hz, cutoff_hz)
                cumS_mocap[k] = _cum_arclen(fx, fy)  # meters on FILTERED Mocap

    if not groups:
        raise RuntimeError("No estimator data found after masking.")

    # Reference group first (for legend ordering)
    ref_grp = 0 if 0 in groups else list(groups.keys())[0]
    grouped_lists = {g: [e for e in estimators if e in groups[g]] for g in groups}
    combined_estimators = grouped_lists[ref_grp] + [e for g, lst in grouped_lists.items() if g != ref_grp for e in lst]

    # --- 3) Axis limits from RAW data only ---
    def _lim(func, axis):
        vals = [func(xys_raw[e][k][axis]) for e in combined_estimators for k in xys_raw[e] if xys_raw[e][k][axis].size]
        return func(vals) * 1.1 if vals else 0.0

    max_extent = max(
        abs(_lim(np.min, 0)),
        abs(_lim(np.max, 0)),
        abs(_lim(np.min, 1)),
        abs(_lim(np.max, 1)),
    )
    lim = max(max_extent, trail_dist) * 1.1 or 1.0

    # --- 3b) Frame indices ---
    total_samples = 0
    for e in combined_estimators:
        for k in xys_raw[e]:
            total_samples = max(total_samples, len(xys_raw[e][k][0]))
    if total_samples <= 1:
        raise RuntimeError("Not enough samples to animate.")

    frame_idx = range(0, total_samples, skip_every_n)
    if len(frame_idx) < 2:
        raise RuntimeError("Need at least two frames – lower skip_every_n.")

    # --- 4) Figure & lines ---
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    mode_str = "past" if tail_mode == "past" else "future"
    # ax.set_title(
    #     f"Relative Trajectories (tail={mode_str}, set by FILTERED Mocap fc={cutoff_hz:g} Hz; plot uses RAW positions)"
    # )

    line_handles = {}
    for est in combined_estimators:
        rgb = _to_mpl_rgb(colors[est])
        for k in xys_raw[est]:
            is_main = (k == main_expe)
            (ln,) = ax.plot(
                [], [],
                lw=(estimator_plot_args[est]["lineWidth"] + 2 if is_main else 1.5),  # <-- original linewidth logic
                ls="--" if (not is_main and est == "Mocap") else "-",
                color=rgb,
                alpha=1.0 if is_main else 0.6,
                label=(estimator_plot_args[est]["name"] if is_main else "_nolegend_"),
            )
            line_handles[(est, k)] = ln

    ax.legend(
        handles=[line_handles[(e, main_expe)] for e in combined_estimators if (e, main_expe) in line_handles],
        labels=[estimator_plot_args[e]["name"] for e in combined_estimators if (e, main_expe) in line_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=max(1, len(combined_estimators)),
        frameon=False,
        fontsize=8,
    )

    robot_img = mpimg.imread("hrp5p.png")
    img_h, img_w = robot_img.shape[0], robot_img.shape[1]
    aspect = img_h / img_w
    real_width = 0.8   # meters
    real_height = real_width * aspect

    # Create an AxesImage artist once
    robot_artist = ax.imshow(
        robot_img,
        extent=[0, real_width, -real_height/2, real_height/2],  # will be adjusted for tail_mode
        zorder=10
    )

    # --- 5) Playback control & speedup label ---
    if slow_mode not in ("duplicate", "fps"):
        slow_mode = "duplicate"

    if slow_mode == "duplicate":
        dup = max(1, int(round(slow_x)))   # render each frame 'dup' times
        fps_out = fps                      # keep original fps
        effective_slow = float(dup)
    else:  # "fps"
        fps_out = max(1, int(round(fps / max(1e-9, slow_x))))
        dup = 1
        effective_slow = float(fps) / float(fps_out)

    # Speedup factor that reflects BOTH skipping and slowdown
    # (x=1 -> speedup = skip_every_n; increasing x reduces speedup accordingly)
    speedup_factor = float(skip_every_n) / max(1e-9, effective_slow)

    # Writer
    writer = FFMpegWriter(
        fps=fps_out,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-profile:v", "high", "-movflags", "faststart"],
    )

    # Plain "x<speedup>" label (no box), bottom-right
    speed_label = ax.text(
        0.99, 0.01, f"x{speedup_factor:.2f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, fontweight="bold"
    )

    # --- 6) Animation ---
    print(f"Encoding frames → {out_mp4} … (fps_out={fps_out}, dup={dup}, speedup≈x{speedup_factor:.2f})")
    with writer.saving(fig, out_mp4, dpi=dpi):
        for t in frame_idx:
            # ----------------- choose tail ONCE from Mocap (FILTERED) -----------------
            n_iters = 1
            tail_len_m = 0.0
            S_moc = cumS_mocap.get(main_expe, np.array([]))
            # fallback: first available Mocap run that covers t
            if S_moc.size == 0 or t >= len(S_moc):
                for k in cumS_mocap:
                    if cumS_mocap[k].size > 0 and t < len(cumS_mocap[k]):
                        S_moc = cumS_mocap[k]
                        break

            if S_moc.size > 0 and t < len(S_moc):
                sl_mocap = _tail_slice_from_filtered_cumlen(S_moc, t, trail_dist, direction=tail_mode)
                if tail_mode == "past":
                    i0 = sl_mocap.start
                    n_iters = int(t - i0 + 1)
                    tail_len_m = float(S_moc[t] - S_moc[i0])
                else:  # future
                    j1 = sl_mocap.stop - 1
                    n_iters = int(j1 - t + 1)


            # Draw each estimator using RAW positions and shared n_iters
            for est in combined_estimators:
                for k in xys_raw[est]:
                    rx = xys_raw[est][k][0]  # RAW
                    ry = xys_raw[est][k][1]
                    yaw = yaws[est][k]

                    if len(rx) == 0 or t >= len(rx):
                        continue

                    if tail_mode == "past":
                        start_i = max(0, t - n_iters + 1)
                        sl = slice(start_i, t + 1)
                    else:  # future
                        end_j = min(len(rx), t + n_iters)  # stop is exclusive
                        sl = slice(t, end_j)

                    # Robot frame at time t from RAW pose & yaw
                    x0, y0 = rx[t], ry[t]
                    yaw0 = yaw[t] if t < len(yaw) else 0.0
                    cy, sy = math.cos(yaw0), math.sin(yaw0)

                    tail_x_w = rx[sl]  # RAW positions for plotting
                    tail_y_w = ry[sl]
                    dx = tail_x_w - x0
                    dy = tail_y_w - y0
                    x_rel =  cy * dx + sy * dy
                    y_rel = -sy * dx + cy * dy

                    line_handles[(est, k)].set_data(x_rel, y_rel)

                    if tail_mode == "future":
                        # Align the anchor at 0.465 m from the left
                        shift = 0.465
                        extent = [-shift, real_width - shift, -real_height/2, real_height/2]
                    else:  # past
                        # Align left edge at 0.0
                        extent = [0, real_width, -real_height/2, real_height/2]

                    robot_artist.set_extent(extent) 

            # Grab (possibly duplicated) frames
            for _ in range(dup):
                writer.grab_frame()

    plt.close(fig)
    print(f"Done ({len(frame_idx)} frames, {len(frame_idx)/fps:.2f}s @ {fps} fps)")




if __name__ == '__main__':
    # Generate color palette for the estimators
    colors_t = px.colors.qualitative.Plotly  # Use Plotly's color palette
    colors_t = [px.colors.hex_to_rgb(colors_t[i]) for i in range(len(default_estimators))]
    colors = dict.fromkeys(default_estimators)
    for i,estimator in enumerate(colors.keys()):
        colors[estimator] = colors_t[i]
    
    #plot_multiple_trajs(default_estimators, default_exps, colors, estimator_names_to_plot_default)
    plot_multiple_trajs(default_estimators, default_exps, colors)