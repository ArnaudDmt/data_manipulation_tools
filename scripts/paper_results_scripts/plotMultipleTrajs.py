import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  # For color palette generation
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
from pathlib import Path

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


import os
import cv2

def generate_video_from_trajs(estimators, exps, colors, estimator_plot_args, path=default_path, main_expe=0, fps=10, video_output="output_video.mp4"):
    estimators = [x for x in estimators if x in estimator_plot_args_default]
    
    for estimatorName in estimators:
        estimator_plot_args[estimatorName].update(estimator_plot_args_default[estimatorName])

    xys = dict.fromkeys(estimators) 

    for e, exp in enumerate(exps):
        file = f'{path}{exp}/output_data/observerResultsCSV.csv'
        df = pd.read_csv(file, sep=';')
        for estimator in estimators:
            xys[estimator][e][0] = df[[estimator + '_position_x']].values
            xys[estimator][e][1] = df[[estimator + '_position_y']].values.values



    # Prepare a temporary directory for saving frames
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    num_frames = len(next(iter(xys.values()))[0][0])  # Number of points

    fig = go.Figure()

    # Create traces once
    traces = []
    for estimator in estimators:
        color = colors[estimator]
        transparent_color = f'rgba({color[0]}, {color[1]}, {color[2]}, 1)'
        for e in range(len(exps)):
            trace = go.Scatter(
                x=[],  # Start with empty data
                y=[],
                mode='lines+markers',
                name=f'{estimator}',
                line=dict(color=transparent_color, width=2)
            )
            traces.append(trace)
            fig.add_trace(trace)

    fig.update_layout(
        xaxis_title="X Position [m]",
        yaxis_title="Y Position [m]",
        template="plotly_white",
        showlegend=True
    )

    # Accumulate and save frames
    for frame_idx in range(num_frames):
        if frame_idx % 10 == 0:
            for i, estimator in enumerate(estimators):
                for e in range(len(exps)):
                    #print(f'remaining: {num_frames - frame_idx}')
                    # Access the trace data and update it
                    trace = fig.data[i * len(exps) + e]
                    trace.x = list(trace.x) + [xys[estimator][e][0][frame_idx]]
                    trace.y = list(trace.y) + [xys[estimator][e][1][frame_idx]]

            # Save the updated figure for this frame
            frame_filename = f"{temp_dir}/frame_{frame_idx:06d}.png"
            fig.write_image(frame_filename)

    # Combine frames into a video
    frame_files = sorted(os.listdir(temp_dir))
    frame_path = os.path.join(temp_dir, frame_files[0])
    frame = cv2.imread(frame_path)
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(temp_dir, frame_file))
        video.write(frame)

    video.release()

    # Clean up
    for frame_file in frame_files:
        os.remove(os.path.join(temp_dir, frame_file))
    os.rmdir(temp_dir)
    print(f"Video saved to {video_output}")




if __name__ == '__main__':
    # Generate color palette for the estimators
    colors_t = px.colors.qualitative.Plotly  # Use Plotly's color palette
    colors_t = [px.colors.hex_to_rgb(colors_t[i]) for i in range(len(default_estimators))]
    colors = dict.fromkeys(default_estimators)
    for i,estimator in enumerate(colors.keys()):
        colors[estimator] = colors_t[i]
    
    #plot_multiple_trajs(default_estimators, default_exps, colors, estimator_names_to_plot_default)
    plot_multiple_trajs(default_estimators, default_exps, colors)