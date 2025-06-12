import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  # For color palette generation
from plotly.subplots import make_subplots

from pathlib import Path

import plotly.io as pio
import os

# Tell webbrowser to use wslview
os.environ["BROWSER"] = "wslview"
pio.renderers.default = "browser"


default_path = '.../Projects/'

# default_exps = [
#     'HRP5_MultiContact_1', 
#     'HRP5_MultiContact_2, 
#     'HRP5_MultiContact_3', 
#     'HRP5_MultiContact_4'
# ]

# default_exps = [
#     'KO_TRO_2024_RHPS1_SLIPPAGE_1', 
#     'KO_TRO_2024_RHPS1_SLIPPAGE_2', 
#     'KO_TRO_2024_RHPS1_SLIPPAGE_3'
# ]

default_exps = [
    'KO_TRO2024_RHPS1_1', 
    'KO_TRO2024_RHPS1_2', 
    'KO_TRO2024_RHPS1_3', 
    'KO_TRO2024_RHPS1_4', 
    'KO_TRO2024_RHPS1_5'
]

default_estimators = [
    'Controller',
    'Vanyte',
    'Hartley',
    'KineticsObserver',
    'KO_APC',
    'KO_ASC',
    'KO_ZPC',
    'KOWithoutWrenchSensors',
    'Mocap',
    'Tilt'
]


# Define columns for each estimator
estimator_plot_args_default = {
    'KineticsObserver': {'group': 0, 'lineWidth': 3, 'column_names': ['KO_posW_tx', 'KO_posW_ty']},
    'Controller': {'group': 1, 'lineWidth': 2, 'column_names': ['Controller_tx', 'Controller_ty']},
    #'Vanyte': {'group': 1, 'lineWidth': 2, 'column_names': ['Vanyte_pose_tx', 'Vanyte_pose_ty']},
    'Hartley': {'group': 1, 'lineWidth': 2, 'column_names':  ['Hartley_Position_x', 'Hartley_Position_y']},
    #'KO_APC': {'group': 1, 'lineWidth': 2, 'column_names': ['KO_APC_posW_tx', 'KO_APC_posW_ty']},
    #'KO_ASC': {'group': 2, 'lineWidth': 2, 'column_names': ['KO_ASC_posW_tx', 'KO_ASC_posW_ty']},
    'KO_ZPC': {'group': 1, 'lineWidth': 2, 'column_names': ['KO_ZPC_posW_tx', 'KO_ZPC_posW_ty']},
    #'KOWithoutWrenchSensors': {'group': 1, 'lineWidth': 2, 'column_names': ['KOWithoutWrenchSensors_posW_tx', 'KOWithoutWrenchSensors_posW_ty']},
    'Mocap': {'group': 0, 'lineWidth': 3, 'column_names': ['Mocap_pos_x', 'Mocap_pos_y']},
    'Tilt': {'group': 0, 'lineWidth': 3, 'column_names': ['Tilt_pose_tx', 'Tilt_pose_ty']}
}



# def plot_multiple_trajs_per_expe(estimators, exps, colors, estimator_plot_args = estimator_plot_args_default, path = default_path):
#     estimators = list(filter(lambda x: x in estimator_plot_args, estimators))

#     home = str(Path.home())

#     xys = dict.fromkeys(estimators)

#     all_columns = []
#     all_groups_keys = set()

#     for estimator in estimators:
#         all_groups_keys.add(estimator_plot_args[estimator]['group'])
#         xys[estimator] = dict.fromkeys(range(len(exps)))
#         for k in range(len(exps)):
#             xys[estimator][k] = {0: [], 1:[]}
#         for col in estimator_plot_args[estimator]['column_names']:
#             all_columns.append(col)

#     all_groups = dict.fromkeys(all_groups_keys)
    
#     all_columns.append("Mocap_datasOverlapping")

#     for group in all_groups.keys():
#         all_groups[group] = {'estimators': [], 'plot_lims': {'xmin': {}, 'xmax': {}, 'ymin': {}, 'ymax': {}}}
#     for estimator in estimators:
#         all_groups[estimator_plot_args[estimator]['group']]["estimators"].append(estimator)

#     for e, exp in enumerate(exps):
#         file = f'{path}{exp}/output_data/observerResultsCSV.csv'
#         df = pd.read_csv(file, sep=';', usecols=all_columns)
#         df_overlap = df[df["Mocap_datasOverlapping"] == "Datas overlap"]
#         for estimator in estimators:
#             xys[estimator][e][0] = df_overlap[estimator_plot_args[estimator]['column_names'][0]]
#             xys[estimator][e][1] = df_overlap[estimator_plot_args[estimator]['column_names'][1]]
    
    

#     for group in list(filter(lambda x: x != 0, all_groups.keys())):
#         xmins = []
#         xmaxs = []
#         ymins = []
#         ymaxs = []
#         combined_estimators = all_groups[group]["estimators"] + all_groups[0]["estimators"]
#         for estimator in combined_estimators:
#             for e in range(len(exps)):
#                 xmins.append(min(xys[estimator][e][0]))
#                 xmaxs.append(max(xys[estimator][e][0]))
#                 ymins.append(min(xys[estimator][e][1]))
#                 ymaxs.append(max(xys[estimator][e][1]))

#         all_groups[group]['plot_lims']['xmin'] = min(xmins)
#         all_groups[group]['plot_lims']['xmax'] = max(xmaxs)
#         all_groups[group]['plot_lims']['ymin'] = min(ymins)
#         all_groups[group]['plot_lims']['ymax'] = max(ymaxs)
                

#     # Create a Plotly figure
    
#     for group in list(filter(lambda x: x != 0, all_groups.keys())):
#         fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True)
#         combined_estimators = all_groups[group]["estimators"] + all_groups[0]["estimators"]
#         for estimator in combined_estimators:
#             estimatorName = estimator_plot_args[estimator]["name"]
#             color = colors[estimator]

#             # Process each CSV for the current estimator
#             for e in xys[estimator].keys():
#                 if(estimator == 'Mocap'):
#                     transparent_color = f'rgba({color[0]}, {color[1]}, {color[2]}, 1)'
#                     # Use transparent_color in the line color
#                     fig.add_trace(go.Scatter(
#                         x=xys[estimator][e][0], y=xys[estimator][e][1],
#                         mode='lines', line=dict(color=transparent_color, width=2),
#                         name=f'{estimatorName}', showlegend=True), row = e+1, col = 1)
#                 else:
#                     transparent_color = f'rgba({color[0]}, {color[1]}, {color[2]}, 1)'
#                     fig.add_trace(go.Scatter(
#                         x=xys[estimator][e][0], y=xys[estimator][e][1],
#                         mode='lines', line=dict(color=transparent_color, width=2),
#                         name=f'{estimatorName}', showlegend=True), row = e+1, col = 1)
               

#         x_min = all_groups[group]['plot_lims']['xmin']
#         y_min = all_groups[group]['plot_lims']['ymin']
#         x_max = all_groups[group]['plot_lims']['xmax']
#         y_max = all_groups[group]['plot_lims']['ymax']

#         # Update layout
#         fig.update_layout(
#             xaxis_title="X Position (m)",
#             yaxis_title="Y Position (m)",
#             template="plotly_white",
#             legend=dict(
#                 yanchor="top",
#                 y=0.99,
#                 xanchor="left",
#                 x=0.01,
#                 orientation='h',
#                 bgcolor = 'rgba(0,0,0,0)',
#                 traceorder='reversed'
#                 ),
#             margin=dict(l=0,r=0,b=0,t=0)
#         )

#         fig.update_yaxes(
#             scaleanchor = "x",
#             scaleratio = 1
#         )

#         fig.update_xaxes(
#             range=[x_min, x_max]
#         )

#         fig.update_yaxes(
#             range=[y_min, y_max]
#         )
#         fig.write_image(f'{home}/Downloads/test{group}.png')

    
#         # Show the plot
#         fig.show()




def plot_multiple_trajs(estimators, exps, colors, estimator_plot_args, path = default_path,  main_expe = 0):
    #estimators = list(set(estimators).intersection(estimator_plot_args_default.keys()))
    estimators = [x for x in estimators if x in estimator_plot_args_default]

    for estimatorName in estimators:
        estimator_plot_args[estimatorName].update(estimator_plot_args_default[estimatorName])

    xys = dict.fromkeys(estimators)

    all_columns = []
    all_groups_keys = set()

    for estimator in estimators:
        all_groups_keys.add(estimator_plot_args[estimator]['group'])
        xys[estimator] = dict.fromkeys(range(len(exps)))
        for k in range(len(exps)):
            xys[estimator][k] = {0: [], 1:[]}
        for col in estimator_plot_args[estimator]['column_names']:
            all_columns.append(col)

    all_groups = dict.fromkeys(all_groups_keys)
    
    all_columns.append("Mocap_datasOverlapping")

    for group in all_groups.keys():
        all_groups[group] = {'estimators': [], 'plot_lims': {'xmin': {}, 'xmax': {}, 'ymin': {}, 'ymax': {}}}
    for estimator in estimators:
        all_groups[estimator_plot_args[estimator]['group']]["estimators"].append(estimator)

    for e, exp in enumerate(exps):
        file = f'{path}{exp}/output_data/observerResultsCSV.csv'
        df = pd.read_csv(file, sep=';', usecols=all_columns)
        df_overlap = df[df["Mocap_datasOverlapping"] == "Datas overlap"]
        for estimator in estimators:
            xys[estimator][e][0] = df_overlap[estimator_plot_args[estimator]['column_names'][0]]
            xys[estimator][e][1] = df_overlap[estimator_plot_args[estimator]['column_names'][1]]
    
    

    for group in list(filter(lambda x: x != 0, all_groups.keys())):
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        combined_estimators = all_groups[0]["estimators"] + all_groups[group]["estimators"]
        combined_estimators = sorted(combined_estimators, key=estimators.index)
        for estimator in combined_estimators:
            for e in range(len(exps)):
                xmins.append(min(xys[estimator][e][0]))
                xmaxs.append(max(xys[estimator][e][0]))
                ymins.append(min(xys[estimator][e][1]))
                ymaxs.append(max(xys[estimator][e][1]))

        all_groups[group]['plot_lims']['xmin'] = min(xmins)
        all_groups[group]['plot_lims']['xmax'] = max(xmaxs)
        all_groups[group]['plot_lims']['ymin'] = min(ymins)
        all_groups[group]['plot_lims']['ymax'] = max(ymaxs)
                

    # Create a Plotly figure
    
    for group in list(filter(lambda x: x != 0, all_groups.keys())):
        fig = go.Figure()
        combined_estimators = all_groups[0]["estimators"] + all_groups[group]["estimators"]
        combined_estimators = sorted(combined_estimators, key=estimators.index)
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


import os
import cv2

def generate_video_from_trajs(estimators, exps, colors, estimator_plot_args, path=default_path, main_expe=0, fps=10, video_output="output_video.mp4"):
    estimators = [x for x in estimators if x in estimator_plot_args_default]
    
    for estimatorName in estimators:
        estimator_plot_args[estimatorName].update(estimator_plot_args_default[estimatorName])

    xys = dict.fromkeys(estimators)
    all_columns = []

    for estimator in estimators:
        xys[estimator] = dict.fromkeys(range(len(exps)))
        for k in range(len(exps)):
            xys[estimator][k] = {0: [], 1: []}
        for col in estimator_plot_args[estimator]['column_names']:
            all_columns.append(col)
    all_columns.append("Mocap_datasOverlapping")

    for e, exp in enumerate(exps):
        file = f'{path}{exp}/output_data/observerResultsCSV.csv'
        df = pd.read_csv(file, sep=';', usecols=all_columns)
        df_overlap = df[df["Mocap_datasOverlapping"] == "Datas overlap"]
        for estimator in estimators:
            xys[estimator][e][0] = df_overlap[estimator_plot_args[estimator]['column_names'][0]].values
            xys[estimator][e][1] = df_overlap[estimator_plot_args[estimator]['column_names'][1]].values

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
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
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