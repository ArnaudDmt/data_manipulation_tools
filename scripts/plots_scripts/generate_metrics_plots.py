#!./env/bin/python

import os
import sys
import yaml
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

import plotly.io as pio

# Tell webbrowser to use wslview
os.environ["BROWSER"] = "wslview"
pio.renderers.default = "browser"

import pathlib
cwd = pathlib.Path(__file__).parent.resolve()
sys.path.append(f'{cwd.parent}')
sys.path.append(f'{cwd.parent}/paper_results_scripts')


def float_representer(dumper, value):
    # Check if the value is already in scientific notation
    if "e" in f"{value}":
        # Format scientific notation with 1 digit after the decimal
        text = f"{value:.1e}"
    else:
        # Convert the float to a string with high precision for analysis
        precise_str = f"{value:.8f}"  # High precision for detecting non-zero digits
        non_zero_index = next((i for i, char in enumerate(precise_str) if char not in {'0', '.'}), None)
        
        if non_zero_index is not None and non_zero_index > 5:
            # If the first non-zero digit is after the 4th decimal place
            text = f"{value:.1e}"

        else:
            # Standard float: round to 3 decimal places
            text = f"{value:.3f}"
    
    return dumper.represent_scalar(u'tag:yaml.org,2002:str', text)


yaml.add_representer(float, float_representer)

import pandas as pd
import pickle


from copy import deepcopy


import argparse

estimator_plot_args = dict()
with open(f'{cwd}/../../observersInfos.yaml', 'r') as file:
    try:
        observersInfos_str = file.read()
        observersInfos_yamlData = yaml.safe_load(observersInfos_str)
        for observer in observersInfos_yamlData['observers']:
            estimator_plot_args[observer["abbreviation"]] = dict()
            estimator_plot_args[observer["abbreviation"]]['name'] = observer["name"]
            estimator_plot_args[observer["abbreviation"]]['lineWidth'] = observer["lineWidth"]
    except yaml.YAMLError as exc:
        print(exc)

estimators_to_plot = [
    'KO',
    'Control',
    'WAIKO',
    'Tilt',
    
    'Vanyte',
    'Hartley',
    
    # 'KineticsObserver',
    # 'KO_APC',
    # 'KO_ASC',
    # 'KO_ZPC',
    # 'KOWithoutWrenchSensors',
    
    'Mocap',
]


def reduce_intensity(color, amount=0.7):
    """Blend the color with gray to reduce intensity."""
    r, g, b, _ = color
    gray = 0.2  # Gray defined as (0.5, 0.5, 0.5) in RGB
    r = (1 - amount) * gray + amount * r
    g = (1 - amount) * gray + amount * g
    b = (1 - amount) * gray + amount * b
    return (r, g, b, 1)  # Return as (R, G, B, Alpha)

def generate_turbo_subset_colors(estimatorsList):
    cmap = plt.get_cmap('turbo')
    listCoeffs = np.linspace(0.2, 0.8, len(estimatorsList))
    colors={}
    # Generate colors and reduce intensity
    for idx, estimator in enumerate(estimatorsList):
        colors[estimator] = reduce_intensity(cmap(listCoeffs[idx]), 0.75)
    
    return colors

def lighten_color(color, amount=0.5):
    r, g, b, _ = color
    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {amount})"

def open_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        return data


def plot_relative_error_statistics_as_boxplot(errorStats, colors):    
    f = open('/tmp/relative_errors.yaml', 'w+')
    yaml.dump(errorStats, f, allow_unicode=True, default_flow_style=False)
    fig = go.Figure()

    all_categories = sorted({cat for distance in errorStats.values() for estimator in distance.values() for cat in estimator.keys()})
    
    dropdown_buttons = []
    
    traces_per_category = {category: [] for category in all_categories}

    # Iterate over categories and create a trace per estimator-category pair
    first_key = next(iter(errorStats))
    for estimator in errorStats[first_key].keys():  # We assume the same estimators are present for all d_subTraj
        for category in all_categories:
            x_vals = []  # Collect x values (d_subTraj)
            lower_fence = []
            q1 = []
            mean = []
            meanAbs = []
            median = []
            q3 = []
            upper_fence = []
            rmse = []

            # Collect the data across all sub-trajectory lengths for each estimator-category pair
            for d_subTraj in errorStats.keys():
                stats = errorStats[d_subTraj][estimator][category]
                x_vals.append(d_subTraj)
                lower_fence.append(stats['min'])
                q1.append(stats['q1'])
                mean.append(stats['mean'])
                meanAbs.append(stats['meanAbs'])
                median.append(stats['median'])
                q3.append(stats['q3'])
                upper_fence.append(stats['max'])
                rmse.append(stats['rmse'])


            trace = go.Box(
                x=x_vals,  # X-axis corresponds to sub-trajectory lengths
                lowerfence=lower_fence,
                q1=q1,
                mean=mean,
                median=median,
                q3=q3,
                upperfence=upper_fence,
                name=f"{estimator_plot_args[estimator]['name']} ({category})",  # Single legend entry per estimator-category pair
                boxpoints=False,
                marker_color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)",  # Outline color
                fillcolor=lighten_color(colors[estimator], 0.3),  # Slightly lighter and transparent background
                line=dict(width=estimator_plot_args[estimator]['lineWidth'], color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)"),  # Well-visible outline
                opacity=0.8,
                visible=False  # Initially hidden
            )

            trace2 = go.Scatter(x=x_vals, y=rmse, name=f"{estimator_plot_args[estimator]['name']} rmse ({category})", marker_color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)", visible=False, mode='markers')


            traces_per_category[category].append(trace)
            fig.add_trace(trace)

            traces_per_category[category].append(trace2)
            fig.add_trace(trace2)

    # Create dropdown buttons to toggle visibility for each category
    for category in all_categories:
        visibility = [False] * len(fig.data)  # Start with all traces hidden
        for i, trace in enumerate(fig.data):
            if trace.name.endswith(f"({category})"):
                visibility[i] = True  # Show traces of the selected category

        button = dict(
            label=category,
            method='update',
            args=[{'visible': visibility}, {'title': rel_category_titles.get(category, 'Default Label'), 'yaxis.title': rel_category_ylabels.get(category, 'Default Label')}]
        )
        dropdown_buttons.append(button)

    fig.update_layout(
        title='Select a Category to Display Boxplots',
        xaxis_title='Sub-trajectory length [m]',
        updatemenus=[
            {
                'buttons': dropdown_buttons,
                'direction': 'down',
                'showactive': True,
            }
        ],
        boxmode='group'  # Group boxes together
    )

    fig.show()



def plot_relative_errors(exps_to_merge, estimatorsList, colors):
    data1 = open_pickle(f"Projects/{exps_to_merge[0]}/output_data/evals/{estimatorsList[0]}/saved_results/traj_est/cached/cached_rel_err.pickle")
    regroupedErrors = dict.fromkeys(data1.keys())

    for d_subTraj in regroupedErrors.keys():
        regroupedErrors[d_subTraj] = dict.fromkeys(estimatorsList)
        for estimator in regroupedErrors[d_subTraj].keys():
            cats = list(data1[list(data1.keys())[0]].keys())
            regroupedErrors[d_subTraj][estimator] = dict.fromkeys(cats)

    for expe in exps_to_merge:        
        for estimator in regroupedErrors[d_subTraj].keys():
            data = open_pickle(f"Projects/{expe}/output_data/evals/{estimator}/saved_results/traj_est/cached/cached_rel_err.pickle")
            for d_subTraj in regroupedErrors.keys():
                for cat in data[d_subTraj]:
                    if isinstance(data[d_subTraj][cat], np.ndarray):
                        if regroupedErrors[d_subTraj][estimator][cat] is None:
                            regroupedErrors[d_subTraj][estimator][cat] = data[d_subTraj][cat]
                        else:
                            regroupedErrors[d_subTraj][estimator][cat] = np.concatenate((regroupedErrors[d_subTraj][estimator][cat], data[d_subTraj][cat]))

    errorStats = dict.fromkeys(regroupedErrors.keys())
    for d_subTraj in regroupedErrors:
        errorStats[d_subTraj] = dict.fromkeys(regroupedErrors[d_subTraj].keys())
        for cat in cats:
            if "stats" in cat:
                cats.remove(cat)
        for estimator in errorStats[d_subTraj]:
            errorStats[d_subTraj][estimator] = dict.fromkeys(cats)
            for cat in cats:
                errorStats[d_subTraj][estimator][cat]  = {
                                        'rmse': 0.0, 'mean': 0.0, 'meanAbs': 0.0, 'median': 0.0, 'q1': 0.0, 'q3': 0.0, 
                                        'std': 0.0, 'min': 0.0, 'max': 0.0  }
                data_vec = regroupedErrors[d_subTraj][estimator][cat]
                errorStats[d_subTraj][estimator][cat]['rmse'] = float(np.sqrt(np.dot(data_vec, data_vec) / len(data_vec)))
                errorStats[d_subTraj][estimator][cat]['mean'] = float(np.mean(data_vec))
                errorStats[d_subTraj][estimator][cat]['meanAbs'] = float(np.mean(np.abs(data_vec)))
                errorStats[d_subTraj][estimator][cat]['median'] = float(np.median(data_vec))
                errorStats[d_subTraj][estimator][cat]['q1'] = float(np.quantile(data_vec, 0.25))
                errorStats[d_subTraj][estimator][cat]['q3'] = float(np.quantile(data_vec, 0.75))
                errorStats[d_subTraj][estimator][cat]['std'] = float(np.std(data_vec))
                errorStats[d_subTraj][estimator][cat]['min'] = float(np.min(data_vec))
                errorStats[d_subTraj][estimator][cat]['max'] = float(np.max(data_vec))

    plot_relative_error_statistics_as_boxplot(errorStats, colors)


def plot_absolute_error_statistics_as_boxplot(errorStats, colors):  
    f = open('/tmp/absolute_errors.yaml', 'w+')
    yaml.dump(errorStats, f, allow_unicode=True, default_flow_style=False)

    fig = go.Figure()

    all_categories = sorted({cat for distance in errorStats.values() for estimator in distance.values() for cat in estimator.keys()})
    
    dropdown_buttons = []
    
    traces_per_category = {category: [] for category in all_categories}

    # Iterate over categories and create a trace per estimator-category pair
    first_key = next(iter(errorStats))
    for estimator in errorStats[first_key].keys():
        for category in all_categories:
            x_vals = []  # Collect x values (d_subTraj)
            lower_fence = []
            q1 = []
            mean = []
            meanAbs = []
            median = []
            q3 = []
            upper_fence = []
            rmse = []

            # Collect the data across all sub-trajectory lengths for each estimator-category pair
            for expe in errorStats.keys():
                stats = errorStats[expe][estimator][category]
                x_vals.append(expe)
                lower_fence.append(stats['min'])
                q1.append(stats['q1'])
                mean.append(stats['mean'])
                meanAbs.append(stats['meanAbs'])
                median.append(stats['median'])
                q3.append(stats['q3'])
                upper_fence.append(stats['max'])
                rmse.append(stats['rmse'])

            trace = go.Box(
                x=x_vals,  # X-axis corresponds to sub-trajectory lengths
                lowerfence=lower_fence,
                q1=q1,
                mean=mean,
                median=median,
                q3=q3,
                upperfence=upper_fence,
                name=f"{estimator_plot_args[estimator]['name']} ({category})",  # Single legend entry per estimator-category pair
                boxpoints=False,
                marker_color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)",  # Outline color
                fillcolor=lighten_color(colors[estimator], 0.3),  # Slightly lighter and transparent background
                line=dict(width=estimator_plot_args[estimator]['lineWidth'], color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)"),  # Well-visible outline
                opacity=0.8,
                visible=False  # Initially hidden
            )

            trace2 = go.Scatter(x=x_vals, y=rmse, name=f"{estimator_plot_args[estimator]['name']} rmse ({category})", marker_color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)", visible=False, mode='markers')

            traces_per_category[category].append(trace)
            fig.add_trace(trace)

            traces_per_category[category].append(trace2)
            fig.add_trace(trace2)

    # Create dropdown buttons to toggle visibility for each category
    for category in all_categories:
        visibility = [False] * len(fig.data)  # Start with all traces hidden
        for i, trace in enumerate(fig.data):
            if trace.name.endswith(f"({category})"):
                visibility[i] = True  # Show traces of the selected category

        button = dict(
            label=category,
            method='update',
            args=[{'visible': visibility}, {'title': abs_category_titles.get(category, 'Default Label'), 'yaxis.title': abs_category_ylabels.get(category, 'Default Label')}]
        )
        dropdown_buttons.append(button)

    fig.update_layout(
        title='Select a Category to Display Boxplots',
        xaxis_title='Sub-trajectory length [m]',
        updatemenus=[
            {
                'buttons': dropdown_buttons,
                'direction': 'down',
                'showactive': True,
            }
        ],
        boxmode='group'  # Group boxes together
    )
    fig.show()

def plot_absolute_errors_raw(exps_to_merge, estimatorsList, colors):
    fig = go.Figure()
    dropdown_buttons = []
    tracesNames = []
    for expe in exps_to_merge:
        for estimator in estimatorsList:
            data = open_pickle(f"Projects/{expe}/output_data/evals/{estimator}/saved_results/traj_est/cached/cached_abs_err.pickle")
            for category in data.keys():
                if(isinstance(data[category],np.ndarray)):
                    if(data[category].size == len(data[category])):
                        name_t = f"{expe}_{estimator_plot_args[estimator]['name']}({category})"
                        tracesNames.append(name_t)
                        fig.add_trace(go.Scatter(x=data['accum_distances'], y=data[category], fill='tozeroy', name=name_t, line_color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)", visible=False, fillcolor=lighten_color(colors[estimator], 0.3)))
                    else:
                        for i in range(3):
                            name_t = f"{expe}_{estimator_plot_args[estimator]['name']}({category}_{i})"
                            tracesNames.append(name_t)
                            fig.add_trace(go.Scatter(x=data['accum_distances'], y=data[category][:,i], fill='tozeroy', name=name_t, line_color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)", visible=False, fillcolor=lighten_color(colors[estimator], 0.3)))
                
    for category in data.keys():
        visibility = [False] * len(fig.data)  # Start with all traces hidden
        for i, trace in enumerate(fig.data):
            if category in trace.name:
                visibility[i] = True  # Show traces of the selected category

        button = dict(
            label=category,
            method='update',
            args=[{'visible': visibility}, {'title': abs_category_titles.get(category, 'Default Label'), 'yaxis.title': abs_category_ylabels.get(category, 'Default Label')}]
        )
        dropdown_buttons.append(button)

    fig.update_layout(
        title='Select a Category to Display the absolute error plots',
        xaxis_title="Distance travelled (m)", yaxis_title="Error",
        updatemenus=[
            {
                'buttons': dropdown_buttons,
                'direction': 'down',
                'showactive': True,
            }
        ],
        boxmode='group'  # Group boxes together
        )
    
    fig.show()

def plot_absolute_errors(exps_to_merge, estimatorsList, colors):
    regroupedErrors = dict.fromkeys(exps_to_merge)
    for expe in exps_to_merge:
        for estimator in estimatorsList:
            regroupedErrors[expe] = dict.fromkeys(estimatorsList)
            for estimator in regroupedErrors[expe].keys():
                regroupedErrors[expe][estimator] = {}
    for expe in regroupedErrors.keys():        
        for estimator in regroupedErrors[expe].keys():
            data = open_pickle(f"Projects/{expe}/output_data/evals/{estimator}/saved_results/traj_est/cached/cached_abs_err.pickle")
            for cat in data.keys():
                if isinstance(data[cat], np.ndarray):
                    if(data[cat].size == len(data[cat])):
                        regroupedErrors[expe][estimator][cat] = data[cat]
                    else:
                        regroupedErrors[expe][estimator][cat + "_0"] = data[cat][:,0]
                        regroupedErrors[expe][estimator][cat + "_1"] = data[cat][:,1]
                        regroupedErrors[expe][estimator][cat + "_2"] = data[cat][:,2]

    errorStats = dict.fromkeys(regroupedErrors.keys())
    for expe in regroupedErrors:
        errorStats[expe] = dict.fromkeys(regroupedErrors[expe].keys())
        
        for estimator in errorStats[expe]:
            errorStats[expe][estimator] = dict.fromkeys(regroupedErrors[expe][estimator].keys())
            for cat in errorStats[expe][estimator].keys():
                errorStats[expe][estimator][cat]  = {
                                        'rmse': 0.0, 'mean': 0.0, 'meanAbs': 0.0, 'median': 0.0, 'q1': 0.0, 'q3': 0.0, 
                                        'std': 0.0, 'min': 0.0, 'max': 0.0  }
                data_vec = regroupedErrors[expe][estimator][cat]
                errorStats[expe][estimator][cat]['rmse'] = float(np.sqrt(np.dot(data_vec, data_vec) / len(data_vec)))
                errorStats[expe][estimator][cat]['mean'] = float(np.mean(data_vec))
                errorStats[expe][estimator][cat]['meanAbs'] = float(np.mean(np.abs(data_vec)))
                errorStats[expe][estimator][cat]['median'] = float(np.median(data_vec))
                errorStats[expe][estimator][cat]['q1'] = float(np.quantile(data_vec, 0.25))
                errorStats[expe][estimator][cat]['q3'] = float(np.quantile(data_vec, 0.75))
                errorStats[expe][estimator][cat]['std'] = float(np.std(data_vec))
                errorStats[expe][estimator][cat]['min'] = float(np.min(data_vec))
                errorStats[expe][estimator][cat]['max'] = float(np.max(data_vec))

    plot_absolute_error_statistics_as_boxplot(errorStats, colors)

def plot_x_y_trajs(exps_to_merge, estimatorsList, colors):
    # For better plots, please check 'plotMultipleTrajs.py'
    fig = go.Figure()
    for expe in exps_to_merge:
        for estimator in estimatorsList:
            data = open_pickle(f"Projects/{expe}/output_data/evals/{estimator}/saved_results/traj_est/cached/x_y_z_traj.pickle")
            fig.add_trace(go.Scatter(x=data['x'], y=data['y'],
                    mode='lines',
                    name=f"{expe}_{estimator_plot_args[estimator]['name']}", line_color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)"))
    
        mocapData = open_pickle(f"Projects/{expe}/output_data/evals/mocap_x_y_z_traj.pickle")
        fig.add_trace(go.Scatter(x=mocapData['x'], y=mocapData['y'],
                        mode='lines',
                        name=f"{expe}_{estimator_plot_args['Ground truth']['name']}", line_color=f"rgba({int(colors['Ground truth'][0]*255)}, {int(colors['Ground truth'][1]*255)}, {int(colors['Ground truth'][2]*255)}, 1)"))
        
    fig.show()

def plot_x_y_z_trajs(exps_to_merge, estimatorsList, colors):
    fig = go.Figure()

    for expe in exps_to_merge:
        for estimator in estimatorsList:
            if estimator == "Controller": continue
            data = open_pickle(f"Projects/{expe}/output_data/evals/{estimator}/saved_results/traj_est/cached/x_y_z_traj.pickle")
            color_rgba = f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)"

            # Plot full trajectory
            fig.add_trace(go.Scatter3d(
                x=data['x'], y=data['y'], z=data['z'],
                mode='lines',
                name=f"{estimator_plot_args[estimator]['name']}",
                line=dict(width=estimator_plot_args[estimator]['lineWidth'], color=color_rgba)
            ))

            # Add Start and End markers
            fig.add_trace(go.Scatter3d(
                x=[data['x'][0], data['x'].iloc[-1]],
                y=[data['y'][0], data['y'].iloc[-1]],
                z=[data['z'][0], data['z'].iloc[-1]],
                mode='markers+text',
                text=['Start', 'End'],
                textposition='top center',
                textfont=dict(size=10, color=color_rgba, weight ='bold'),
                marker=dict(size=2, color=color_rgba),
                showlegend=False  # Don't duplicate in legend
            ))

        # Mocap trajectory
        mocapData = open_pickle(f"Projects/{expe}/output_data/evals/mocap_x_y_z_traj.pickle")
        color_rgba = f"rgba({int(colors['Ground truth'][0]*255)}, {int(colors['Ground truth'][1]*255)}, {int(colors['Ground truth'][2]*255)}, 1)"

        fig.add_trace(go.Scatter3d(
            x=mocapData['x'], y=mocapData['y'], z=mocapData['z'],
            mode='lines',
            name=f"{estimator_plot_args['Ground truth']['name']}",
            line=dict(color=color_rgba)
        ))

        fig.add_trace(go.Scatter3d(
            x=[mocapData['x'][0], mocapData['x'][-1]],
            y=[mocapData['y'][0], mocapData['y'][-1]],
            z=[mocapData['z'][0], mocapData['z'][-1]],
            mode='markers+text',
            text=['Start', 'End'],
            textposition='top center',
            textfont=dict(size=10, color=color_rgba,  weight ='bold'),
            marker=dict(size=2, color=color_rgba),
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Z Position (m)'
        ),
        title="3D Trajectories with Start/End Markers"
    )

    fig.update_layout(scene_camera=dict(
    eye=dict(x=1.8154531518195243, y=1.1734882040732764, z=0.12064530015476083 ),
    # eye=dict(x=2.029952817952063, y=0.7164110209288601, z=0.2314018279533368),
    center=dict(x=0, y=0, z=0),
    up=dict(x=0, y=0, z=1)
        ))

    fig.show()
    fig.write_image("/tmp/trajectory_plot.svg")


def plot_llve_statistics_as_boxplot(errorStats, colors, expe):
    fig = go.Figure()
    f = open('/tmp/velocity_errors.yaml', 'w+')
    yaml.dump(errorStats, f, allow_unicode=True, default_flow_style=False)
    all_categories = sorted({cat for estimator in errorStats.keys() for cat in errorStats[estimator].keys()})

    # Store traces per category for boxplots and velocity plots
    traces_to_plot = []

    # Load velocity data from pickle
    def load_velocity_data(estimator, category, isMocap=False):
        if isMocap == False:
            data = open_pickle(f"Projects/{expe}/output_data/evals/{estimator}/saved_results/traj_est/cached/loc_vel.pickle")
        else:
            data = open_pickle(f"Projects/{expe}/output_data/evals/mocap_loc_vel.pickle")
        return data[category]  # Assuming each category has velocity data by axis

    # Create traces for boxplots and velocity plots
    for estimator in errorStats.keys():
        for category in all_categories:
            if category not in errorStats[estimator]:
                continue
            x_vals = []  # Collect x values (d_subTraj)
            lower_fence = []
            q1 = []
            mean = []
            meanAbs = []
            median = []
            q3 = []
            upper_fence = []
            rmse = []

            # Collect the data across all sub-trajectory lengths for each estimator-category pair
            for axis in errorStats[estimator][category].keys():
                stats = errorStats[estimator][category][axis]
                
                x_vals.append(axis)
                lower_fence.append(stats['min'])
                q1.append(stats['q1'])
                mean.append(stats['mean'])
                meanAbs.append(stats['meanAbs'])
                median.append(stats['median'])
                q3.append(stats['q3'])
                upper_fence.append(stats['max'])
                rmse.append(stats['rmse'])

            # Create boxplot trace
            trace_boxplot = go.Box(
                x=x_vals,
                lowerfence=lower_fence,
                q1=q1,
                mean=mean,
                median=median,
                q3=q3,
                upperfence=upper_fence,
                name=f"{estimator_plot_args[estimator]['name']}",  # Only the estimator name in the legend
                boxpoints=False,
                marker_color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)",  # Outline color
                fillcolor=lighten_color(colors[estimator], 0.3),  # Slightly lighter and transparent background
                line=dict(width=estimator_plot_args[estimator]['lineWidth'], color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)"),
                opacity=0.8,
                visible=False  # Initially not visible
            )

            fig.add_trace(trace_boxplot)
            traces_to_plot.append({'type': 'boxplot', 'category': category})

            trace2 = go.Scatter(x=x_vals, y=rmse, name=f"{estimator_plot_args[estimator]['name']} rmse ({category})", marker_color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)", visible=False, mode='markers')

            fig.add_trace(trace2)
            traces_to_plot.append({'type': 'boxplot', 'category': category})

            # Create velocity plot traces
            velocity_data = load_velocity_data(estimator, category)
            for axis, values in velocity_data.items():
                trace_velocity = go.Scatter(
                    x=list(range(len(values))),  # Assuming the values are ordered by time
                    y=values,
                    mode='lines',
                    name=f"{estimator_plot_args[estimator]['name']} ({axis})",  # Include the axis name in the legend
                    line=dict(width=estimator_plot_args[estimator]['lineWidth'], color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)"),
                    visible=False  # Initially not visible
                )

                fig.add_trace(trace_velocity)
                traces_to_plot.append({'type': 'scatter', 'category': category})

    for category in all_categories:
        velocity_data = load_velocity_data('MoCap', category, True)
        for axis, values in velocity_data.items():
            trace_velocity = go.Scatter(
                x=list(range(len(values))),  # Assuming the values are ordered by time
                y=values,
                mode='lines',
                name=f"{estimator_plot_args['Ground truth']['name']} ({axis})",  # Include the axis name in the legend
                line=dict(width=estimator_plot_args['Ground truth']['lineWidth'], color=f"rgba({int(colors['Ground truth'][0]*255)}, {int(colors['Ground truth'][1]*255)}, {int(colors['Ground truth'][2]*255)}, 1)"),
                visible=False  # Initially not visible
            )

            fig.add_trace(trace_velocity)
            traces_to_plot.append({'type': 'scatter', 'category': category})

            

    # Update layout with buttons
    buttonsBoxplots = []
    buttonsVel = []
    for category in all_categories:
        buttonsBoxplots.append(dict(
            label=f"Show {category} Boxplots",
            method="update",
            args=[{"visible":  [trace['category'] == category and trace['type'] == 'boxplot' for trace in traces_to_plot] },  # Show boxplots for selected category
                   {"title": f"{category} Boxplots"}]
        ))
        buttonsVel.append(dict(
            label=f"Show {category} Velocities",
            method="update",
            args=[{"visible": [trace['category'] == category and trace['type'] == 'scatter' for trace in traces_to_plot] },  # Show velocities for selected category
                   {"title": f"{category} Velocities"}]
        ))

    fig.update_layout(
        title='Select Plot Type and Categories',
        xaxis_title='Sub-trajectory length [m]',
        updatemenus=[
            dict(
                type="buttons",
                buttons= buttonsBoxplots + buttonsVel
        )],
        boxmode='group'
    )

    fig.show()


def plot_llve(exps_to_merge, estimatorsList, colors):    
    regroupedErrors = dict.fromkeys(estimatorsList)
    for estimator in estimatorsList:
        data = open_pickle(f"Projects/{exps_to_merge[0]}/output_data/evals/{estimator}/saved_results/traj_est/cached/loc_vel.pickle")
        regroupedErrors[estimator] = dict.fromkeys(data.keys())
        for cat in data.keys():
            regroupedErrors[estimator][cat] = dict.fromkeys(data[cat].keys())

    for expe in exps_to_merge:
        mocapData = open_pickle(f"Projects/{expe}/output_data/evals/mocap_loc_vel.pickle")
        for estimator in estimatorsList:
            data = open_pickle(f"Projects/{expe}/output_data/evals/{estimator}/saved_results/traj_est/cached/loc_vel.pickle")
            
            for cat in data.keys():
                for axis in data[cat].keys():
                    data[cat][axis] = data[cat][axis] - mocapData[cat][axis]    
                    if regroupedErrors[estimator][cat][axis] is None:
                        regroupedErrors[estimator][cat][axis] = data[cat][axis]
                    else:
                        regroupedErrors[estimator][cat][axis] = np.concatenate((regroupedErrors[estimator][cat][axis], data[cat][axis]))
    
    
    # print(data.keys())
    for estimator in estimatorsList:
        for cat in regroupedErrors[estimator].keys():
            
            for axis in regroupedErrors[estimator][cat].keys():
                regroupedErrors[estimator][cat][axis] = np.abs(regroupedErrors[estimator][cat][axis])

            # Combine x, y, z components into a single array
            components = ['x', 'y', 'z']  # Adjust keys as needed
            combined = np.stack([regroupedErrors[estimator][cat][comp] for comp in components], axis=-1)

            # Compute the Euclidean norm for each sample
            regroupedErrors[estimator][cat]["norm"] = np.linalg.norm(combined, axis=-1)
            
            components = ['x', 'y']  # Adjust keys as needed
            combined = np.stack([regroupedErrors[estimator][cat][comp] for comp in components], axis=-1)

            # Compute the Euclidean norm for each sample
            regroupedErrors[estimator][cat]["velXY_norm"] = np.linalg.norm(combined, axis=-1)

   
    errorStats = dict.fromkeys(regroupedErrors.keys())
    for estimator in estimatorsList:
        errorStats[estimator] = dict.fromkeys(regroupedErrors[estimator].keys())
        for cat in errorStats[estimator].keys():
            errorStats[estimator][cat] = dict.fromkeys(regroupedErrors[estimator][cat].keys())
            for axis in errorStats[estimator][cat].keys():
                errorStats[estimator][cat][axis]  = {
                                        'rmse': 0.0, 'mean': 0.0, 'meanAbs': 0.0, 'median': 0.0, 'q1': 0.0, 'q3': 0.0, 
                                        'std': 0.0, 'min': 0.0, 'max': 0.0  }
                data_vec = regroupedErrors[estimator][cat][axis]

                errorStats[estimator][cat][axis]['rmse'] = float(np.sqrt(np.dot(data_vec, data_vec) / len(data_vec)))
                errorStats[estimator][cat][axis]['mean'] = float(np.mean(np.abs(data_vec)))
                errorStats[estimator][cat][axis]['meanAbs'] = float(np.mean(np.abs(data_vec)))
                errorStats[estimator][cat][axis]['median'] = float(np.median(data_vec))
                errorStats[estimator][cat][axis]['q1'] = float(np.quantile(data_vec, 0.25))
                errorStats[estimator][cat][axis]['q3'] = float(np.quantile(data_vec, 0.75))
                errorStats[estimator][cat][axis]['std'] = float(np.std(data_vec))
                errorStats[estimator][cat][axis]['min'] = float(np.min(data_vec))
                errorStats[estimator][cat][axis]['max'] = float(np.max(data_vec))


    plot_llve_statistics_as_boxplot(errorStats, colors, exps_to_merge[0])
    

def plot_errors_per_walk_cycle_as_boxplot(errorStats, colors):    
    f = open('/tmp/errors_per_walk_cycle.yaml', 'w+')
    yaml.dump(errorStats, f, allow_unicode=True, default_flow_style=False)
    fig = go.Figure()

    all_categories = sorted({cat for distance in errorStats.values() for estimator in distance.values() for cat in estimator.keys()})
    
    dropdown_buttons = []
    
    traces_per_category = {category: [] for category in all_categories}

    # Iterate over categories and create a trace per estimator-category pair
    first_key = next(iter(errorStats))
    for estimator in errorStats[first_key].keys():  # We assume the same estimators are present for all d_subTraj
        for category in all_categories:
            x_vals = []  # Collect x values (d_subTraj)
            lower_fence = []
            q1 = []
            mean = []
            meanAbs = []
            median = []
            q3 = []
            upper_fence = []
            rmse = []

            # Collect the data across all sub-trajectory lengths for each estimator-category pair
            for d_subTraj in errorStats.keys():
                stats = errorStats[d_subTraj][estimator][category]
                x_vals.append(d_subTraj)
                lower_fence.append(stats['min'])
                q1.append(stats['q1'])
                mean.append(stats['mean'])
                meanAbs.append(stats['meanAbs'])
                median.append(stats['median'])
                q3.append(stats['q3'])
                upper_fence.append(stats['max'])
                rmse.append(stats['rmse'])


            trace = go.Box(
                x=x_vals,  # X-axis corresponds to sub-trajectory lengths
                lowerfence=lower_fence,
                q1=q1,
                mean=mean,
                median=median,
                q3=q3,
                upperfence=upper_fence,
                name=f"{estimator_plot_args[estimator]['name']} ({category})",  # Single legend entry per estimator-category pair
                boxpoints=False,
                marker_color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)",  # Outline color
                fillcolor=lighten_color(colors[estimator], 0.3),  # Slightly lighter and transparent background
                line=dict(width=estimator_plot_args[estimator]['lineWidth'], color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)"),  # Well-visible outline
                opacity=0.8,
                visible=False  # Initially hidden
            )

            trace2 = go.Scatter(x=x_vals, y=rmse, name=f"{estimator_plot_args[estimator]['name']} rmse ({category})", marker_color=f"rgba({int(colors[estimator][0]*255)}, {int(colors[estimator][1]*255)}, {int(colors[estimator][2]*255)}, 1)", visible=False, mode='markers')


            traces_per_category[category].append(trace)
            fig.add_trace(trace)

            traces_per_category[category].append(trace2)
            fig.add_trace(trace2)

    # Create dropdown buttons to toggle visibility for each category
    for category in all_categories:
        visibility = [False] * len(fig.data)  # Start with all traces hidden
        for i, trace in enumerate(fig.data):
            if trace.name.endswith(f"({category})"):
                visibility[i] = True  # Show traces of the selected category

        button = dict(
            label=category,
            method='update',
            args=[{'visible': visibility}, {'title': rel_category_titles.get(category, 'Default Label'), 'yaxis.title': rel_category_ylabels.get(category, 'Default Label')}]
        )
        dropdown_buttons.append(button)

    fig.update_layout(
        title='Select a Category to Display Boxplots',
        xaxis_title='Sub-trajectory length [m]',
        updatemenus=[
            {
                'buttons': dropdown_buttons,
                'direction': 'down',
                'showactive': True,
            }
        ],
        boxmode='group'  # Group boxes together
    )

    fig.show()


def plot_errors_per_walk_cycle(exps_to_merge, estimatorsList, colors):
    first = open_pickle(f"Projects/{exps_to_merge[0]}/output_data/evals/error_walk_cycle.pickle")
    valid_cycle_lengths = [k for k, v in first.items() if isinstance(v, dict)]

    regroupedErrors = {
        cycleLength: {
            estimator: {"pos_lateral": [], "pos_z": [], "tilt": [], "yaw": []}
            for estimator in estimatorsList
        }
        for cycleLength in valid_cycle_lengths
    }

    for expe in exps_to_merge:
        errorsByCycleLength_expe = open_pickle(f"Projects/{expe}/output_data/evals/error_walk_cycle.pickle")
        for cycleLength in valid_cycle_lengths:
            block = errorsByCycleLength_expe.get(cycleLength)
            if not isinstance(block, dict):
                continue
            for estimator in estimatorsList:
                est_block = block.get(estimator)
                if not isinstance(est_block, dict):
                    continue
                pos_list = est_block.get("pos", [])
                for pos in pos_list:
                    lateral = float(np.linalg.norm(pos[0:2]))
                    z_abs = float(np.abs(pos[2]))
                    regroupedErrors[cycleLength][estimator]["pos_lateral"].append(lateral)
                    regroupedErrors[cycleLength][estimator]["pos_z"].append(z_abs)
                regroupedErrors[cycleLength][estimator]["tilt"].extend(
                    [float(x) for x in est_block.get("tilt", [])]
                )
                regroupedErrors[cycleLength][estimator]["yaw"].extend(
                    [float(x) for x in est_block.get("yaw", [])]
                )

    errorStats = {cycleLength: {} for cycleLength in regroupedErrors.keys()}
    for cycleLength in errorStats:
        errorStats[cycleLength] = {estimator: {} for estimator in regroupedErrors[cycleLength].keys()}
        for estimator in errorStats[cycleLength]:
            for cat in regroupedErrors[cycleLength][estimator].keys():
                data_vec = np.asarray(regroupedErrors[cycleLength][estimator][cat], dtype=float)
                if data_vec.size == 0:
                    errorStats[cycleLength][estimator][cat] = {
                        'rmse': float('nan'),
                        'mean': float('nan'),
                        'meanAbs': float('nan'),
                        'median': float('nan'),
                        'q1': float('nan'),
                        'q3': float('nan'),
                        'std': float('nan'),
                        'min': float('nan'),
                        'max': float('nan')
                    }
                else:
                    errorStats[cycleLength][estimator][cat] = {
                        'rmse': float(np.sqrt(np.dot(data_vec, data_vec) / data_vec.size)),
                        'mean': float(np.mean(data_vec)),
                        'meanAbs': float(np.mean(np.abs(data_vec))),
                        'median': float(np.median(data_vec)),
                        'q1': float(np.quantile(data_vec, 0.25)),
                        'q3': float(np.quantile(data_vec, 0.75)),
                        'std': float(np.std(data_vec)),
                        'min': float(np.min(data_vec)),
                        'max': float(np.max(data_vec))
                    }

    plot_errors_per_walk_cycle_as_boxplot(errorStats, colors)


def main():
    parser = argparse.ArgumentParser()
    
    # Array argument: list of sublengths
    parser.add_argument('--exps_to_merge', nargs='+', help='List of folders whose we want to merge the computed errors', required=True)
    args = parser.parse_args()

    nb_estimators=0
    exps_to_merge = args.exps_to_merge
    for expe in exps_to_merge:
        if(len(next(os.walk(f"Projects/{expe}/output_data/evals/"))[1]) == 0 or (nb_estimators != 0 and nb_estimators!= len(next(os.walk(f"Projects/{expe}/output_data/evals/"))[1]))):
           sys.exit("The experiments don't contain results from the same estimators, or don't contain any results.") 
        nb_estimators = len(next(os.walk(f"Projects/{expe}/output_data/evals/"))[1])

    estimatorsList = [d for d in os.listdir(f"Projects/{exps_to_merge[0]}/output_data/evals/") 
                  if os.path.isdir(f"Projects/{exps_to_merge[0]}/output_data/evals/{d}")]

    # if "KineticsObserver" in estimatorsList:
    #     estimatorsList.insert(0, estimatorsList.pop(estimatorsList.index("KineticsObserver")))


    # estimatorsList = [e for e in estimatorsList if e in estimator_plot_args]
    estimatorsList = list(dict.fromkeys(
        e for e in estimators_to_plot
        if e in estimatorsList and e in estimator_plot_args
    ))

    estimatorsForErrors = estimatorsList.copy()
    

    if(estimatorsList.count("Mocap") == 0):
        estimatorsList.append("Mocap")

    colors = generate_turbo_subset_colors(estimatorsList)

    # plot_llve(exps_to_merge, estimatorsList, colors)
    
    # plot_absolute_errors_raw(exps_to_merge, estimatorsList, colors)
    
    # plot_absolute_errors(exps_to_merge, estimatorsList, colors)

    plot_relative_errors(exps_to_merge, estimatorsForErrors, colors)

    plot_errors_per_walk_cycle(exps_to_merge, estimatorsForErrors, colors)

    # plot_x_y_trajs(exps_to_merge, estimatorsList, colors)
    # plot_x_y_z_trajs(exps_to_merge, estimatorsList, colors)

    import plotMultipleTrajs
    
    colors_to_plot = colors

    # plotMultipleTrajs.plot_relative_trajs_video_distance(estimatorsList, exps_to_merge, colors_to_plot, estimator_plot_args, 'Projects/')
    # plotMultipleTrajs.plot_relative_trajs_video_temporal(estimatorsList, exps_to_merge, colors_to_plot, estimator_plot_args, 'Projects/')
    # plotMultipleTrajs.plot_multiple_trajs_video(estimatorsList, exps_to_merge, colors_to_plot, estimator_plot_args, 'Projects/')
    #plotMultipleTrajs.generate_video_from_trajs(estimatorsList, exps_to_merge, colors_to_plot, estimator_plot_args, 'Projects/', main_expe=0, fps=20, video_output="output_video.mp4")

    #import plotExternalForceAndBias
    #plotExternalForceAndBias.computeExtWrenchError(exps_to_merge)
    
    if(len(exps_to_merge) == 1):
        # import plotPoseAndVelocity
        # plotPoseAndVelocity.plotPoseVel(estimatorsList, f'Projects/{exps_to_merge[0]}', colors_to_plot, estimator_plot_args)

    #if(len(exps_to_merge) == 1):
        import plotContactPoses
        #plotContactPoses.plotContactPoses(estimatorsList, colors_to_plot, f'Projects/{exps_to_merge[0]}')
        #plotContactPoses.plotContactRestPoses(colors_to_plot, f'Projects/{exps_to_merge[0]}')
        
    if(len(exps_to_merge) == 1):
        import plotExternalForceAndBias
        #plotExternalForceAndBias.plotGyroBias(colors_to_plot, f'Projects/{exps_to_merge[0]}')
        #plotExternalForceAndBias.plotExtWrench(colors_to_plot, f'Projects/{exps_to_merge[0]}')
    
    import plotAndFormatResults
    # for expe in exps_to_merge:
        # plotAndFormatResults.run(True, False, f"Projects/{expe}", estimatorsList, colors_to_plot)
    
    
    
if __name__ == '__main__':
    rel_category_titles = {
        'rel_gravity': 'Relative Error on the gravity estimate',
        'rel_rot': 'Relative Error on the orientation estimate',
        'rel_rot_deg_per_m': 'Relative Error on the orientation estimate (per meter travelled)',
        'rel_trans': 'Relative Error on the translation estimate',
        'rel_trans_perc': 'Relative Error on the translation estimate (in percentage of the travelled distance)',
        'rel_yaw': 'Relative Error on the yaw estimate'
    }

    rel_category_ylabels = {
        'rel_gravity': 'Gravity',
        'rel_rot': 'Rotation [deg]',
        'rel_rot_deg_per_m': 'Rotation [deg/meter]',
        'rel_trans': 'Translation error [m]',
        'rel_trans_perc': 'Translation [%]',
        'rel_yaw': 'Yaw error [deg]'
    }

    abs_category_titles = {
    'abs_e_rot': 'Absolute Error on the rotation estimate',
    'abs_e_scale_perc': 'Absolute Error on the scale estimate (percentage)',
    'abs_e_trans': 'Absolute Error on the translation estimate',
    'abs_e_trans_vec_0': 'Absolute Error on the translation estimate along x',
    'abs_e_trans_vec_1': 'Absolute Error on the translation estimate along y',
    'abs_e_trans_vec_2': 'Absolute Error on the translation estimate along z',
    'abs_e_ypr_0': 'Absolute Error on the rotation estimate in yaw',
    'abs_e_ypr_1': 'Absolute Error on the rotation estimate in pitch',
    'abs_e_ypr_2': 'Absolute Error on the rotation estimate in roll',
    'accum_distances': 'Total distance'
    }

    abs_category_ylabels = {
        'abs_e_rot': 'Rotation error [deg]',
        'abs_e_scale_perc': 'Scale error',
        'abs_e_trans': 'Translation error [m]',
        'abs_e_trans_vec_0': 'Translation error along x [m]',
        'abs_e_trans_vec_1': 'Translation error along y [m]',
        'abs_e_trans_vec_2': 'Translation error along z [m]',
        'abs_e_ypr_0': 'Yaw error [deg]',
        'abs_e_ypr_1': 'Pitch error [deg]',
        'abs_e_ypr_2': 'Roll error [deg]',
        'accum_distances': 'Total distance [m]'
    }

    main()