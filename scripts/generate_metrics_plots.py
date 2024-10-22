#!./env/bin/python

import os
import sys
import yaml
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


import pandas as pd
import pickle


import argparse

def reduce_intensity(color, amount=0.7):
    """Blend the color with gray to reduce intensity."""
    r, g, b, _ = color
    gray = 0.5  # Gray defined as (0.5, 0.5, 0.5) in RGB
    r = (1 - amount) * gray + amount * r
    g = (1 - amount) * gray + amount * g
    b = (1 - amount) * gray + amount * b
    return (r, g, b, 1)  # Return as (R, G, B, Alpha)

def generate_turbo_subset_colors(num_colors):
    cmap = plt.get_cmap('turbo')
    # Generate colors and reduce intensity
    colors = [reduce_intensity(cmap(i), 0.6) for i in np.linspace(0.2, 0.8, num_colors)]
    return colors

def lighten_color(color, amount=0.5):
    r, g, b, _ = color
    return f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {amount})'

def open_pickle(pickle_file, csv_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
        return data


def plot_relative_error_statistics_as_boxplot(errorStats):    
    colors = generate_turbo_subset_colors(len(errorStats[0.1].keys()))
    fig = go.Figure()

    all_categories = sorted({cat for distance in errorStats.values() for estimator in distance.values() for cat in estimator.keys()})
    
    dropdown_buttons = []
    
    traces_per_category = {category: [] for category in all_categories}

    # Iterate over categories and create a trace per estimator-category pair
    for i, estimator in enumerate(errorStats[0.1].keys()):  # We assume the same estimators are present for all d_subTraj
        for category in all_categories:
            x_vals = []  # Collect x values (d_subTraj)
            lower_fence = []
            q1 = []
            mean = []
            median = []
            q3 = []
            upper_fence = []

            # Collect the data across all sub-trajectory lengths for each estimator-category pair
            for d_subTraj in errorStats.keys():
                stats = errorStats[d_subTraj][estimator][category]
                x_vals.append(d_subTraj)
                lower_fence.append(stats['min'])
                q1.append(stats['q1'])
                mean.append(stats['mean'])
                median.append(stats['median'])
                q3.append(stats['q3'])
                upper_fence.append(stats['max'])

            trace = go.Box(
                x=x_vals,  # X-axis corresponds to sub-trajectory lengths
                lowerfence=lower_fence,
                q1=q1,
                mean=mean,
                median=median,
                q3=q3,
                upperfence=upper_fence,
                name=f"{estimator} ({category})",  # Single legend entry per estimator-category pair
                boxpoints=False,
                marker_color=f'rgba({int(colors[i][0]*255)}, {int(colors[i][1]*255)}, {int(colors[i][2]*255)}, 1)',  # Outline color
                fillcolor=lighten_color(colors[i], 0.3),  # Slightly lighter and transparent background
                line=dict(width=2, color=f'rgba({int(colors[i][0]*255)}, {int(colors[i][1]*255)}, {int(colors[i][2]*255)}, 1)'),  # Well-visible outline
                opacity=0.8,
                visible=False  # Initially hidden
            )
            traces_per_category[category].append(trace)
            fig.add_trace(trace)

    # Create dropdown buttons to toggle visibility for each category
    for category in all_categories:
        visibility = [False] * len(fig.data)  # Start with all traces hidden
        for i, trace in enumerate(fig.data):
            if trace.name.endswith(f"({category})"):
                visibility[i] = True  # Show traces of the selected category

        button = dict(
            label=category,
            method='update',
            args=[{'visible': visibility}, {'title': category_titles.get(category, 'Default Label'), 'yaxis.title': category_ylabels.get(category, 'Default Label')}]
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

    estimatorsList = os.listdir(f"Projects/{exps_to_merge[0]}/output_data/evals/") 

    data1 = open_pickle(f'Projects/{exps_to_merge[0]}/output_data/evals/{estimatorsList[0]}/saved_results/traj_est/cached/cached_rel_err.pickle', '/tmp/output.csv')

    regroupedErrors = dict.fromkeys(data1.keys())

    for d_subTraj in regroupedErrors.keys():
        regroupedErrors[d_subTraj] = dict.fromkeys(estimatorsList)
        for estimator in estimatorsList:
            cats = list(data1[list(data1.keys())[0]].keys())
            regroupedErrors[d_subTraj][estimator] = dict.fromkeys(cats)

    for expe in exps_to_merge:        
        for estimator in estimatorsList:
            data = open_pickle(f'Projects/{expe}/output_data/evals/{estimator}/saved_results/traj_est/cached/cached_rel_err.pickle', '/tmp/output.csv')
            for d_subTraj in regroupedErrors.keys():
                for cat in data[d_subTraj]:
                    if isinstance(data[d_subTraj][cat], np.ndarray):
                        if regroupedErrors[d_subTraj][estimator][cat] is None:
                            regroupedErrors[d_subTraj][estimator][cat] = data[d_subTraj][cat]
                        else:
                            regroupedErrors[d_subTraj][estimator][cat] = np.concatenate((regroupedErrors[d_subTraj][estimator][cat], data[d_subTraj][cat]))

    errorStats = dict.fromkeys(regroupedErrors.keys())
    for d_subTraj in regroupedErrors:
        errorStats[d_subTraj] = dict.fromkeys(estimatorsList)
        for cat in cats:
            if "stats" in cat:
                cats.remove(cat)
        for estimator in errorStats[d_subTraj]:
            errorStats[d_subTraj][estimator] = dict.fromkeys(cats)
            for cat in cats:
                errorStats[d_subTraj][estimator][cat]  = {
                                        'rmse': 0.0, 'mean': 0.0, 'median': 0.0, 'q1': 0.0, 'q3': 0.0, 
                                        'std': 0.0, 'min': 0.0, 'max': 0.0  }
                data_vec = regroupedErrors[d_subTraj][estimator][cat]
                errorStats[d_subTraj][estimator][cat]['rmse'] = float(np.sqrt(np.dot(data_vec, data_vec) / len(data_vec)))
                errorStats[d_subTraj][estimator][cat]['mean'] = float(np.mean(data_vec))
                errorStats[d_subTraj][estimator][cat]['median'] = float(np.median(data_vec))
                errorStats[d_subTraj][estimator][cat]['q1'] = float(np.quantile(data_vec, 0.25))
                errorStats[d_subTraj][estimator][cat]['q3'] = float(np.quantile(data_vec, 0.75))
                errorStats[d_subTraj][estimator][cat]['std'] = float(np.std(data_vec))
                errorStats[d_subTraj][estimator][cat]['min'] = float(np.min(data_vec))
                errorStats[d_subTraj][estimator][cat]['max'] = float(np.max(data_vec))

    plot_relative_error_statistics_as_boxplot(errorStats)
    
if __name__ == '__main__':
    category_titles = {
    'gravity': 'Relative Error on the gravity estimate',
    'rot': 'Relative Error on the orientation estimate',
    'rot_deg_per_m': 'Relative Error on the orientation estimate (per meter travelled)',
    'trans': 'Relative Error on the translation estimate',
    'trans_perc': 'Relative Error on the translation estimate (in percentage of the travelled distance)',
    'yaw': 'Relative Error on the yaw estimate'
    }

    category_ylabels = {
        'gravity': 'Gravity',
        'rot': 'Rotation [deg]',
        'rot_deg_per_m': 'Rotation [deg/meter]',
        'trans': 'Translation error [m]',
        'trans_perc': 'Translation [%]',
        'yaw': 'Yaw error [deg]'
    }

    main()