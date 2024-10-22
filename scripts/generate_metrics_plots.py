import os
import yaml
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

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


def load_relative_error_data(directory):
    all_data = {}
    colors = generate_turbo_subset_colors(len(next(os.walk(directory))[1]))

    for index, estimator in enumerate(os.listdir(directory)):
        estimator_path = os.path.join(directory, estimator, "saved_results", "traj_est")
        if os.path.isdir(estimator_path):
            all_data[estimator] = {}
            for filename in os.listdir(estimator_path):
                if filename.startswith("relative_error_statistics") and filename.endswith(".yaml"):
                    length_str = filename.split('_')[-2] + '.' + filename.split('_')[-1].replace('.yaml', '')
                    sub_trajectory_length = float(length_str)

                    with open(os.path.join(estimator_path, filename), 'r') as file:
                        relative_error_data = yaml.safe_load(file)

                    for category, stats in relative_error_data.items():
                        if category not in all_data[estimator]:
                            all_data[estimator][category] = {
                                'lengths': [], 'means': [], 'medians': [], 'q1': [], 'q3': [], 
                                'stds': [], 'mins': [], 'maxs': [], 'color': ''
                            }
                        all_data[estimator][category]['lengths'].append(sub_trajectory_length)
                        all_data[estimator][category]['means'].append(stats['mean'])
                        all_data[estimator][category]['medians'].append(stats['median'])
                        all_data[estimator][category]['q1'].append(stats['q1'])
                        all_data[estimator][category]['q3'].append(stats['q3'])
                        all_data[estimator][category]['stds'].append(stats['std'])
                        all_data[estimator][category]['mins'].append(stats['min'])
                        all_data[estimator][category]['maxs'].append(stats['max'])
                        all_data[estimator][category]['color'] = colors[index]

            # Sort the statistics by lengths
            for category in all_data[estimator]:
                sorted_indices = np.argsort(all_data[estimator][category]['lengths'])
                all_data[estimator][category] = {
                    'lengths': np.array(all_data[estimator][category]['lengths'])[sorted_indices].tolist(),
                    'means': np.array(all_data[estimator][category]['means'])[sorted_indices].tolist(),
                    'medians': np.array(all_data[estimator][category]['medians'])[sorted_indices].tolist(),
                    'q1': np.array(all_data[estimator][category]['q1'])[sorted_indices].tolist(),
                    'q3': np.array(all_data[estimator][category]['q3'])[sorted_indices].tolist(),
                    'stds': np.array(all_data[estimator][category]['stds'])[sorted_indices].tolist(),
                    'mins': np.array(all_data[estimator][category]['mins'])[sorted_indices].tolist(),
                    'maxs': np.array(all_data[estimator][category]['maxs'])[sorted_indices].tolist(),
                    'color': all_data[estimator][category]['color']  # Keep the color unchanged
                }

    return all_data

def plot_relative_error_statistics_as_boxplot(all_data):    
    fig = go.Figure()
    
    all_categories = sorted({cat for est_data in all_data.values() for cat in est_data.keys()})
    
    dropdown_buttons = []
    
    traces_per_category = {category: [] for category in all_categories}
    
    for estimator in all_data.keys():
        
        for category in all_categories:
            if category in all_data[estimator]:
                stats = all_data[estimator][category]
                trace = go.Box(
                    x=stats['lengths'],
                    lowerfence=stats['mins'],
                    q1=stats['q1'],
                    mean=stats['means'],
                    median=stats['medians'],
                    q3=stats['q3'],
                    upperfence=stats['maxs'],
                    name=f"{estimator} ({category})",
                    boxpoints=False,
                    marker_color=f'rgba({int(stats["color"][0]*255)}, {int(stats["color"][1]*255)}, {int(stats["color"][2]*255)}, 1)',  # Outline color
                    fillcolor=lighten_color(stats['color'], 0.3),  # Slightly lighter and transparent background
                    line=dict(width=2, color=f'rgba({int(stats["color"][0]*255)}, {int(stats["color"][1]*255)}, {int(stats["color"][2]*255)}, 1)'),  # Well-visible outline
                    opacity=0.8,
                    visible=False  # Initially hidden
                )
                traces_per_category[category].append(trace)
                fig.add_trace(trace)

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
        xaxis_title= 'Sub-trajectory length [m]',
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

# Specify the directory containing the evaluator folders
directory = "../Projects/HRP5_MultiContact_1/output_data/evals/"

# Load the data from YAML files
all_relative_error_data = load_relative_error_data(directory)

# Plot the relative error statistics
plot_relative_error_statistics_as_boxplot(all_relative_error_data)
