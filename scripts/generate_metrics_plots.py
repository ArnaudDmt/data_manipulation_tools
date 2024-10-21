import os
import yaml
import plotly.graph_objects as go

# fig = go.Figure()

# fig.add_trace(go.Box(q1=[ 1, 2, 3 ], median=[ 4, 5, 6 ],
#                   q3=[ 7, 8, 9 ], lowerfence=[-1, 0, 1],
#                   upperfence=[7, 8, 9], mean=[ 2.2, 2.8, 3.2 ],
#                   sd=[ 0.2, 0.4, 0.6 ], notchspan=[ 0.2, 0.4, 0.6 ], name="Precompiled Quartiles"))
# fig.show()
# exit(1)
def load_relative_error_data(directory):
    # Dictionary to store data for each estimator and sub-trajectory length
    data = {}

    # Iterate over all YAML files in the directory
    for filename in os.listdir(directory):
        if filename.startswith("relative_error_statistics") and filename.endswith(".yaml"):
            # Correctly extract the sub-trajectory length (e.g., "3.8" for 3_8 meters)
            length_str = filename.split('_')[-2] + '.' + filename.split('_')[-1].replace('.yaml', '')
            sub_trajectory_length = float(length_str)

            # Load the YAML file content
            with open(os.path.join(directory, filename), 'r') as file:
                relative_error_data = yaml.safe_load(file)

            # Store the data for each error category
            for category, stats in relative_error_data.items():
                if category not in data:
                    data[category] = {
                        'lengths': [], 'means': [], 'medians': [], 'q1': [], 'q3': [],
                        'stds': [], 'mins': [], 'maxs': []
                    }
                data[category]['lengths'].append(sub_trajectory_length)
                data[category]['means'].append(stats['mean'])
                data[category]['medians'].append(stats['median'])
                data[category]['q1'].append(stats['q1'])
                data[category]['q3'].append(stats['q3'])
                data[category]['stds'].append(stats['std'])
                data[category]['mins'].append(stats['min'])
                data[category]['maxs'].append(stats['max'])

    return data

def plot_relative_error_statistics_as_boxplot(data):
    fig = go.Figure()

    # Create a boxplot for each error category
    for category, stats in data.items():
        # Approximate q1 and q3 using mean ± std as proxies
        median = stats['medians']
        q1 = stats['q1']
        q3 = stats['q3']
        min_values = stats['mins']
        max_values = stats['maxs']
        lengths = stats['lengths']

        # Add the box plot trace
        fig.add_trace(go.Box(
            x=lengths,
            lowerfence=min_values,
            q1=q1,
            median=median,
            q3=q3,
            upperfence=max_values,
            name=category,
            boxpoints=False,  # No individual data points displayed
            line=dict(width=1),
            fillcolor='lightgray'
        ))

    # Update layout
    fig.update_layout(
        title='Relative Error Statistics Across Sub-Trajectory Lengths',
        xaxis_title='Sub-Trajectory Length (m)',
        yaxis_title='Error',
        showlegend=True
    )

    # Show the figure
    fig.show()


# Specify the directory containing the YAML files
directory = "../Projects/HRP5_MultiContact_1/output_data/evals/KineticsObserver/saved_results/traj_est/"

# Load the data from YAML files
relative_error_data = load_relative_error_data(directory)

# Plot the relative error statistics
plot_relative_error_statistics_as_boxplot(relative_error_data)