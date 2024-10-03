import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px  # For color palette generation

def plot_average(csv_files, columns):
    # Initialize a list to store the data from each CSV
    dfs = []

    # Read each CSV into a DataFrame and append to the list, using ';' as the delimiter
    for file in csv_files:
        df = pd.read_csv(file, sep=';')
        dfs.append(df)

    # Ensure that the 't' column exists in all files
    if 't' not in dfs[0].columns:
        raise ValueError("Column 't' must exist in all CSV files")

    # Initialize a DataFrame for averages, using the 't' column as the index
    avg_df = pd.DataFrame()
    avg_df['t'] = dfs[0]['t']

    # Generate a color palette for the number of columns to average
    colors = px.colors.qualitative.Plotly[:len(columns)]

    # Create a Plotly figure
    fig = go.Figure()

    # Plot each individual CSV file with lighter color and transparent lines
    for df in dfs:
        for i, col in enumerate(columns):
            rgb_color = px.colors.hex_to_rgb(colors[i])  # Convert hex to RGB
            transparent_color = f'rgba({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}, 0.2)'  # 20% opacity
            fig.add_trace(go.Scatter(x=df['t'], y=df[col],
                                     mode='lines',
                                     line=dict(color=transparent_color, width=2),
                                     showlegend=False))

    # Compute the average for each desired column
    for i, col in enumerate(columns):
        avg_df[col] = np.mean([df[col] for df in dfs], axis=0)

        # Plot the average with a thicker, darker line
        fig.add_trace(go.Scatter(x=avg_df['t'], y=avg_df[col],
                                 mode='lines',
                                 name=f'Average {col}',
                                 line=dict(color=colors[i], width=3)))

    # Update layout
    fig.update_layout(title="Average of Selected Columns Over Time",
                      xaxis_title="Time (t)",
                      yaxis_title="Values",
                      template="plotly_white")

    # Show the plot
    fig.show()

# Example usage
csv_files = ['../Projects/HRP5_MultiContact_1/output_data/observerResultsCSV.csv', 
             '../Projects/HRP5_MultiContact_2/output_data/observerResultsCSV.csv', 
             '../Projects/HRP5_MultiContact_3/output_data/observerResultsCSV.csv', 
             '../Projects/HRP5_MultiContact_4/output_data/observerResultsCSV.csv']  # list of CSV file paths
columns = ['Observers_MainObserverPipeline_MCKineticsObserver_mcko_fb_posW_tx', 'Mocap_pos_x']  # list of columns to average and plot
plot_average(csv_files, columns)
