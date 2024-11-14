import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  # For color palette generation





# default_exps = [
#     '../HRP5_MultiContact_1', 
#     '../HRP5_MultiContact_2, 
#     '../HRP5_MultiContact_3', 
#     '../HRP5_MultiContact_4'
# ]

default_exps = [
    '../KO_TRO_2024_RHPS1_SLIPPAGE_1', 
    '../KO_TRO_2024_RHPS1_SLIPPAGE_2', 
    '../KO_TRO_2024_RHPS1_SLIPPAGE_3'
]

# default_exps = [
#     '../KO_TRO2024_RHPS1_1', 
#     '../KO_TRO2024_RHPS1_2', 
#     '../KO_TRO2024_RHPS1_3', 
#     '../KO_TRO2024_RHPS1_4', 
#     '../KO_TRO2024_RHPS1_5'
# ]

default_estimators = [
    'KineticsObserver', 
    'Mocap'
]


# Define columns for each estimator
data = {
    'Controller': ['Controller_tx', 'Controller_ty'],
    'Vanyte': ['Vanyte_pose_tx', 'Vanyte_pose_ty'],
    'Hartley': ['Hartley_Position_x', 'Hartley_Position_y'],
    'KineticsObserver': ['KO_posW_tx', 'KO_posW_ty'],
    'KO_APC': ['KO_APC_posW_tx', 'KO_APC_posW_ty'],
    'KO_ASC': ['KO_ASC_posW_tx', 'KO_ASC_posW_ty'],
    'KO_Disabled': ['KO_Disabled_posW_tx', 'KO_Disabled_posW_ty'],
    'Mocap': ['Mocap_pos_x', 'Mocap_pos_y']
}



def plot_multiple_trajs(estimators, exps, colors):
    xys = dict.fromkeys(estimators)

    all_columns = []
    for estimator in estimators:
        xys[estimator] = dict.fromkeys(range(len(exps)))
        for k in range(len(exps)):
            xys[estimator][k] = {0: [], 1:[]}
        for col in data[estimator]:
            all_columns.append(col)
    all_columns.append("Mocap_datasOverlapping")

    for e, exp in enumerate(exps):
        file = f'Projects/{exp}/output_data/observerResultsCSV.csv'
        df = pd.read_csv(file, sep=';', usecols=all_columns)
        df_overlap = df[df["Mocap_datasOverlapping"] == "Datas overlap"]
        for estimator in estimators:
            xys[estimator][e][0] = df_overlap[data[estimator][0]]
            xys[estimator][e][1] = df_overlap[data[estimator][1]]
                

    # Create a Plotly figure
    fig = go.Figure()

    for estimator in estimators:
        xs = []
        ys = []
        color = colors[estimator]

        # Process each CSV for the current estimator
        for e in xys[estimator].keys():
            xs.append(xys[estimator][e][0].values)
            ys.append(xys[estimator][e][1].values)
            
            if(e == 0):
                if(estimator == 'Mocap'):
                    transparent_color = f'rgba({color[0]}, {color[1]}, {color[2]}, 1)'
                else:
                    transparent_color = f'rgba({color[0]}, {color[1]}, {color[2]}, 0.7)'
            else:
                if(estimator == 'Mocap'):
                    transparent_color = f'rgba({color[0]}, {color[1]}, {color[2]}, 0.25)'
                else:
                    transparent_color = f'rgba({color[0]}, {color[1]}, {color[2]}, 0.20)'

            # Use transparent_color in the line color
            fig.add_trace(go.Scatter(
                x=xys[estimator][e][0], y=xys[estimator][e][1],
                mode='lines', line=dict(color=transparent_color, width=1.5),
                name=f'{estimator} - CSV {e+1} X', showlegend=True))
            

    # Update layout
    fig.update_layout(
        title="Top view trajectory of all the trials",
        xaxis_title="X Position (m)",
        yaxis_title="Y Position (m)",
        template="plotly_white"
    )

    # Show the plot
    fig.show()


def plot_multiple_trajs_default(default_estimators, default_exps):
    # Generate color palette for the estimators
    colors_t = px.colors.qualitative.Plotly  # Use Plotly's color palette
    colors_t = [px.colors.hex_to_rgb(colors[i]) for i in range(len(default_estimators))]
    colors = dict.fromkeys(default_estimators)
    for i,estimator in enumerate(colors.keys()):
        colors[estimator] = colors_t[i]
    
    plot_multiple_trajs(default_estimators, default_exps, colors)


#plot_multiple_trajs_default(default_estimators, default_exps)