from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  # For color palette generation

import plotly.io as pio   
pio.kaleido.scope.mathjax = None


default_path = '.../Projects/HRP5_MultiContact_1'




contactNames = ["RightFootForceSensor"] #, "LeftFootForceSensor", "LeftHandForceSensor"] # ["RightFootForceSensor", "LeftFootForceSensor", "LeftHandForceSensor"]
contactNames2 = ["LeftHandForceSensor"]

contactNameToPlot = {"RightFootForceSensor": "Right foot", "LeftFootForceSensor": "Left foot", "LeftHandForceSensor": "Left hand"}



estimator_plot_args = {
    'KineticsObserver': {'name': 'Kinetics Observer', 'lineWidth': 3},
    'Mocap': {'name': 'Ground truth', 'lineWidth': 2}
}


def plotGyroBias(colors = None, path = default_path):
        Hartley_data = pd.read_csv(f'{path}/output_data/HartleyOutputCSV.csv',  delimiter=';')

        # Load the CSV files into pandas dataframes
        observer_data = pd.read_csv(f'{path}/output_data/logReplay.csv',  delimiter=';')

        #Hartley_data = Hartley_data.truncate(after=200)
        #observer_data = observer_data.truncate(after=200)

       
        timeAxis = observer_data["t"].to_numpy()
        timeAxis = np.hstack((np.array(0.0).reshape(1,), timeAxis))
        
        HartleyBias = Hartley_data[['IMU_GyroBias_x', 'IMU_GyroBias_y', 'IMU_GyroBias_z']].to_numpy()
        KineticsBias = observer_data[['Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_gyroBias_Accelerometer_x', 'Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_gyroBias_Accelerometer_y', 'Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_gyroBias_Accelerometer_z']].to_numpy()
        HartleyBias = np.vstack((np.array([0.0, 0.0, 0.0]).reshape(1,3), HartleyBias))
        KineticsBias = np.vstack((np.array([0.0, 0.0, 0.0]).reshape(1,3), KineticsBias))

        withTrueBias = False
        if(withTrueBias):
                trueBias =  observer_data[['NoisySensors_gyro_Accelerometer_bias_x', 'NoisySensors_gyro_Accelerometer_bias_y', 'NoisySensors_gyro_Accelerometer_bias_z']].to_numpy()
                trueBias = np.vstack((np.array([0.0, 0.0, 0.0]).reshape(1,3), trueBias))


        if(colors == None):
                # Generate color palette for the estimators
                colors_t = px.colors.qualitative.Plotly  # Use Plotly's color palette
                colors_t = [px.colors.hex_to_rgb(colors_t[i]) for i in range(len(estimator_plot_args))]
                colors = dict.fromkeys(estimator_plot_args)
                
                for i,estimator in enumerate(colors.keys()):
                        colors[estimator] = colors_t[i]

        figBias = go.Figure()
        color_Hartley = f'rgba({colors["Hartley"][0]}, {colors["Hartley"][1]}, {colors["Hartley"][2]}, 1)'
        figBias.add_trace(go.Scatter(x=observer_data['t'], y=HartleyBias[:,0], mode='lines', line=dict(color = color_Hartley), name='Hartley_GyroBias_x'))
        figBias.add_trace(go.Scatter(x=observer_data['t'], y=HartleyBias[:,1], mode='lines', line=dict(color = color_Hartley), name='Hartley_GyroBias_y'))
        figBias.add_trace(go.Scatter(x=observer_data['t'], y=HartleyBias[:,2], mode='lines', line=dict(color = color_Hartley), name='Hartley_GyroBias_z'))

        color_Kinetics = f'rgba({colors["KineticsObserver"][0]}, {colors["KineticsObserver"][1]}, {colors["KineticsObserver"][2]}, 1)'
        figBias.add_trace(go.Scatter(x=observer_data["t"], y=KineticsBias[:,0], mode='lines', line=dict(color = color_Kinetics), name='KineticsObserver_GyroBias_x'))
        figBias.add_trace(go.Scatter(x=observer_data["t"], y=KineticsBias[:,1], mode='lines', line=dict(color = color_Kinetics), name='KineticsObserver_GyroBias_y'))
        figBias.add_trace(go.Scatter(x=observer_data["t"], y=KineticsBias[:,2], mode='lines', line=dict(color = color_Kinetics), name='KineticsObserver_GyroBias_z'))

        if(withTrueBias):
                figBias.add_trace(go.Scatter(x=observer_data["t"], y=trueBias[:,0], mode='lines', name='True Bias x'))
                figBias.add_trace(go.Scatter(x=observer_data["t"], y=trueBias[:,1], mode='lines', name='True Bias y'))
                figBias.add_trace(go.Scatter(x=observer_data["t"], y=trueBias[:,2], mode='lines', name='True Bias z'))

        
        figBias.update_layout(
                xaxis_title="Time (s)",
                yaxis_title="Bias (rad.s-1)",
                template="plotly_white",
                legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        orientation='h',
                        bgcolor = 'rgba(0,0,0,0)',
                        font = dict(family = 'Times New Roman')
                        )
                        
                )

        # Show the plotly figure
        figBias.show()





def plotExtWrench(colors = None, path = default_path):
        if(colors == None):
                # Generate color palette for the estimators
                colors_t = px.colors.qualitative.Plotly  # Use Plotly's color palette
                colors_t = [px.colors.hex_to_rgb(colors_t[i]) for i in range(len(estimator_plot_args))]
                colors = dict.fromkeys(estimator_plot_args)
                
                for i,estimator in enumerate(colors.keys()):
                        colors[estimator] = colors_t[i]


        #observer_data = pd.read_csv(f'{path}/output_data/observerResultsCSV.csv',  delimiter=';')
        observer_data = pd.read_csv(f'{path}/output_data/logReplay.csv',  delimiter=';')
        shapes = []

        # colors_t = px.colors.qualitative.Plotly  # Use Plotly's color palette
        # colors_t = [px.colors.hex_to_rgb(colors_t[i]) for i in range(2)]

        figForceX = go.Figure()

        def generate_turbo_subset_colors(colormapName, contactsList):
                cmap = plt.get_cmap(colormapName)
                listCoeffs = np.linspace(0.2, 0.8, len(contactsList))
                colors={}
                # Generate colors and reduce intensity
                for idx, estimator in enumerate(contactsList):
                        r, g, b, t = cmap(listCoeffs[idx])
                        colors[estimator] = (r, g, b, 1) 
        
                return colors

        colors2 = generate_turbo_subset_colors('rainbow', contactNames2)

        contactLegendPlots = []
        for contactName2 in contactNames2:
                is_set_mask = observer_data[f"Observers_MainObserverPipeline_MCKineticsObserver_debug_contactState_isSet_{contactName2}"] == "Set"
                #contact_state = [encoders_data[f"Observers_MainObserverPipeline_MCKineticsObserver_debug_contactState_isSet_{contactName2}"]  for _ in iterations]
                # Identify "Set" regions
                set_regions = []
                start = None
                for i, state in enumerate(is_set_mask):
                        if state and start is None:
                                # Begin a new region
                                start = observer_data["t"][i]  # Start of "Set" region
                        elif not state and start is not None:
                                # End the current region
                                set_regions.append((start, observer_data["t"][i - 1]))
                                start = None

                # Handle the case where the last region ends at the final iteration
                if start is not None:
                        set_regions.append((start, observer_data["t"][-1]))

                # Assign color for the current contact
                fillcolor2 = colors2[contactName2]
                fillcolor2 = f'rgba({fillcolor2[0]}, {fillcolor2[1]}, {fillcolor2[2]}, 0.3)'

                # Add an scatter trace as a legend proxy
                contactLegendPlots.append(go.Scatter(
                x=[None],  # Dummy x value
                y=[None],  # Dummy y value
                mode="markers",
                marker=dict(size=10, color=fillcolor2),
                name=f"{contactNameToPlot[contactName2]}",
                legend="legend2"
                ))
                
                # Create shapes for shaded regions
                y_min, y_max = -1000, 1000  # Set bounds for y-axis
                for start, end in set_regions:
                        shapes.append(dict(
                                type="rect",
                                xref="x",
                                yref="y",
                                x0=start,
                                y0=y_min,
                                x1=end,
                                y1=y_max,
                                opacity=0.4,
                                line_width=0,
                                layer="below",
                                fillcolor=fillcolor2,
                        ))



        figForceX.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_x"], mode='lines', name='Estimated force'))
        figForceX.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["LeftHandForceSensor_fx"], mode='lines', name='Ground truth'))
        for k, cont2 in enumerate(contactNames2):
                figForceX.add_trace(contactLegendPlots[k])

        y_min = min(
                np.min(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_x"]),
                np.min(observer_data["LeftHandForceSensor_fx"]))
        y_max = max(
                np.max(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_x"]),
                np.max(observer_data["LeftHandForceSensor_fx"]))

        max_abs = max(np.abs(y_min), np.abs(y_max))

        y_min = y_min - max_abs * 0.1
        y_max = y_max + max_abs * 0.1

        figForceX.update_yaxes(
                range=[y_min, y_max]
            )
        
        figForceX.update_layout(
                shapes=shapes,
                xaxis_title="Time (s)",
                yaxis_title="Force X (N)",
                template="plotly_white",
                legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        orientation='h',
                        bgcolor = 'rgba(0,0,0,0)',
                        font = dict(family = 'Times New Roman')
                        ),
                legend2=dict(
                        yanchor="top",
                        y=0.92,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(0,0,0,0)',
                ),
                margin=dict(l=0,r=0,b=0,t=0),
                font = dict(family = 'Times New Roman')
                        
                )
        
        

        # Show the plotly figure
        figForceX.show()

        figForceY = go.Figure()

        figForceY.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_y"], mode='lines', name='Estimated force'))
        figForceY.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["LeftHandForceSensor_fy"], mode='lines', name='Ground truth'))

        y_min = min(
                np.min(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_y"]),
                np.min(observer_data["LeftHandForceSensor_fy"]))
        y_max = max(
                np.max(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_y"]),
                np.max(observer_data["LeftHandForceSensor_fy"]))

        max_abs = max(np.abs(y_min), np.abs(y_max))

        y_min = y_min - max_abs * 0.1
        y_max = y_max + max_abs * 0.1
        
        figForceY.update_yaxes(
                range=[y_min, y_max]
            )

        for k, cont2 in enumerate(contactNames2):
                figForceY.add_trace(contactLegendPlots[k])

        figForceY.update_layout(
                shapes=shapes,
                xaxis_title="Time (s)",
                yaxis_title="Force Y (N)",
                template="plotly_white",
                legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        orientation='h',
                        bgcolor = 'rgba(0,0,0,0)',
                        font = dict(family = 'Times New Roman')
                        ),
                legend2=dict(
                        yanchor="top",
                        y=0.92,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(0,0,0,0)',
                        ),
                margin=dict(l=0,r=0,b=0,t=0),
                font = dict(family = 'Times New Roman')
                )

        # Show the plotly figure
        figForceY.show()

        figForceZ = go.Figure()

        figForceZ.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_z"], mode='lines', name='Estimated force'))
        figForceZ.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["LeftHandForceSensor_fz"], mode='lines', name='Ground truth'))
        for k, cont2 in enumerate(contactNames2):
                figForceZ.add_trace(contactLegendPlots[k])

        y_min = min(
                np.min(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_z"]),
                np.min(observer_data["LeftHandForceSensor_fz"]))
        y_max = max(
                np.max(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_z"]),
                np.max(observer_data["LeftHandForceSensor_fz"]))

        max_abs = max(np.abs(y_min), np.abs(y_max))

        y_min = y_min - max_abs * 0.1
        y_max = y_max + max_abs * 0.1
        
        figForceZ.update_yaxes(
                range=[y_min, y_max]
            )

        figForceZ.update_layout(
                shapes=shapes,
                xaxis_title="Time (s)",
                yaxis_title="Force Z (N)",
                template="plotly_white",
                legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        orientation='h',
                        bgcolor = 'rgba(0,0,0,0)',
                        font = dict(family = 'Times New Roman')
                        ),
                legend2=dict(
                        yanchor="top",
                        y=0.92,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(0,0,0,0)',
                ),
                margin=dict(l=0,r=0,b=0,t=0),
                font = dict(family = 'Times New Roman')
                )

        # Show the plotly figure
        figForceZ.show()


        figTorqueX = go.Figure()

        figTorqueX.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_x"], mode='lines', name='Estimated Torque'))
        figTorqueX.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["LeftHandForceSensor_cx"], mode='lines', name='Ground truth'))
        for k, cont2 in enumerate(contactNames2):
                figTorqueX.add_trace(contactLegendPlots[k])

        y_min = min(
                np.min(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_x"]),
                np.min(observer_data["LeftHandForceSensor_cx"]))
        y_max = max(
                np.max(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_x"]),
                np.max(observer_data["LeftHandForceSensor_cx"]))

        max_abs = max(np.abs(y_min), np.abs(y_max))

        y_min = y_min - max_abs * 0.1
        y_max = y_max + max_abs * 0.1
        
        figTorqueX.update_yaxes(
                range=[y_min, y_max]
            )
        
        figTorqueX.update_layout(
                shapes=shapes,
                xaxis_title="Time (s)",
                yaxis_title="Torque X (N.m)",
                template="plotly_white",
                legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        orientation='h',
                        bgcolor = 'rgba(0,0,0,0)',
                        font = dict(family = 'Times New Roman')
                        ),
                legend2=dict(
                        yanchor="top",
                        y=0.92,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(0,0,0,0)',
                ),
                margin=dict(l=0,r=0,b=0,t=0),
                font = dict(family = 'Times New Roman')
                )

        # Show the plotly figure
        figTorqueX.show()

        figTorqueY = go.Figure()

        figTorqueY.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_y"], mode='lines', name='Estimated Torque'))
        figTorqueY.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["LeftHandForceSensor_cy"], mode='lines', name='Ground truth'))
        for k, cont2 in enumerate(contactNames2):
                figTorqueY.add_trace(contactLegendPlots[k])

        y_min = min(
                np.min(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_y"]),
                np.min(observer_data["LeftHandForceSensor_cy"]))
        y_max = max(
                np.max(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_y"]),
                np.max(observer_data["LeftHandForceSensor_cy"]))

        max_abs = max(np.abs(y_min), np.abs(y_max))

        y_min = y_min - max_abs * 0.1
        y_max = y_max + max_abs * 0.1
        
        figTorqueY.update_yaxes(
                range=[y_min, y_max]
            )
        
        figTorqueY.update_layout(
                shapes=shapes,
                xaxis_title="Time (s)",
                yaxis_title="Torque Y (N.m)",
                template="plotly_white",
                legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        orientation='h',
                        bgcolor = 'rgba(0,0,0,0)',
                        font = dict(family = 'Times New Roman')
                        ),
                legend2=dict(
                        yanchor="top",
                        y=0.92,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(0,0,0,0)',
                ),
                margin=dict(l=0,r=0,b=0,t=0),
                font = dict(family = 'Times New Roman')
                )

        # Show the plotly figure
        figTorqueY.show()

        figTorqueZ = go.Figure()

        figTorqueZ.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_z"], mode='lines', name='Estimated Torque'))
        figTorqueZ.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["LeftHandForceSensor_cz"], mode='lines', name='Ground truth'))
        for k, cont2 in enumerate(contactNames2):
                figTorqueZ.add_trace(contactLegendPlots[k])

        y_min = min(
                np.min(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_z"]),
                np.min(observer_data["LeftHandForceSensor_cz"]))
        y_max = max(
                np.max(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_z"]),
                np.max(observer_data["LeftHandForceSensor_cz"]))

        max_abs = max(np.abs(y_min), np.abs(y_max))

        y_min = y_min - max_abs * 0.1
        y_max = y_max + max_abs * 0.1
        
        figTorqueZ.update_yaxes(
                range=[y_min, y_max]
            )
        
        figTorqueZ.update_layout(
                shapes=shapes,
                xaxis_title="Time (s)",
                yaxis_title="Torque Z (N)",
                template="plotly_white",
                legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        orientation='h',
                        bgcolor = 'rgba(0,0,0,0)',
                        font = dict(family = 'Times New Roman')
                        ),
                legend2=dict(
                        yanchor="top",
                        y=0.92,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(0,0,0,0)',
                ),
                margin=dict(l=0,r=0,b=0,t=0),
                font = dict(family = 'Times New Roman')
                )

        # Show the plotly figure
        figTorqueZ.show()


if __name__ == '__main__':
    plotExtWrench()
