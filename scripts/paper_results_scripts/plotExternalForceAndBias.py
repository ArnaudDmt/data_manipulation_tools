from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px  # For color palette generation

import plotly.io as pio   
pio.kaleido.scope.mathjax = None


default_path = '.../Projects/HRP5_MultiContact_1'




contactNames = ["RightFootForceSensor"] #, "LeftFootForceSensor", "LeftHandForceSensor"] # ["RightFootForceSensor", "LeftFootForceSensor", "LeftHandForceSensor"]
contacts_area_when_set = [] # ["LeftHandForceSensor"]

contactNameToPlot = {"RightFootForceSensor": "Right foot", "LeftFootForceSensor": "Left foot", "LeftHandForceSensor": "Left hand"}



estimator_plot_args = {
    'KineticsObserver': {'name': 'Kinetics Observer', 'lineWidth': 7},
    'Hartley': {'name': 'RI-EKF', 'lineWidth': 5},
    'GroundTruth': {'name': 'Gound truth', 'lineWidth': 5}
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

        # Calculate y-axis limits
        def calculate_limits(*datas):
                # Finding the axis limits linked to the max spike
                margin = 0.001

                max_signed_value = 0
                for data in datas:
                        # Find the index of the maximum absolute value
                        max_abs_index = np.argmax(np.abs(data))
                        # Get the value at this index (with the original sign)
                        max_sv = data[max_abs_index]
                        if(np.abs(max_sv) > np.abs(max_signed_value)):
                                max_signed_value = max_sv

                y_min_spike = max_signed_value * (1 - margin * np.sign(max_signed_value))
                y_max_spike = max_signed_value * (1 + margin * np.sign(max_signed_value))

                # Finding the axis limits linked to final values

                final_data = [data[-1] for data in datas]

                min_signed_value = min(final_data)
                max_signed_value = max(final_data)

                # Apply the margin to the min and max signed values
                y_min_end = min_signed_value * (1 - margin * np.sign(min_signed_value))
                y_max_end = max_signed_value * (1 + margin * np.sign(max_signed_value))

                return (min(y_min_spike, y_min_end), max(y_max_spike, y_max_end))

        if(withTrueBias):
                y_limits_x = calculate_limits(HartleyBias[:, 0], KineticsBias[:, 0], trueBias[:, 0])
                y_limits_y = calculate_limits(HartleyBias[:, 1], KineticsBias[:, 1], trueBias[:, 1])
                y_limits_z = calculate_limits(HartleyBias[:, 2], KineticsBias[:, 2], trueBias[:, 2])
        else:
                y_limits_x = calculate_limits(HartleyBias[:, 0], KineticsBias[:, 0])
                y_limits_y = calculate_limits(HartleyBias[:, 1], KineticsBias[:, 1])
                y_limits_z = calculate_limits(HartleyBias[:, 2], KineticsBias[:, 2])

        # Create the figure
        figBias = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1, x_title='Time (s)',
        )

        # Add traces for x-axis bias
        color_Hartley = f'rgba({colors["Hartley"][0]}, {colors["Hartley"][1]}, {colors["Hartley"][2]}, 1)'
        color_Kinetics = f'rgba({colors["KineticsObserver"][0]}, {colors["KineticsObserver"][1]}, {colors["KineticsObserver"][2]}, 1)'
        color_Gt = "black"

        figBias.add_trace(go.Scatter(
            x=[None],  # Dummy x value
            y=[None],  # Dummy y value
            mode="lines",
            marker=dict(size=10, color=color_Kinetics),
            name=f"KineticsObserver"
        ))
        if(withTrueBias):
                figBias.add_trace(go.Scatter(
                x=[None],  # Dummy x value
                y=[None],  # Dummy y value
                mode="lines",
                marker=dict(size=10, color=color_Gt),
                name="Ground truth"
                ))
        figBias.add_trace(go.Scatter(
            x=[None],  # Dummy x value
            y=[None],  # Dummy y value
            mode="lines",
            marker=dict(size=10, color=color_Hartley),
            name=f"RI-EKF"
        ))
        

        figBias.add_trace(
        go.Scatter(
                x=observer_data["t"],
                y=HartleyBias[:, 0],
                mode="lines",showlegend= False,
                line=dict(width=estimator_plot_args["Hartley"]["lineWidth"], color=color_Hartley)
        ),
        row=1,
        col=1,
        )
        figBias.add_trace(
        go.Scatter(
                x=observer_data["t"],
                y=KineticsBias[:, 0],
                mode="lines",showlegend= False,
                line=dict(width=estimator_plot_args["KineticsObserver"]["lineWidth"], color=color_Kinetics)
        ),
        row=1,
        col=1,
        )
        if withTrueBias:
                figBias.add_trace(
                        go.Scatter(
                        x=observer_data["t"],
                        y=trueBias[:, 0],
                        line=dict(width=estimator_plot_args["GroundTruth"]["lineWidth"], color=color_Gt),
                        mode="lines",showlegend= False,
                        ),
                        row=1,
                        col=1,
                )

        # Add traces for y-axis bias
        figBias.add_trace(
        go.Scatter(
                x=observer_data["t"],
                y=HartleyBias[:, 1],
                mode="lines",showlegend= False,
                line=dict(width=estimator_plot_args["Hartley"]["lineWidth"], color=color_Hartley),
        ),
        row=2,
        col=1,
        )
        figBias.add_trace(
        go.Scatter(
                x=observer_data["t"],
                y=KineticsBias[:, 1],
                mode="lines",showlegend= False,
                line=dict(width=estimator_plot_args["KineticsObserver"]["lineWidth"], color=color_Kinetics),
        ),
        row=2,
        col=1,
        )
        if withTrueBias:
                figBias.add_trace(
                        go.Scatter(
                        x=observer_data["t"],
                        y=trueBias[:, 1],
                        line=dict(width=estimator_plot_args["GroundTruth"]["lineWidth"], color=color_Gt),showlegend= False,
                        mode="lines",
                        ),
                        row=2,
                        col=1,
        )

        # Add traces for z-axis bias
        figBias.add_trace(
        go.Scatter(
                x=observer_data["t"],
                y=HartleyBias[:, 2],
                mode="lines",showlegend= False,
                line=dict(width=estimator_plot_args["Hartley"]["lineWidth"], color=color_Hartley),
        ),
        row=3,
        col=1,
        )
        figBias.add_trace(
        go.Scatter(
                x=observer_data["t"],
                y=KineticsBias[:, 2],
                mode="lines",
                showlegend= False,
                line=dict(width=estimator_plot_args["KineticsObserver"]["lineWidth"], color=color_Kinetics),
        ),
        row=3,
        col=1,
        )
        if withTrueBias:
                figBias.add_trace(
                        go.Scatter(
                        x=observer_data["t"],
                        y=trueBias[:, 2],
                        line=dict(width=estimator_plot_args["GroundTruth"]["lineWidth"], color=color_Gt), showlegend= False,
                        mode="lines",
                        ),
                        row=3,
                        col=1,
                )

        

        # Apply calculated y-axis limits
        figBias.update_yaxes(range=y_limits_x, row=1, col=1)
        figBias.update_yaxes(range=y_limits_y, row=2, col=1)
        figBias.update_yaxes(range=y_limits_z, row=3, col=1)

        # Update layout
        figBias.update_layout(
                template="plotly_white",
                legend=dict(
                        yanchor="bottom",
                        y=1.06,
                        xanchor="left",
                        x=0.01,
                        orientation="h",
                        bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Times New Roman"),
                ),
                font = dict(family = 'Times New Roman'),
                yaxis=dict(
                        title=dict(text="Bias x (rad.s⁻¹)", standoff=20),  # Add spacing for alignment
                ),
                yaxis2=dict(
                        title=dict(text="Bias y (rad.s⁻¹)", standoff=20),
                ),
                yaxis3=dict(
                        title=dict(text="Bias z (rad.s⁻¹)", standoff=20),
                ),
                margin=dict(l=0.0,r=0.0,b=0.0,t=0.0)
                ,autosize=True  # Automatically adjusts the figure size
        )

        figBias.update_yaxes(
                dtick=0.01,        # Set interval between tick marks
                row=2, col=1
        )
        
        # Show the plot
        figBias.show()

        figBias.write_image(f'/tmp/biases.pdf')



def plotExtWrench(colors = None, path = default_path):
        figForces = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, x_title='Time (s)',
        )
        if(colors == None):
                # Generate color palette for the estimators
                colors_t = px.colors.qualitative.Plotly  # Use Plotly's color palette
                colors_t = [px.colors.hex_to_rgb(colors_t[i]) for i in range(len(estimator_plot_args))]
                colors = dict.fromkeys(estimator_plot_args)
                
                for i,estimator in enumerate(colors.keys()):
                        colors[estimator] = colors_t[i]


        #observer_data = pd.read_csv(f'{path}/output_data/observerResultsCSV.csv',  delimiter=';')
        observer_data = pd.read_csv(f'{path}/output_data/logReplay.csv',  delimiter=';')
        observer_data = observer_data.truncate(after=7000)
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
        
        color_Kinetics = f'rgba({colors["KineticsObserver"][0]}, {colors["KineticsObserver"][1]}, {colors["KineticsObserver"][2]}, 1)'
        color_Gt = f'rgba({colors["Mocap"][0]}, {colors["Mocap"][1]}, {colors["Mocap"][2]}, 1)'

        colors2 = generate_turbo_subset_colors('rainbow', contacts_area_when_set)

        contactLegendPlots = []
        for contactName2 in contacts_area_when_set:
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


        meas_forceX = observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_force_x"]
        est_forceX = observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_x"]
        maxhude = np.max(np.abs(meas_forceX - est_forceX))
        meandejid = np.mean(np.abs(meas_forceX - est_forceX))

        print(f"Force x: mean: {np.mean(np.abs(meas_forceX - est_forceX))} | std: {np.std(meas_forceX - est_forceX)} | MeanMaxRel: {(meandejid)/maxhude*100}")

        # figForceX.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_x"], mode='lines', name='Estimated force'))
        # figForceX.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_force_x"], mode='lines', name='Ground truth'))
        # for k, cont2 in enumerate(contacts_area_when_set):
        #         figForceX.add_trace(contactLegendPlots[k])

        # y_min = min(
        #         np.min(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_x"]),
        #         np.min(observer_data["LeftHandForceSensor_fx"]))
        # y_max = max(
        #         np.max(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_x"]),
        #         np.max(observer_data["LeftHandForceSensor_fx"]))

        # max_abs = max(np.abs(y_min), np.abs(y_max))

        # y_min = y_min - max_abs * 0.1
        # y_max = y_max + max_abs * 0.1

        # figForceX.update_yaxes(
        #         range=[y_min, y_max]
        #     )
        
        # figForceX.update_layout(
        #         shapes=shapes,
        #         xaxis_title="Time (s)",
        #         yaxis_title="Force X (N)",
        #         template="plotly_white",
        #         legend=dict(
        #                 yanchor="top",
        #                 y=0.99,
        #                 xanchor="left",
        #                 x=0.01,
        #                 orientation='h',
        #                 bgcolor = 'rgba(0,0,0,0)',
        #                 font = dict(family = 'Times New Roman')
        #                 ),
        #         legend2=dict(
        #                 yanchor="top",
        #                 y=0.92,
        #                 xanchor="left",
        #                 x=0.01,
        #                 bgcolor='rgba(0,0,0,0)',
        #         ),
        #         margin=dict(l=0,r=0,b=0,t=0),
        #         font = dict(family = 'Times New Roman')
                        
        #         )
        
        

        # # Show the plotly figure
        # figForceX.show()

        # figForceY = go.Figure()

        meas_forceY = observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_force_y"]
        est_forceY = observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_y"]

        maxhude = np.max(np.abs(meas_forceY - est_forceX))
        meandejid = np.mean(np.abs(meas_forceY - est_forceX))
        
        print(f"Force y: mean: {np.mean(np.abs(meas_forceY - est_forceY))} | std: {np.std(meas_forceY - est_forceY)} | MeanMaxRel: {(meandejid)/maxhude*100}")

        # figForceY.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_y"], mode='lines', name='Estimated force'))
        # figForceY.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_force_y"], mode='lines', name='Ground truth'))

        # y_min = min(
        #         np.min(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_y"]),
        #         np.min(observer_data["LeftHandForceSensor_fy"]))
        # y_max = max(
        #         np.max(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_y"]),
        #         np.max(observer_data["LeftHandForceSensor_fy"]))

        # max_abs = max(np.abs(y_min), np.abs(y_max))

        # y_min = y_min - max_abs * 0.1
        # y_max = y_max + max_abs * 0.1
        
        # figForceY.update_yaxes(
        #         range=[y_min, y_max]
        #     )

        # for k, cont2 in enumerate(contacts_area_when_set):
        #         figForceY.add_trace(contactLegendPlots[k])

        # figForceY.update_layout(
        #         shapes=shapes,
        #         xaxis_title="Time (s)",
        #         yaxis_title="Force Y (N)",
        #         template="plotly_white",
        #         legend=dict(
        #                 yanchor="top",
        #                 y=0.99,
        #                 xanchor="left",
        #                 x=0.01,
        #                 orientation='h',
        #                 bgcolor = 'rgba(0,0,0,0)',
        #                 font = dict(family = 'Times New Roman')
        #                 ),
        #         legend2=dict(
        #                 yanchor="top",
        #                 y=0.92,
        #                 xanchor="left",
        #                 x=0.01,
        #                 bgcolor='rgba(0,0,0,0)',
        #                 ),
        #         margin=dict(l=0,r=0,b=0,t=0),
        #         font = dict(family = 'Times New Roman')
        #         )

        # # Show the plotly figure
        # figForceY.show()

        meas_forceZ = observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_force_z"]
        est_forceZ = observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_z"]

        maxhude = np.max(np.abs(meas_forceZ - est_forceZ))
        meandejid = np.mean(np.abs(meas_forceZ - est_forceZ))
        
        print(f"Force z: mean: {np.mean(np.abs(meas_forceZ - est_forceZ))} | std: {np.std(meas_forceZ - est_forceZ)} | MeanMaxRel: {(meandejid)/maxhude*100}")

        figForces.add_trace(go.Scatter(x=observer_data["t"], y=meas_forceZ, line=dict(width = estimator_plot_args['GroundTruth']['lineWidth'], color=color_Gt), showlegend= False, mode='lines', name='Ground truth'), row = 1, col = 1)

        figForces.add_trace(go.Scatter(x=observer_data["t"], y=est_forceZ, line=dict(width = estimator_plot_args["KineticsObserver"]['lineWidth'], color=color_Kinetics), mode='lines', showlegend= False, name='Estimated force'), row = 1, col = 1)
        

        y_min = min(
                np.min(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_z"]),
                np.min(observer_data["LeftHandForceSensor_fz"]))
        y_max = max(
                np.max(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_z"]),
                np.max(observer_data["LeftHandForceSensor_fz"]))

        max_abs = max(np.abs(y_min), np.abs(y_max))

        y_min = y_min - max_abs * 0.1
        y_max = y_max + max_abs * 0.1


        # Apply calculated y-axis limits
        #figForces.update_yaxes(range=(y_min, y_max), row=1, col=1)
        
        
        # figForceZ.update_yaxes(
        #         range=[y_min, y_max]
        #     )

        # figForceZ.update_layout(
        #         shapes=shapes,
        #         xaxis_title="Time (s)",
        #         yaxis_title="Force Z (N)",
        #         template="plotly_white",
        #         legend=dict(
        #                 yanchor="top",
        #                 y=1.03,
        #                 xanchor="left",
        #                 x=0.01,
        #                 orientation='h',
        #                 bgcolor = 'rgba(0,0,0,0)',
        #                 font = dict(family = 'Times New Roman')
        #                 ),
        #         legend2=dict(
        #                 yanchor="top",
        #                 y=0.92,
        #                 xanchor="left",
        #                 x=0.01,
        #                 bgcolor='rgba(0,0,0,0)',
        #         ),
        #         margin=dict(l=0,r=0,b=0,t=0),
        #         font = dict(family = 'Times New Roman')
        #         )

        meas_torqueX = observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_torque_x"]
        est_torqueX = observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_x"]

        maxhude = np.max(np.abs(meas_torqueX - est_torqueX))
        meandejid = np.mean(np.abs(meas_torqueX - est_torqueX))
        
        print(f"Torque x: mean: {np.mean(np.abs(meas_torqueX - est_torqueX))} | std: {np.std(meas_torqueX - est_torqueX)} | MeanMaxRel: {(meandejid)/maxhude*100}")

        figForces.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_torque_x"], line=dict(width = estimator_plot_args['GroundTruth']['lineWidth'], color=color_Gt), showlegend= False, mode='lines'), row = 2, col = 1)
        figForces.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_x"], line=dict(width = estimator_plot_args["KineticsObserver"]['lineWidth'], color=color_Kinetics), showlegend= False, mode='lines'), row = 2, col = 1)
        

        y_min = min(
                np.min(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_x"]),
                np.min(observer_data["LeftHandForceSensor_cx"]))
        y_max = max(
                np.max(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_x"]),
                np.max(observer_data["LeftHandForceSensor_cx"]))

        max_abs = max(np.abs(y_min), np.abs(y_max))

        y_min = y_min - max_abs * 0.1
        y_max = y_max + max_abs * 0.1
        
        #figForces.update_yaxes(range=(y_min, y_max), row=2, col=1)
        
        # figForces.update_yaxes(
        #         range=[y_min, y_max]
        #     )

        # figTorqueX.add_annotation(
        #         x=0.9,
        #         y=0.9,
        #         xref="x domain",
        #         yref="y domain",
        #         text="STD:",
        #         font=dict(
        #         family="Times New Roman",
        #         size=16,
        #         color="black"
        #         ),
        #         bgcolor="rgba(0,0,0,0)",
        #         )


        # figTorqueY = go.Figure()

        meas_torqueY = observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_torque_y"]
        est_torqueY = observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_y"]

        maxhude = np.max(np.abs(meas_torqueY - est_torqueY))
        meandejid = np.mean(np.abs(meas_torqueY - est_torqueY))

        print(f"Torque y: mean: {np.mean(np.abs(meas_torqueY - est_torqueY))} | std: {np.std(meas_torqueY - est_torqueY)} | MeanMaxRel: {(meandejid)/maxhude*100}")

        # figTorqueY.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_y"], mode='lines', name='Estimated torque'))
        # figTorqueY.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_torque_y"], mode='lines', name='Ground truth'))
        # for k, cont2 in enumerate(contacts_area_when_set):
        #         figTorqueY.add_trace(contactLegendPlots[k])

        # y_min = min(
        #         np.min(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_y"]),
        #         np.min(observer_data["LeftHandForceSensor_cy"]))
        # y_max = max(
        #         np.max(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_y"]),
        #         np.max(observer_data["LeftHandForceSensor_cy"]))

        # max_abs = max(np.abs(y_min), np.abs(y_max))

        # y_min = y_min - max_abs * 0.1
        # y_max = y_max + max_abs * 0.1
        
        # figTorqueY.update_yaxes(
        #         range=[y_min, y_max]
        #     )
        
        # figTorqueY.update_layout(
        #         shapes=shapes,
        #         xaxis_title="Time (s)",
        #         yaxis_title="Torque Y (N.m)",
        #         template="plotly_white",
        #         legend=dict(
        #                 yanchor="top",
        #                 y=0.99,
        #                 xanchor="left",
        #                 x=0.01,
        #                 orientation='h',
        #                 bgcolor = 'rgba(0,0,0,0)',
        #                 font = dict(family = 'Times New Roman')
        #                 ),
        #         legend2=dict(
        #                 yanchor="top",
        #                 y=0.92,
        #                 xanchor="left",
        #                 x=0.01,
        #                 bgcolor='rgba(0,0,0,0)',
        #         ),
        #         margin=dict(l=0,r=0,b=0,t=0),
        #         font = dict(family = 'Times New Roman')
        #         )

        # # Show the plotly figure
        # figTorqueY.show()

        # figTorqueZ = go.Figure()

        meas_torqueZ = observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_torque_z"]
        est_torqueZ = observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_z"]

        maxhude = np.max(np.abs(meas_torqueZ - est_torqueZ))
        meandejid = np.mean(np.abs(meas_torqueZ - est_torqueZ))

        print(f"Torque z: mean: {np.mean(np.abs(meas_torqueZ - est_torqueZ))} | std: {np.std(meas_torqueZ - est_torqueZ)} | MeanMaxRel: {(meandejid)/maxhude*100}")

        # figTorqueZ.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_z"], mode='lines', name='Estimated torque'))
        # figTorqueZ.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_torque_z"], mode='lines', name='Ground truth'))
        # for k, cont2 in enumerate(contacts_area_when_set):
        #         figTorqueZ.add_trace(contactLegendPlots[k])

        # y_min = min(
        #         np.min(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_z"]),
        #         np.min(observer_data["LeftHandForceSensor_cz"]))
        # y_max = max(
        #         np.max(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_z"]),
        #         np.max(observer_data["LeftHandForceSensor_cz"]))

        # max_abs = max(np.abs(y_min), np.abs(y_max))

        # y_min = y_min - max_abs * 0.1
        # y_max = y_max + max_abs * 0.1
        
        # figTorqueZ.update_yaxes(
        #         range=[y_min, y_max]
        #     )
        
        # figTorqueZ.update_layout(
        #         shapes=shapes,
        #         xaxis_title="Time (s)",
        #         yaxis_title="Torque Z (N)",
        #         template="plotly_white",
        #         legend=dict(
        #                 yanchor="top",
        #                 y=0.99,
        #                 xanchor="left",
        #                 x=0.01,
        #                 orientation='h',
        #                 bgcolor = 'rgba(0,0,0,0)',
        #                 font = dict(family = 'Times New Roman')
        #                 ),
        #         legend2=dict(
        #                 yanchor="top",
        #                 y=0.92,
        #                 xanchor="left",
        #                 x=0.01,
        #                 bgcolor='rgba(0,0,0,0)',
        #         ),
        #         margin=dict(l=0,r=0,b=0,t=0),
        #         font = dict(family = 'Times New Roman')
        #         )

        # # Show the plotly figure
        # figTorqueZ.show()
        
        
        figForces.add_trace(go.Scatter(
            x=[None],  # Dummy x value
            y=[None],  # Dummy y value
            mode="lines",
            line=dict(width = estimator_plot_args['GroundTruth']['lineWidth'], color=color_Gt),
            name="Ground truth"
        ))

        figForces.add_trace(go.Scatter(
            x=[None],  # Dummy x value
            y=[None],  # Dummy y value
            mode="lines",
            line=dict(width = estimator_plot_args['KineticsObserver']['lineWidth'], color=color_Kinetics),
            name=f"Estimated wrench"
        ))

        
        

        figForces.update_layout(
                template="plotly_white",
                legend=dict(
                        yanchor="bottom",
                        y=1.03,
                        xanchor="left",
                        x=0.01,
                        orientation="h",
                        bgcolor="rgba(0,0,0,0)",
                        traceorder='reversed',
                        font = dict(family = 'Times New Roman', size=22, color="black"),
                ),
                font = dict(family = 'Times New Roman', size=22, color="black"),
                yaxis=dict(
                        title=dict(text="Force Z (N)", standoff=20),  # Add spacing for alignment
                ),
                yaxis2=dict(
                        title=dict(text="Torque X (N.m)", standoff=20),
                ),
                margin=dict(l=0.0,r=0.0,b=0.0,t=0.0)
                ,autosize=True  # Automatically adjusts the figure size
        )

        figForces.show()
        figForces.write_image(f'/tmp/extForces.pdf')


def computeExtWrenchError(expsToMerge):
        meas_ForceX = []
        est_ForceX = []
        meas_ForceY = []
        est_ForceY = []
        meas_ForceZ = []
        est_ForceZ = []
        meas_TorqueX = []
        est_TorqueX = []
        meas_TorqueY = []
        est_TorqueY = []
        meas_TorqueZ = []
        est_TorqueZ = []

        
        for exp in expsToMerge:
                path = f'Projects/{exp}'
                observer_data = pd.read_csv(f'{path}/output_data/logReplay.csv',  delimiter=';')

                meas_ForceX.append(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_force_x"])
                est_ForceX.append(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_x"])

                meas_ForceY.append(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_force_y"])
                est_ForceY.append(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_y"])
                
                meas_ForceZ.append(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_force_z"])
                est_ForceZ.append(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_z"])
                
                meas_TorqueX.append(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_torque_x"])
                est_TorqueX.append(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_x"])
                
                meas_TorqueY.append(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_torque_y"])
                est_TorqueY.append(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_y"])

                meas_TorqueZ.append(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_debug_wrenchesInCentroid_LeftHandForceSensor_torque_z"])
                est_TorqueZ.append(observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_z"])


        meas_ForceX = np.concatenate(meas_ForceX)
        est_ForceX = np.concatenate(est_ForceX)
        meas_ForceY = np.concatenate(meas_ForceY)
        est_ForceY = np.concatenate(est_ForceY)
        meas_ForceZ = np.concatenate(meas_ForceZ)
        est_ForceZ = np.concatenate(est_ForceZ)
        meas_TorqueX = np.concatenate(meas_TorqueX)
        est_TorqueX = np.concatenate(est_TorqueX)
        meas_TorqueY = np.concatenate(meas_TorqueY)
        est_TorqueY = np.concatenate(est_TorqueY)
        meas_TorqueZ = np.concatenate(meas_TorqueZ)
        est_TorqueZ = np.concatenate(est_TorqueZ)

        max_ForceX = np.max(np.abs(meas_ForceX - est_ForceX))
        mean_ForceX = np.mean(np.abs(meas_ForceX - est_ForceX))
        max_ForceY = np.max(np.abs(meas_ForceY - est_ForceY))
        mean_ForceY = np.mean(np.abs(meas_ForceY - est_ForceY))
        max_ForceZ = np.max(np.abs(meas_ForceZ - est_ForceZ))
        mean_ForceZ = np.mean(np.abs(meas_ForceZ - est_ForceZ))
        max_TorqueX = np.max(np.abs(meas_TorqueX - est_TorqueX))
        mean_TorqueX = np.mean(np.abs(meas_TorqueX - est_TorqueX))
        max_TorqueY = np.max(np.abs(meas_TorqueY - est_TorqueY))
        mean_TorqueY = np.mean(np.abs(meas_TorqueY - est_TorqueY))
        max_TorqueZ = np.max(np.abs(meas_TorqueZ - est_TorqueZ))
        mean_TorqueZ = np.mean(np.abs(meas_TorqueZ - est_TorqueZ))

        print(f"Force x: mean: {np.mean(np.abs(meas_ForceX - est_ForceX))} | std: {np.std(meas_ForceX - est_ForceX)} | MeanMaxRel: {(mean_ForceX)/max_ForceX*100}")
        print(f"Force y: mean: {np.mean(np.abs(meas_ForceY - est_ForceY))} | std: {np.std(meas_ForceY - est_ForceY)} | MeanMaxRel: {(mean_ForceY)/max_ForceY*100}")
        print(f"Force z: mean: {np.mean(np.abs(meas_ForceZ - est_ForceZ))} | std: {np.std(meas_ForceZ - est_ForceZ)} | MeanMaxRel: {(mean_ForceZ)/max_ForceZ*100}")
        print(f"Torque x: mean: {np.mean(np.abs(meas_TorqueX - est_TorqueX))} | std: {np.std(meas_TorqueX - est_TorqueX)} | MeanMaxRel: {(mean_TorqueX)/max_TorqueX*100}")
        print(f"Torque y: mean: {np.mean(np.abs(meas_TorqueY - est_TorqueY))} | std: {np.std(meas_TorqueY - est_TorqueY)} | MeanMaxRel: {(mean_TorqueY)/max_TorqueY*100}")
        print(f"Torque z: mean: {np.mean(np.abs(meas_TorqueZ - est_TorqueZ))} | std: {np.std(meas_TorqueZ - est_TorqueZ)} | MeanMaxRel: {(mean_TorqueZ)/max_TorqueZ*100}")


if __name__ == '__main__':
    plotExtWrench()
