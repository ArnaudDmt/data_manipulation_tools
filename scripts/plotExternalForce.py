import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  # For color palette generation


# Load the CSV files into pandas dataframes
observer_data = pd.read_csv(f'../Projects/HRP5_MultiContact_1/output_data/logReplay.csv',  delimiter=';')

colors_t = px.colors.qualitative.Plotly  # Use Plotly's color palette
colors_t = [px.colors.hex_to_rgb(colors_t[i]) for i in range(2)]

figForceX = go.Figure()

figForceX.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_x"], mode='lines', name='Estimated force'))
figForceX.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["LeftHandForceSensor_fx"], mode='lines', name='Ground truth'))

figForceX.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Force X (N)",
            template="plotly_white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                orientation='h',
                bgcolor = 'rgba(0,0,0,0)'
                )
        )

# Show the plotly figure
figForceX.show()

figForceY = go.Figure()

figForceY.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_y"], mode='lines', name='Estimated force'))
figForceY.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["LeftHandForceSensor_fy"], mode='lines', name='Ground truth'))

figForceY.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Force Y (N)",
            template="plotly_white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                orientation='h',
                bgcolor = 'rgba(0,0,0,0)'
                )
        )

# Show the plotly figure
figForceY.show()

figForceZ = go.Figure()

figForceZ.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extForceCentr_z"], mode='lines', name='Estimated force'))
figForceZ.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["LeftHandForceSensor_fz"], mode='lines', name='Ground truth'))

figForceZ.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Force Z (N)",
            template="plotly_white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                orientation='h',
                bgcolor = 'rgba(0,0,0,0)'
                )
        )

# Show the plotly figure
figForceZ.show()


figTorqueX = go.Figure()

figTorqueX.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_x"], mode='lines', name='Estimated Torque'))
figTorqueX.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["LeftHandForceSensor_cx"], mode='lines', name='Ground truth'))

figTorqueX.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Torque X (N.m)",
            template="plotly_white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                orientation='h',
                bgcolor = 'rgba(0,0,0,0)'
                )
        )

# Show the plotly figure
figTorqueX.show()

figTorqueY = go.Figure()

figTorqueY.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_y"], mode='lines', name='Estimated Torque'))
figTorqueY.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["LeftHandForceSensor_cy"], mode='lines', name='Ground truth'))

figTorqueY.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Torque Y (N.m)",
            template="plotly_white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                orientation='h',
                bgcolor = 'rgba(0,0,0,0)'
                )
        )

# Show the plotly figure
figTorqueY.show()

figTorqueZ = go.Figure()

figTorqueZ.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_extTorqueCentr_z"], mode='lines', name='Estimated Torque'))
figTorqueZ.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["LeftHandForceSensor_cz"], mode='lines', name='Ground truth'))

figTorqueZ.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Torque Z (N)",
            template="plotly_white",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                orientation='h',
                bgcolor = 'rgba(0,0,0,0)'
                )
        )

# Show the plotly figure
figTorqueZ.show()