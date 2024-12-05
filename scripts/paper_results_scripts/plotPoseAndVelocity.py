from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px  # For color palette generation
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter,filtfilt

import plotly.io as pio   
pio.kaleido.scope.mathjax = None


default_path = '.../Projects/HRP5_MultiContact_1'


timeStep_s = 0.005

contactNames = ["RightFootForceSensor"] #, "LeftFootForceSensor", "LeftHandForceSensor"] # ["RightFootForceSensor", "LeftFootForceSensor", "LeftHandForceSensor"]
contacts_area_when_set = [] # ["LeftHandForceSensor"]

contactNameToPlot = {"RightFootForceSensor": "Right foot", "LeftFootForceSensor": "Left foot", "LeftHandForceSensor": "Left hand"}

zeros_row = np.zeros((1, 3))

estimator_plot_args = {
    'KineticsObserver': {'name': 'Kinetics Observer', 'lineWidth': 1},
    'Hartley': {'name': 'RI-EKF', 'lineWidth': 1},
    'Mocap': {'name': 'Gound truth', 'lineWidth': 1}
}


def plotPoseVel(estimators, path = default_path, colors = None):
    figPoseVel = make_subplots(
    rows=3, cols=3, shared_xaxes=True, vertical_spacing=0.05, horizontal_spacing=0.09
    )

    observer_data = pd.read_csv(f'{path}/output_data/observerResultsCSV.csv',  delimiter=';')
    observer_data = observer_data[observer_data["Mocap_datasOverlapping"] == "Datas overlap"]

    # Pose and vels of the imu in the floating base
    posImuFb_overlap = observer_data[['HartleyIEKF_imuFbKine_position_x', 'HartleyIEKF_imuFbKine_position_y', 'HartleyIEKF_imuFbKine_position_z']].to_numpy()
    quaternions_rImuFb_overlap = observer_data[['HartleyIEKF_imuFbKine_ori_x', 'HartleyIEKF_imuFbKine_ori_y', 'HartleyIEKF_imuFbKine_ori_z', 'HartleyIEKF_imuFbKine_ori_w']].to_numpy()
    rImuFb_overlap = R.from_quat(quaternions_rImuFb_overlap)
    linVelImuFb_overlap = observer_data[['HartleyIEKF_imuFbKine_linVel_x', 'HartleyIEKF_imuFbKine_linVel_y', 'HartleyIEKF_imuFbKine_linVel_z']].to_numpy()
    angVelImuFb_overlap = observer_data[['HartleyIEKF_imuFbKine_angVel_x', 'HartleyIEKF_imuFbKine_angVel_y', 'HartleyIEKF_imuFbKine_angVel_z']].to_numpy()
    posFbImu_overlap = - rImuFb_overlap.apply(posImuFb_overlap, inverse=True)
    linVelFbImu_overlap = rImuFb_overlap.apply(np.cross(angVelImuFb_overlap, posImuFb_overlap), inverse=True) - rImuFb_overlap.apply(linVelImuFb_overlap, inverse=True)


    estimatorsPoses = { 'Mocap': {'pos': observer_data[['Mocap_pos_x', 'Mocap_pos_y', 'Mocap_pos_z']].to_numpy(), \
                                    'ori': R.from_quat(observer_data[['Mocap_ori_x', 'Mocap_ori_y', 'Mocap_ori_z', 'Mocap_ori_w']].to_numpy()), \
                                    'linVel': None, \
                                    'angVel': None}, \
                        # 'Controller': {'pos': observer_data[['Controller_tx', 'Controller_ty', 'Controller_tz']].to_numpy(), \
                        #             'ori': R.from_quat(observer_data[['Controller_qx', 'Controller_qy', 'Controller_qz', 'Controller_qw']].to_numpy())}, \
                        'KineticsObserver': {  'pos': observer_data[['KO_posW_tx', 'KO_posW_ty', 'KO_posW_tz']].to_numpy(), \
                                                'ori': R.from_quat(observer_data[['KO_posW_qx', 'KO_posW_qy', 'KO_posW_qz', 'KO_posW_qw']].to_numpy()), \
                                                'linVel': observer_data[['KO_velW_vx', 'KO_velW_vy', 'KO_velW_vz']].to_numpy(), \
                                                'angVel': observer_data[['KO_velW_wx', 'KO_velW_wy', 'KO_velW_wz']].to_numpy()}, \
                        'KO_ZPC': {'pos': observer_data[['KO_ZPC_posW_tx', 'KO_ZPC_posW_ty', 'KO_ZPC_posW_tz']].to_numpy(), \
                                    'ori': R.from_quat(observer_data[['KO_ZPC_posW_qx', 'KO_ZPC_posW_qy', 'KO_ZPC_posW_qz', 'KO_ZPC_posW_qw']].to_numpy()), \
                                    'linVel': observer_data[['KO_ZPC_velW_vx', 'KO_ZPC_velW_vy', 'KO_ZPC_velW_vz']].to_numpy(), \
                                    'angVel': observer_data[['KO_ZPC_velW_wx', 'KO_ZPC_velW_wy', 'KO_ZPC_velW_wz']].to_numpy()}, \
                                        
                        'Hartley': {'pos': observer_data[['Hartley_Position_x', 'Hartley_Position_y', 'Hartley_Position_z']].to_numpy(), \
                                    'ori': R.from_quat(observer_data[['Hartley_Orientation_x', 'Hartley_Orientation_y', 'Hartley_Orientation_z', 'Hartley_Orientation_w']].to_numpy()), \
                                    'linVel': None, \
                                    'angVel': None}
                        }
    
    # Velocity of Hartley (different as we already have the velocity of the IMU)
    # estimated velocity
    linVelImu_Hartley_overlap = observer_data[['Hartley_IMU_Velocity_x', 'Hartley_IMU_Velocity_y', 'Hartley_IMU_Velocity_z']].to_numpy()
    quaternionsHartley_fb_overlap = observer_data[['Hartley_Orientation_x', 'Hartley_Orientation_y', 'Hartley_Orientation_z', 'Hartley_Orientation_w']].to_numpy()
    rHartley_fb_overlap = R.from_quat(quaternionsHartley_fb_overlap)
    rWorldImuHartley_overlap = rHartley_fb_overlap * rImuFb_overlap.inv()
    locVelHartley_imu_estim = rWorldImuHartley_overlap.apply(linVelImu_Hartley_overlap, inverse=True)
    estimatorsPoses["Hartley"]["linVel"] = locVelHartley_imu_estim

    # Velocity of the mocap (different as we only have the position)
    posMocap_overlap = observer_data[['Mocap_pos_x', 'Mocap_pos_y', 'Mocap_pos_z']].to_numpy()
    quaternionsMocap_overlap = observer_data[['Mocap_ori_x', 'Mocap_ori_y', 'Mocap_ori_z', 'Mocap_ori_w']].to_numpy()
    rMocap_overlap = R.from_quat(quaternionsMocap_overlap)
    posMocap_imu_overlap = posMocap_overlap + rMocap_overlap.apply(posFbImu_overlap)
    velMocap_imu_overlap = np.diff(posMocap_imu_overlap, axis=0)/timeStep_s
    velMocap_imu_overlap = np.vstack((zeros_row,velMocap_imu_overlap))
    rWorldImuMocap_overlap = rMocap_overlap * rImuFb_overlap.inv()
    locVelMocap_imu_estim = rWorldImuMocap_overlap.apply(velMocap_imu_overlap, inverse=True)
    b, a = butter(2, 0.15, analog=False)
    locVelMocap_imu_estim = filtfilt(b, a, locVelMocap_imu_estim, axis=0)
    estimatorsPoses["Mocap"]["linVel"] = locVelMocap_imu_estim        

    def computeObserverLocVel(observerName):
        linVelObserver_imu_overlap = estimatorsPoses[observerName]["linVel"] + np.cross(estimatorsPoses[observerName]["angVel"], estimatorsPoses[observerName]["ori"].apply(posFbImu_overlap)) + estimatorsPoses[observerName]["ori"].apply(linVelFbImu_overlap)

        rWorldImuObserver_overlap = estimatorsPoses[observerName]["ori"] * rImuFb_overlap.inv()
        locVelObserver_imu_estim = rWorldImuObserver_overlap.apply(linVelObserver_imu_overlap, inverse=True)
        estimatorsPoses[observerName]["linVel"] = locVelObserver_imu_estim

    for estimator in estimators:
        if estimator in estimator_plot_args and estimator in estimatorsPoses.keys() and estimator not in ["Mocap", "Hartley"]:
            computeObserverLocVel(estimator)

    def plotPoseAndVel(observerName):
        color_Observer = f'rgba({colors[observerName][0]}, {colors[observerName][1]}, {colors[observerName][2]}, 1)'
        figPoseVel.add_trace(
            go.Scatter(
                    x=observer_data["t"],
                    y=estimatorsPoses[observerName]["pos"][:, 0],
                    mode="lines",showlegend= False,
                    line=dict(width=estimator_plot_args[observerName]["lineWidth"], color=color_Observer)
            ),
            row=1,
            col=1,
            )
        figPoseVel.add_trace(
            go.Scatter(
                    x=observer_data["t"],
                    y=estimatorsPoses[observerName]["pos"][:, 1],
                    mode="lines",showlegend= False,
                    line=dict(width=estimator_plot_args[observerName]["lineWidth"], color=color_Observer)
            ),
            row=2,
            col=1,
            )
        figPoseVel.add_trace(
            go.Scatter(
                    x=observer_data["t"],
                    y=estimatorsPoses[observerName]["pos"][:, 2],
                    mode="lines",showlegend= False,
                    line=dict(width=estimator_plot_args[observerName]["lineWidth"], color=color_Observer)
            ),
            row=3,
            col=1,
            )        
        
        
        def continuous_euler(angles):
            continuous_angles = np.empty_like(angles)
            continuous_angles[0] = angles[0]
            for i in range(1, len(angles)):
                diff = angles[i] - angles[i-1]
                # Check each element of the diff array
                for j in range(len(diff)):
                    if diff[j] > np.pi:
                        diff[j] -= 2*np.pi
                    elif diff[j] < -np.pi:
                        diff[j] += 2*np.pi
                continuous_angles[i] = continuous_angles[i-1] + diff
            return continuous_angles
        
        worldObserverOri_euler = estimatorsPoses[observerName]["ori"].as_euler('xyz')
        worldObserverOri_euler_continuous = np.degrees(continuous_euler(worldObserverOri_euler))
        figPoseVel.add_trace(
            go.Scatter(
                    x=observer_data["t"],
                    y=worldObserverOri_euler_continuous[:, 0],
                    mode="lines",showlegend= False,
                    line=dict(width=estimator_plot_args[observerName]["lineWidth"], color=color_Observer)
            ),
            row=1,
            col=2,
            )
        figPoseVel.add_trace(
            go.Scatter(
                    x=observer_data["t"],
                    y=worldObserverOri_euler_continuous[:, 1],
                    mode="lines",showlegend= False,
                    line=dict(width=estimator_plot_args[observerName]["lineWidth"], color=color_Observer)
            ),
            row=2,
            col=2,
            )
        figPoseVel.add_trace(
            go.Scatter(
                    x=observer_data["t"],
                    y=worldObserverOri_euler_continuous[:, 2],
                    mode="lines",showlegend= False,
                    line=dict(width=estimator_plot_args[observerName]["lineWidth"], color=color_Observer)
            ),
            row=3,
            col=2,
            )


        figPoseVel.add_trace(
            go.Scatter(
                    x=observer_data["t"],
                    y=estimatorsPoses[observerName]["linVel"][:, 0],
                    mode="lines",showlegend= False,
                    line=dict(width=estimator_plot_args[observerName]["lineWidth"], color=color_Observer)
            ),
            row=1,
            col=3,
            )
        figPoseVel.add_trace(
            go.Scatter(
                    x=observer_data["t"],
                    y=estimatorsPoses[observerName]["linVel"][:, 1],
                    mode="lines",showlegend= False,
                    line=dict(width=estimator_plot_args[observerName]["lineWidth"], color=color_Observer)
            ),
            row=2,
            col=3,
            )
        figPoseVel.add_trace(
            go.Scatter(
                    x=observer_data["t"],
                    y=estimatorsPoses[observerName]["linVel"][:, 2],
                    mode="lines",showlegend= False,
                    line=dict(width=estimator_plot_args[observerName]["lineWidth"], color=color_Observer)
            ),
            row=3,
            col=3,
            )


        figPoseVel.add_trace(go.Scatter(
            x=[None],  # Dummy x value
            y=[None],  # Dummy y value
            mode="lines",
            marker=dict(size=10, color=color_Observer),
            name=f"{estimator_plot_args[observerName]['name']}"
            ))
    
    for estimator in estimators:
        if estimator in estimator_plot_args and estimator in estimatorsPoses.keys():
            plotPoseAndVel(estimator)

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

    # y_limits_x = calculate_limits(HartleyBias[:, 0], KineticsBias[:, 0], trueBias[:, 0])
    # y_limits_y = calculate_limits(HartleyBias[:, 1], KineticsBias[:, 1], trueBias[:, 1])
    # y_limits_z = calculate_limits(HartleyBias[:, 2], KineticsBias[:, 2], trueBias[:, 2])

    # Create the figure
    


    # Apply calculated y-axis limits
    # figPoseVel.update_yaxes(range=y_limits_x, row=1, col=1)
    # figPoseVel.update_yaxes(range=y_limits_y, row=2, col=1)
    # figPoseVel.update_yaxes(range=y_limits_z, row=3, col=1)

    # Update layout

    figPoseVel.update_yaxes(title=dict(text="Translation x (m)", standoff=10), row=1, col=1)
    figPoseVel.update_yaxes(title=dict(text="Translation y (m)", standoff=10), row=2, col=1)
    figPoseVel.update_yaxes(title=dict(text="Translation z (m)", standoff=10), row=3, col=1)
    figPoseVel.update_yaxes(title=dict(text="Roll (°)", standoff=5), row=1, col=2)
    figPoseVel.update_yaxes(title=dict(text="Pitch (°)", standoff=5), row=2, col=2)
    figPoseVel.update_yaxes(title=dict(text="Yaw (°)", standoff=5), row=3, col=2)
    figPoseVel.update_yaxes(title=dict(text="Velocity x (m.s⁻¹)", standoff=5), row=1, col=3)
    figPoseVel.update_yaxes(title=dict(text="Velocity y (m.s⁻¹)", standoff=5), row=2, col=3)
    figPoseVel.update_yaxes(title=dict(text="Velocity z (m.s⁻¹)", standoff=5), row=3, col=3)


    figPoseVel.update_xaxes(title_text="Time (s)", row=3, col=1)
    figPoseVel.update_xaxes(title_text="Time (s)", row=3, col=2)
    figPoseVel.update_xaxes(title_text="Time (s)", row=3, col=3)

                 

    figPoseVel.update_layout(
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
            margin=dict(l=0.0,r=0.0,b=0.0,t=0.0)
            ,autosize=True  # Automatically adjusts the figure size
    )

    
    # Show the plot
    figPoseVel.show()

    figPoseVel.write_image(f'/tmp/poseAndVel.pdf')