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

def plotPoseVel(estimators, path = default_path, colors = None):
    estimators.reverse()
    figPoseVel = make_subplots(
    rows=3, cols=3, shared_xaxes=True, vertical_spacing=0.05, horizontal_spacing=0.09, insets=[dict(cell=(1,1), l=0.15, w= 0.30, b= 0.25, h= 0.35), dict(cell=(2,1), l=0.15, w= 0.30, b= 0.25, h= 0.35), dict(cell=(3,1), l=0.15, w= 0.30, b= 0.55, h= 0.35), dict(cell=(3,2), l=0.15, w= 0.35, b= 0.15, h= 0.45), dict(cell=(1,3), l=0.13, w= 0.18, b= 0.55, h= 0.45), dict(cell=(2,3), l=0.13, w= 0.18, b= 0.55, h= 0.45), dict(cell=(3,3), l=0.13, w= 0.18, b= 0.55, h= 0.45), dict(cell=(2,2), l=0.13, w= 0.25, b= 0.15, h= 0.45)]
    )

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
            font = dict(family = 'Times New Roman', size=10, color="black"),
            margin=dict(l=0.0,r=0.0,b=0.0,t=0.0)
            ,autosize=True  # Automatically adjusts the figure size
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
    estimatorsPoses["Hartley"]["ori"] = estimatorsPoses["Hartley"]["ori"].as_euler('xyz')
    estimatorsPoses["Hartley"]["ori"] = np.degrees(continuous_euler(estimatorsPoses["Hartley"]["ori"]))
    

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
    estimatorsPoses["Mocap"]["ori"] = estimatorsPoses["Mocap"]["ori"].as_euler('xyz')
    estimatorsPoses["Mocap"]["ori"] = np.degrees(continuous_euler(estimatorsPoses["Mocap"]["ori"]))

    

    index_t_z_138 = 27400
    index_t_z_160 = 35800 

    index_t_yaw_200 = 40000
    index_t_yaw_240 = 48001

    index_t_vel_139_5 = 27900
    index_t_vel_141_5 = 28300

    positions = estimatorsPoses["Mocap"]["pos"][index_t_z_138:index_t_z_160 + 1]

    # Initialize cumulative distance
    cumulative_distance = 0.0

    # Iterate over consecutive pairs of points
    for i in range(len(positions) - 1):
        # Get current and next position (only x and y components)
        pos_current = positions[i][:2]  # Take x and y components
        pos_next = positions[i + 1][:2]  # Take x and y components
        
        # Compute the 2D distance between consecutive points
        distance = np.sqrt((pos_next[0] - pos_current[0])**2 + (pos_next[1] - pos_current[1])**2)
        
        # Add to cumulative distance
        cumulative_distance += distance

    print(f"Cumulative 2D Distance along x and y: {cumulative_distance}")


    rect_lims = {"pos_x": [None, None, None, None], "pos_y": [None, None, None, None], "pos_z": [None, None, None, None], "pitch": [None, None, None, None], "yaw": [None, None, None, None], "vel_x": [None, None, None, None], "vel_y": [None, None, None, None], "vel_z": [None, None, None, None]}
    
    def computeObserverLocVel(observerName):
        linVelObserver_imu_overlap = estimatorsPoses[observerName]["linVel"] + np.cross(estimatorsPoses[observerName]["angVel"], estimatorsPoses[observerName]["ori"].apply(posFbImu_overlap)) + estimatorsPoses[observerName]["ori"].apply(linVelFbImu_overlap)

        rWorldImuObserver_overlap = estimatorsPoses[observerName]["ori"] * rImuFb_overlap.inv()
        locVelObserver_imu_estim = rWorldImuObserver_overlap.apply(linVelObserver_imu_overlap, inverse=True)
        estimatorsPoses[observerName]["linVel"] = locVelObserver_imu_estim

        estimatorsPoses[observerName]["ori"] = estimatorsPoses[observerName]["ori"].as_euler('xyz')
        estimatorsPoses[observerName]["ori"] = np.degrees(continuous_euler(estimatorsPoses[observerName]["ori"]))

    for estimator in estimators:
        if estimator in estimator_plot_args and estimator in estimatorsPoses.keys():
            if estimator not in ["Mocap", "Hartley"]:
                computeObserverLocVel(estimator)

            x_min_z = observer_data["t"][index_t_z_138]
            x_max_z = observer_data["t"][index_t_z_160]

            x_min_pitch = observer_data["t"][index_t_z_138]
            x_max_pitch = observer_data["t"][index_t_z_160]

            x_min_yaw = observer_data["t"][index_t_yaw_200]
            x_max_yaw = observer_data["t"][index_t_yaw_240]

            x_min_vel = observer_data["t"][index_t_vel_139_5]
            x_max_vel = observer_data["t"][index_t_vel_141_5]

            y_min_pos_x = np.min(estimatorsPoses[estimator]["pos"][index_t_yaw_200:index_t_yaw_240, 0])
            y_max_pos_x = np.max(estimatorsPoses[estimator]["pos"][index_t_yaw_200:index_t_yaw_240, 0])
            y_min_pos_y = np.min(estimatorsPoses[estimator]["pos"][index_t_yaw_200:index_t_yaw_240, 1])
            y_max_pos_y = np.max(estimatorsPoses[estimator]["pos"][index_t_yaw_200:index_t_yaw_240, 1])
            y_min_pos_z = np.min(estimatorsPoses[estimator]["pos"][index_t_z_138:index_t_z_160, 2])
            y_max_pos_z = np.max(estimatorsPoses[estimator]["pos"][index_t_z_138:index_t_z_160, 2])
            
            y_min_pitch = np.min(estimatorsPoses[estimator]["ori"][index_t_z_138:index_t_z_160, 1])
            y_max_pitch = np.max(estimatorsPoses[estimator]["ori"][index_t_z_138:index_t_z_160, 1])
            y_min_yaw = np.min(estimatorsPoses[estimator]["ori"][index_t_yaw_200:index_t_yaw_240, 2])
            y_max_yaw = np.max(estimatorsPoses[estimator]["ori"][index_t_yaw_200:index_t_yaw_240, 2])

            y_min_vel_x = np.min(estimatorsPoses[estimator]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 0])
            y_max_vel_x = np.max(estimatorsPoses[estimator]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 0])
            y_min_vel_y = np.min(estimatorsPoses[estimator]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 1])
            y_max_vel_y = np.max(estimatorsPoses[estimator]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 1])
            y_min_vel_z = np.min(estimatorsPoses[estimator]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 2])
            y_max_vel_z = np.max(estimatorsPoses[estimator]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 2])

            # Update global limits
            rect_lims["pos_x"][0] = x_min_yaw if rect_lims["pos_x"][0] is None else min(rect_lims["pos_x"][0], x_min_yaw)
            rect_lims["pos_x"][1] = x_max_yaw if rect_lims["pos_x"][1] is None else max(rect_lims["pos_x"][1], x_max_yaw)
            rect_lims["pos_x"][2] = y_min_pos_x if rect_lims["pos_x"][2] is None else min(rect_lims["pos_x"][2], y_min_pos_x)
            rect_lims["pos_x"][3] = y_max_pos_x if rect_lims["pos_x"][3] is None else max(rect_lims["pos_x"][3], y_max_pos_x)

            rect_lims["pos_y"][0] = x_min_yaw if rect_lims["pos_y"][0] is None else min(rect_lims["pos_y"][0], x_min_yaw)
            rect_lims["pos_y"][1] = x_max_yaw if rect_lims["pos_y"][1] is None else max(rect_lims["pos_y"][1], x_max_yaw)
            rect_lims["pos_y"][2] = y_min_pos_y if rect_lims["pos_y"][2] is None else min(rect_lims["pos_y"][2], y_min_pos_y)
            rect_lims["pos_y"][3] = y_max_pos_y if rect_lims["pos_y"][3] is None else max(rect_lims["pos_y"][3], y_max_pos_y)

            rect_lims["pos_z"][0] = x_min_z if rect_lims["pos_z"][0] is None else min(rect_lims["pos_z"][0], x_min_z)
            rect_lims["pos_z"][1] = x_max_z if rect_lims["pos_z"][1] is None else max(rect_lims["pos_z"][1], x_max_z)
            rect_lims["pos_z"][2] = y_min_pos_z if rect_lims["pos_z"][2] is None else min(rect_lims["pos_z"][2], y_min_pos_z)
            rect_lims["pos_z"][3] = y_max_pos_z if rect_lims["pos_z"][3] is None else max(rect_lims["pos_z"][3], y_max_pos_z)

            rect_lims["pitch"][0] = x_min_pitch if rect_lims["pitch"][0] is None else min(rect_lims["pitch"][0], x_min_pitch)
            rect_lims["pitch"][1] = x_max_pitch if rect_lims["pitch"][1] is None else max(rect_lims["pitch"][1], x_max_pitch)
            rect_lims["pitch"][2] = y_min_pitch if rect_lims["pitch"][2] is None else min(rect_lims["pitch"][2], y_min_pitch)
            rect_lims["pitch"][3] = y_max_pitch if rect_lims["pitch"][3] is None else max(rect_lims["pitch"][3], y_max_pitch) 

            rect_lims["yaw"][0] = x_min_yaw if rect_lims["yaw"][0] is None else min(rect_lims["yaw"][0], x_min_yaw)
            rect_lims["yaw"][1] = x_max_yaw if rect_lims["yaw"][1] is None else max(rect_lims["yaw"][1], x_max_yaw)
            rect_lims["yaw"][2] = y_min_yaw if rect_lims["yaw"][2] is None else min(rect_lims["yaw"][2], y_min_yaw)
            rect_lims["yaw"][3] = y_max_yaw if rect_lims["yaw"][3] is None else max(rect_lims["yaw"][3], y_max_yaw) 

            rect_lims["vel_x"][0] = x_min_vel if rect_lims["vel_x"][0] is None else min(rect_lims["vel_x"][0], x_min_vel)
            rect_lims["vel_x"][1] = x_max_vel if rect_lims["vel_x"][1] is None else max(rect_lims["vel_x"][1], x_max_vel)
            rect_lims["vel_x"][2] = y_min_vel_x if rect_lims["vel_x"][2] is None else min(rect_lims["vel_x"][2], y_min_vel_x)
            rect_lims["vel_x"][3] = y_max_vel_x if rect_lims["vel_x"][3] is None else max(rect_lims["vel_x"][3], y_max_vel_x)

            rect_lims["vel_y"][0] = x_min_vel if rect_lims["vel_y"][0] is None else min(rect_lims["vel_y"][0], x_min_vel)
            rect_lims["vel_y"][1] = x_max_vel if rect_lims["vel_y"][1] is None else max(rect_lims["vel_y"][1], x_max_vel)
            rect_lims["vel_y"][2] = y_min_vel_y if rect_lims["vel_y"][2] is None else min(rect_lims["vel_y"][2], y_min_vel_y)
            rect_lims["vel_y"][3] = y_max_vel_y if rect_lims["vel_y"][3] is None else max(rect_lims["vel_y"][3], y_max_vel_y)

            rect_lims["vel_z"][0] = x_min_vel if rect_lims["vel_z"][0] is None else min(rect_lims["vel_z"][0], x_min_vel)
            rect_lims["vel_z"][1] = x_max_vel if rect_lims["vel_z"][1] is None else max(rect_lims["vel_z"][1], x_max_vel)
            rect_lims["vel_z"][2] = y_min_vel_z if rect_lims["vel_z"][2] is None else min(rect_lims["vel_z"][2], y_min_vel_z)
            rect_lims["vel_z"][3] = y_max_vel_z if rect_lims["vel_z"][3] is None else max(rect_lims["vel_z"][3], y_max_vel_z)


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
        
        
        figPoseVel.add_trace(
            go.Scatter(
                    x=observer_data["t"],
                    y=estimatorsPoses[observerName]["ori"][:, 0],
                    mode="lines",showlegend= False,
                    line=dict(width=estimator_plot_args[observerName]["lineWidth"], color=color_Observer)
            ),
            row=1,
            col=2,
            )

        # Add the inset plot as an additional trace
        figPoseVel.add_trace(
            go.Scatter(
                x=observer_data["t"][index_t_yaw_200:index_t_yaw_240],
                y=estimatorsPoses[observerName]["pos"][index_t_yaw_200:index_t_yaw_240, 0],
                mode='lines',
                showlegend= False,
                line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                xaxis='x10', 
                yaxis='y10'
            )
        )

        # Add a rectangle to subplot (1,1) surrounding the inset plot
        figPoseVel.add_shape(
            type="rect",
            xref="x1",  # Absolute positioning on the x-axis of subplot (3,3)
            yref="y1",  # Absolute positioning on the y-axis of subplot (3,3)
            x0=rect_lims["pos_x"][0],  # Start of x-range
            x1=rect_lims["pos_x"][1],  # End of x-range
            y0=rect_lims["pos_x"][2],  # Start of y-range
            y1=rect_lims["pos_x"][3],  # End of y-range
            line=dict(color="grey", width=1),
            layer="above"  # Ensures the rectangle appears above the plot
        )
        
        
        figPoseVel.add_trace(
            go.Scatter(
                x=observer_data["t"][index_t_yaw_200:index_t_yaw_240],
                y=estimatorsPoses[observerName]["pos"][index_t_yaw_200:index_t_yaw_240, 1],
                mode='lines',
                showlegend= False,
                line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                xaxis='x11', 
                yaxis='y11'
            )
        )
        # Add a rectangle to subplot (1,1) surrounding the inset plot
        figPoseVel.add_shape(
            type="rect",
            xref="x4",  # Absolute positioning on the x-axis of subplot (3,3)
            yref="y4",  # Absolute positioning on the y-axis of subplot (3,3)
            x0=rect_lims["pos_y"][0],  # Start of x-range
            x1=rect_lims["pos_y"][1],  # End of x-range
            y0=rect_lims["pos_y"][2],  # Start of y-range
            y1=rect_lims["pos_y"][3],  # End of y-range
            line=dict(color="grey", width=1),
            layer="above"  # Ensures the rectangle appears above the plot
        )

        figPoseVel.add_trace(
            go.Scatter(
                x=observer_data["t"][index_t_z_138:index_t_z_160],
                y=estimatorsPoses[observerName]["pos"][index_t_z_138:index_t_z_160, 2],
                mode='lines',
                showlegend= False,
                line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                xaxis='x12', 
                yaxis='y12'
            )
        )
        # Add a rectangle to subplot (1,1) surrounding the inset plot
        figPoseVel.add_shape(
            type="rect",
            xref="x7",  # Absolute positioning on the x-axis of subplot (3,3)
            yref="y7",  # Absolute positioning on the y-axis of subplot (3,3)
            x0=rect_lims["pos_z"][0],  # Start of x-range
            x1=rect_lims["pos_z"][1],  # End of x-range
            y0=rect_lims["pos_z"][2],  # Start of y-range
            y1=rect_lims["pos_z"][3],  # End of y-range
            line=dict(color="grey", width=1),
            layer="above"  # Ensures the rectangle appears above the plot
        )
        
        
        # Add the inset plot as an additional trace
        figPoseVel.add_trace(
            go.Scatter(
                x=observer_data["t"][index_t_z_138:index_t_z_160],
                y=estimatorsPoses[observerName]["ori"][index_t_z_138:index_t_z_160, 1],
                mode='lines',
                showlegend= False,
                line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                xaxis='x17', 
                yaxis='y17'
            )
        )

        print(f"{np.mean(estimatorsPoses['Hartley']['ori'][index_t_yaw_200:index_t_yaw_240, 1] - estimatorsPoses['Mocap']['ori'][index_t_yaw_200:index_t_yaw_240, 1])}")
        

        # Add the inset plot as an additional trace
        figPoseVel.add_trace(
            go.Scatter(
                x=observer_data["t"][index_t_yaw_200:index_t_yaw_240],
                y=estimatorsPoses[observerName]["ori"][index_t_yaw_200:index_t_yaw_240, 2],
                mode='lines',
                showlegend= False,
                line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                xaxis='x13', 
                yaxis='y13'
            )
        )

        # Add a rectangle to subplot (1,1) surrounding the inset plot
        figPoseVel.add_shape(
            type="rect",
            xref="x5",  # Absolute positioning on the x-axis of subplot (3,3)
            yref="y5",  # Absolute positioning on the y-axis of subplot (3,3)
            x0=rect_lims["pitch"][0],  # Start of x-range
            x1=rect_lims["pitch"][1],  # End of x-range
            y0=rect_lims["pitch"][2],  # Start of y-range
            y1=rect_lims["pitch"][3],  # End of y-range
            line=dict(color="grey", width=1),
            layer="above"  # Ensures the rectangle appears above the plot
        )
        
        # Add a rectangle to subplot (1,1) surrounding the inset plot
        figPoseVel.add_shape(
            type="rect",
            xref="x8",  # Absolute positioning on the x-axis of subplot (3,3)
            yref="y8",  # Absolute positioning on the y-axis of subplot (3,3)
            x0=rect_lims["yaw"][0],  # Start of x-range
            x1=rect_lims["yaw"][1],  # End of x-range
            y0=rect_lims["yaw"][2],  # Start of y-range
            y1=rect_lims["yaw"][3],  # End of y-range
            line=dict(color="grey", width=1),
            layer="above"  # Ensures the rectangle appears above the plot
        )

        # Add the inset plot as an additional trace
        figPoseVel.add_trace(
            go.Scatter(
                x=observer_data["t"][index_t_vel_139_5:index_t_vel_141_5],
                y=estimatorsPoses[observerName]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 0],
                mode='lines',
                showlegend= False,
                line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                xaxis='x14', 
                yaxis='y14'
            )
        )
        # Add a rectangle to subplot (7,7) surrounding the inset plot
        figPoseVel.add_shape(
            type="rect",
            xref="x3",  # Absolute positioning on the x-axis of subplot (3,3)
            yref="y3",  # Absolute positioning on the y-axis of subplot (3,3)
            x0=rect_lims["vel_x"][0],  # Start of x-range
            x1=rect_lims["vel_x"][1],  # End of x-range
            y0=rect_lims["vel_x"][2],  # Start of y-range
            y1=rect_lims["vel_x"][3],  # End of y-range
            line=dict(color="grey", width=1),
            layer="above"  # Ensures the rectangle appears above the plot
        )

        # Add the inset plot as an additional trace
        figPoseVel.add_trace(
            go.Scatter(
                x=observer_data["t"][index_t_vel_139_5:index_t_vel_141_5],
                y=estimatorsPoses[observerName]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 1],
                mode='lines',
                showlegend= False,
                line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                xaxis='x15', 
                yaxis='y15'
            )
        )
        # Add a rectangle to subplot (1,1) surrounding the inset plot
        figPoseVel.add_shape(
            type="rect",
            xref="x6",  # Absolute positioning on the x-axis of subplot (3,3)
            yref="y6",  # Absolute positioning on the y-axis of subplot (3,3)
            x0=rect_lims["vel_y"][0],  # Start of x-range
            x1=rect_lims["vel_y"][1],  # End of x-range
            y0=rect_lims["vel_y"][2],  # Start of y-range
            y1=rect_lims["vel_y"][3],  # End of y-range
            line=dict(color="grey", width=1),
            layer="above"  # Ensures the rectangle appears above the plot
        )

        # Add the inset plot as an additional trace
        figPoseVel.add_trace(
            go.Scatter(
                x=observer_data["t"][index_t_vel_139_5:index_t_vel_141_5],
                y=estimatorsPoses[observerName]["linVel"][index_t_vel_139_5:index_t_vel_141_5, 2],
                mode='lines',
                showlegend= False,
                line=dict(width=estimator_plot_args[observerName]["lineWidth"]/2, color=color_Observer),
                xaxis='x16', 
                yaxis='y16'
            )
        )
        # Add a rectangle to subplot (1,1) surrounding the inset plot
        figPoseVel.add_shape(
            type="rect",
            xref="x9",  # Absolute positioning on the x-axis of subplot (3,3)
            yref="y9",  # Absolute positioning on the y-axis of subplot (3,3)
            x0=rect_lims["vel_z"][0],  # Start of x-range
            x1=rect_lims["vel_z"][1],  # End of x-range
            y0=rect_lims["vel_z"][2],  # Start of y-range
            y1=rect_lims["vel_z"][3],  # End of y-range
            line=dict(color="grey", width=1),
            layer="above"  # Ensures the rectangle appears above the plot
        )

        
        
        figPoseVel.update_layout(
                xaxis10=dict(
                        dtick=20, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black")),
                yaxis10=dict(
                        gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"))
                )
        
        figPoseVel.update_layout(
                xaxis11=dict(
                        dtick=20, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black")),
                yaxis11=dict(
                        gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"))
                )
        
        figPoseVel.update_layout(
                xaxis12=dict(
                        dtick=10, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),),
                yaxis12=dict(
                        gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),)
                )
        
        figPoseVel.update_layout(
                xaxis13=dict(
                        dtick=20, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),),
                yaxis13=dict(
                        gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),)
                )
        
        figPoseVel.update_layout(
                xaxis14=dict(
                        dtick=1, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),),
                yaxis14=dict(
                        gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),)
                )
        
        figPoseVel.update_layout(
                xaxis15=dict(
                        dtick=1, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),),
                yaxis15=dict(
                        gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),)
                )
        
        figPoseVel.update_layout(
                xaxis16=dict(
                        dtick=1, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),),
                yaxis16=dict(
                        gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),)
                )
        
        figPoseVel.update_layout(
                xaxis17=dict(
                        dtick=10, gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),),
                yaxis17=dict(
                        gridcolor= 'lightgrey', zerolinecolor= 'lightgrey', linecolor= 'lightgrey', mirror=True, ticks='outside', showline=True, tickcolor='lightgrey', tickfont = dict(family = 'Times New Roman', size=7, color="black"),)
                )

        figPoseVel.add_trace(
            go.Scatter(
                    x=observer_data["t"],
                    y=estimatorsPoses[observerName]["ori"][:, 1],
                    mode="lines",showlegend= False,
                    line=dict(width=estimator_plot_args[observerName]["lineWidth"], color=color_Observer)
            ),
            row=2,
            col=2,
            )
        
        figPoseVel.add_trace(
            go.Scatter(
                    x=observer_data["t"],
                    y=estimatorsPoses[observerName]["ori"][:, 2],
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


    
    # Show the plot
    figPoseVel.show()

    figPoseVel.write_image(f'/tmp/poseAndVel.pdf')