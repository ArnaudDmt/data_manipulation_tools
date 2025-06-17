import math
import pickle
import signal
import sys
import numpy as np
import pandas as pd
import yaml
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from rdp import rdp

import plotly.express as px  # For color palette generation
from scipy.signal import butter,filtfilt



###############################  Main variables initialization  ###############################


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


def run(displayLogs, writeFormattedData, path_to_project, estimatorsList = None, colors = None, scriptName = "finalResults"):

    with open(f'{path_to_project}/output_data/observers_infos.yaml', 'r') as file:
        try:
            infos_yaml_str = file.read()
            infos_yamlData = yaml.safe_load(infos_yaml_str)
            timeStep_s = float(infos_yamlData.get("timeStep_s"))
        except yaml.YAMLError as exc:
            print(exc)

    data_df = pd.read_csv(f'{path_to_project}/output_data/finalDataCSV.csv', delimiter=';')

    with open(f'{path_to_project}/output_data/observers_infos.yaml', 'r') as file:
        try:
            observers_infos_str = file.read()
            observers_infos_yamlData = yaml.safe_load(observers_infos_str)
            mocapBody = observers_infos_yamlData.get('mocapBody')
        except yaml.YAMLError as exc:
            print(exc)

    if(estimatorsList == None):
        with open(f'{path_to_project}/output_data/observers_infos.yaml', 'r') as file:
            try:
                observers_infos_str = file.read()
                observers_infos_yamlData = yaml.safe_load(observers_infos_str)
                estimatorsList = set(observers_infos_yamlData.get('observers'))
            except yaml.YAMLError as exc:
                print(exc)

    if (colors == None):
        colors_t = px.colors.qualitative.Plotly  # Use Plotly's color palette
        colors_t = [px.colors.hex_to_rgb(colors_t[i]) for i in range(len(estimatorsList))]
        colors = dict.fromkeys(estimatorsList)

        for i,estimator in enumerate(colors.keys()):
            colors[estimator] = f'rgb({colors_t[i][0]}, {colors_t[i][1]}, {colors_t[i][2]})'
    else:
        for estimator in colors.keys():
            colors[estimator] = f'rgb({colors[estimator][0]}, {colors[estimator][1]}, {colors[estimator][2]})'
    


    ###############################  Computations  ###############################
    
    kinematics = dict()

    if("Mocap" in estimatorsList):
        # data_df = data_df[data_df["Mocap_datasOverlapping"] == "Datas overlap"]

        kinematics["Mocap"] = dict()
        kinematics["Mocap"][mocapBody] = dict()
        kinematics["Mocap"][mocapBody]["position"] = data_df[['Mocap_position_x', 'Mocap_position_y', 'Mocap_position_z']].to_numpy()

        # Original 2D trajectory
        positions_xy = kinematics["Mocap"][mocapBody]["position"][:, :2]

        # Apply RDP simplification
        simplified_xy = rdp(positions_xy, epsilon=0.1)

        total_distance = np.sum(np.linalg.norm(np.diff(simplified_xy, axis=0), axis=1))

        print(f"total_distance: {total_distance}")

        # Create plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=positions_xy[:, 0],
            y=positions_xy[:, 1],
            mode='lines',
            name='Original Trajectory',
            line=dict(color='blue', width=2),
            opacity=0.4
        ))

        fig.add_trace(go.Scatter(
            x=simplified_xy[:, 0],
            y=simplified_xy[:, 1],
            mode='lines+markers',
            name='Simplified Trajectory (RDP)',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=6, color='red')
        ))

        fig.update_layout(
            title='Original vs Simplified Trajectory (XY Plane)',
            xaxis_title='X [m]',
            yaxis_title='Y [m]',
            legend=dict(x=0.02, y=0.98),
            width=800,
            height=600
        )

        fig.show()

        sys.exit(1)

        kinematics["Mocap"][mocapBody]["quaternions"] = data_df[['Mocap_orientation_x', 'Mocap_orientation_y', 'Mocap_orientation_z', 'Mocap_orientation_w']].to_numpy()
        kinematics["Mocap"][mocapBody]["R"] = R.from_quat(kinematics["Mocap"][mocapBody]["quaternions"])
        euler_angles_Mocap = kinematics["Mocap"][mocapBody]["R"].as_euler('xyz')
        kinematics["Mocap"][mocapBody]["euler_angles"] = continuous_euler(euler_angles_Mocap)
    
    for estimator in estimatorsList:
        if estimator != "RI-EKF" and estimator != "Mocap":
            kinematics[estimator] = dict()
            kinematics[estimator][mocapBody] = dict()
            kinematics[estimator][mocapBody]["position"] = data_df[[estimator + '_position_x', estimator + '_position_y', estimator + '_position_z']].to_numpy()
            kinematics[estimator][mocapBody]["quaternions"] = data_df[[estimator + '_orientation_x', estimator + '_orientation_y', estimator + '_orientation_z', estimator + '_orientation_w']].to_numpy()
            rot = R.from_quat(kinematics[estimator][mocapBody]["quaternions"])
            kinematics[estimator][mocapBody]["R"] = rot
            euler_angles = rot.as_euler('xyz')
            euler_angles = continuous_euler(euler_angles)
            kinematics[estimator][mocapBody]["euler_angles"] = euler_angles

            if("Mocap" in estimatorsList):
                kinematics[estimator][mocapBody]["position"] = data_df[[estimator + '_position_x', estimator + '_position_y', estimator + '_position_z']].to_numpy()
                kinematics[estimator][mocapBody]["quaternions"] = data_df[[estimator + '_orientation_x', estimator + '_orientation_y', estimator + '_orientation_z', estimator + '_orientation_w']].to_numpy()
                kinematics[estimator][mocapBody]["R"] = R.from_quat(kinematics[estimator][mocapBody]["quaternions"])
                euler_angles = kinematics[estimator][mocapBody]["R"].as_euler('xyz')
                kinematics[estimator][mocapBody]["euler_angles"] = continuous_euler(euler_angles)

    if("RI-EKF" in estimatorsList and "Mocap" in estimatorsList):
        kinematics["RI-EKF"] = dict()
        kinematics["RI-EKF"][mocapBody] = dict()
        kinematics["RI-EKF"][mocapBody]["position"] = data_df[['RI-EKF_position_x', 'RI-EKF_position_y', 'RI-EKF_position_z']].to_numpy()
        kinematics["RI-EKF"][mocapBody]["quaternions"] = data_df[['RI-EKF_orientation_x', 'RI-EKF_orientation_y', 'RI-EKF_orientation_z', 'RI-EKF_orientation_w']].to_numpy()
        kinematics["RI-EKF"][mocapBody]["R"] = R.from_quat(kinematics["RI-EKF"][mocapBody]["quaternions"])

        euler_angles_Hartley = kinematics["RI-EKF"][mocapBody]["R"].as_euler('xyz')
        kinematics["RI-EKF"][mocapBody]["euler-angles"] = continuous_euler(euler_angles_Hartley)

        if("Mocap" in estimatorsList):
            kinematics["RI-EKF"][mocapBody]["position"] = data_df[['RI-EKF_position_x', 'RI-EKF_position_y', 'RI-EKF_position_z']].to_numpy()
            kinematics["RI-EKF"][mocapBody]["quaternions"] = data_df[['RI-EKF_orientation_x', 'RI-EKF_orientation_y', 'RI-EKF_orientation_z', 'RI-EKF_orientation_w']].to_numpy()
            kinematics["RI-EKF"][mocapBody]["R"] = R.from_quat(kinematics["RI-EKF"][mocapBody]["quaternions"])
            euler_angles_Hartley = kinematics["RI-EKF"][mocapBody]["R"].as_euler('xyz')
            kinematics["RI-EKF"][mocapBody]["euler_angles"] = continuous_euler(euler_angles_Hartley)

            kinematics["RI-EKF"]["IMU"] = dict()
            kinematics["RI-EKF"]["IMU"]["position"] = data_df[['RI-EKF_IMU_position_x', 'RI-EKF_IMU_position_y', 'RI-EKF_IMU_position_z']].to_numpy()
            kinematics["RI-EKF"]["IMU"]["quaternions"] = data_df[['RI-EKF_IMU_orientation_x', 'RI-EKF_IMU_orientation_y', 'RI-EKF_IMU_orientation_z', 'RI-EKF_IMU_orientation_w']].to_numpy()
            kinematics["RI-EKF"]["IMU"]["R"] = R.from_quat(kinematics["RI-EKF"]["IMU"]["quaternions"])
            euler_angles_Hartley = kinematics["RI-EKF"]["IMU"]["R"].as_euler('xyz')
            kinematics["RI-EKF"]["IMU"]["euler_angles"] = continuous_euler(euler_angles_Hartley)
            
            
            posImuFb = data_df[['HartleyIEKF_imuFbKine_position_x', 'HartleyIEKF_imuFbKine_position_y', 'HartleyIEKF_imuFbKine_position_z']].to_numpy()
            quaternions_rImuFb = data_df[['HartleyIEKF_imuFbKine_ori_x', 'HartleyIEKF_imuFbKine_ori_y', 'HartleyIEKF_imuFbKine_ori_z', 'HartleyIEKF_imuFbKine_ori_w']].to_numpy()
            rImuFb = R.from_quat(quaternions_rImuFb)

            linVelImuFb = data_df[['HartleyIEKF_imuFbKine_linVel_x', 'HartleyIEKF_imuFbKine_linVel_y', 'HartleyIEKF_imuFbKine_linVel_z']].to_numpy()
            angVelImuFb = data_df[['HartleyIEKF_imuFbKine_angVel_x', 'HartleyIEKF_imuFbKine_angVel_y', 'HartleyIEKF_imuFbKine_angVel_z']].to_numpy()
            posFbImu = - rImuFb.apply(posImuFb, inverse=True)
            linVelFbImu = rImuFb.apply(np.cross(angVelImuFb, posImuFb), inverse=True) - rImuFb.apply(linVelImuFb, inverse=True)



    ###################################################### Format data ######################################################

    if(writeFormattedData):
        if("Mocap" in estimatorsList):
            dfMocapPose = pd.DataFrame(columns=['#', 'timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
            dfMocapPose['timestamp'] = data_df['t']
            dfMocapPose['tx'] = kinematics["Mocap"][mocapBody]["position"][:,0]
            dfMocapPose['ty'] = kinematics["Mocap"][mocapBody]["position"][:,1]
            dfMocapPose['tz'] = kinematics["Mocap"][mocapBody]["position"][:,2]
            dfMocapPose['qx'] = kinematics["Mocap"][mocapBody]["quaternions"][:,0]
            dfMocapPose['qy'] = kinematics["Mocap"][mocapBody]["quaternions"][:,1]
            dfMocapPose['qz'] = kinematics["Mocap"][mocapBody]["quaternions"][:,2]
            dfMocapPose['qw'] = kinematics["Mocap"][mocapBody]["quaternions"][:,3]

            # dfMocapPose = dfMocapPose[data_df["Mocap_datasOverlapping"] == "Datas overlap"]

            txtOutput = f'{path_to_project}/output_data/formattedMocap_Traj.txt'
            dfMocapPose.to_csv(txtOutput, header=None, index=None, sep=' ')

            line = '# timestamp tx ty tz qx qy qz qw' 
            with open(txtOutput, 'r+') as file: 
                file_data = file.read() 
                file.seek(0, 0) 
                file.write(line + '\n' + file_data) 

            d = {'x': kinematics["Mocap"][mocapBody]["position"][:, 0], 'y': kinematics["Mocap"][mocapBody]["position"][:, 1], 'z': kinematics["Mocap"][mocapBody]["position"][:, 2]}
            with open(f'{path_to_project}/output_data/mocap_x_y_z_traj.pickle', 'wb') as f:
                pickle.dump(d, f)

            for estimator in estimatorsList:
                dfPose = pd.DataFrame(columns=['#', 'timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
                dfPose['timestamp'] = data_df['t']
                dfPose['tx'] = kinematics[estimator][mocapBody]["position"][:,0]
                dfPose['ty'] = kinematics[estimator][mocapBody]["position"][:,1]
                dfPose['tz'] = kinematics[estimator][mocapBody]["position"][:,2]
                dfPose['qx'] = kinematics[estimator][mocapBody]["quaternions"][:,0]
                dfPose['qy'] = kinematics[estimator][mocapBody]["quaternions"][:,1]
                dfPose['qz'] = kinematics[estimator][mocapBody]["quaternions"][:,2]
                dfPose['qw'] = kinematics[estimator][mocapBody]["quaternions"][:,3]

                txtOutput = f'{path_to_project}/output_data/formatted_{estimator}_Traj.txt'
                dfPose.to_csv(txtOutput, header=None, index=None, sep=' ')

                line = '# timestamp tx ty tz qx qy qz qw' 
                with open(txtOutput, 'r+') as file: 
                    file_data = file.read() 
                    file.seek(0, 0) 
                    file.write(line + '\n' + file_data) 

                d = {'x': dfPose['tx'], 'y': dfPose['ty'], 'z': dfPose['tz']}
                with open(f'{path_to_project}/output_data/{estimator}_x_y_z_traj.pickle', 'wb') as f:
                    pickle.dump(d, f)


    ###################################################### Display logs ######################################################


    if(displayLogs):
        # Create the figure
        fig_pose = go.Figure()
        for estimator in estimatorsList:
            # Add traces for each plot
            fig_pose.add_trace(go.Scatter(x=data_df['t'], y=kinematics[estimator][mocapBody]["position"][:, 0], mode='lines', line=dict(color = colors[estimator]), name=f'{estimator}_Position_x'))
            fig_pose.add_trace(go.Scatter(x=data_df['t'], y=kinematics[estimator][mocapBody]["position"][:, 1], mode='lines', line=dict(color = colors[estimator]), name=f'{estimator}_Position_y'))
            fig_pose.add_trace(go.Scatter(x=data_df['t'], y=kinematics[estimator][mocapBody]["position"][:, 2], mode='lines', line=dict(color = colors[estimator]), name=f'{estimator}_Position_z'))
            fig_pose.add_trace(go.Scatter(x=data_df['t'], y=kinematics[estimator][mocapBody]["euler_angles"][:, 0], mode='lines', line=dict(color = colors[estimator]), name=f'{estimator}_Roll'))
            fig_pose.add_trace(go.Scatter(x=data_df['t'], y=kinematics[estimator][mocapBody]["euler_angles"][:, 1], mode='lines', line=dict(color = colors[estimator]), name=f'{estimator}_Pitch'))
            fig_pose.add_trace(go.Scatter(x=data_df['t'], y=kinematics[estimator][mocapBody]["euler_angles"][:, 2], mode='lines', line=dict(color = colors[estimator]), name=f'{estimator}_Yaw'))


        # Update layout
        fig_pose.update_layout(
            title= f'{scriptName}: Pose over time',
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x'
        )
        # Show the interactive plot
        fig_pose.show()

        fig_traj_2d = go.Figure()

        for estimator in estimatorsList:
            fig_traj_2d.add_trace(go.Scatter(x=kinematics[estimator][mocapBody]["position"][:, 0], y=kinematics[estimator][mocapBody]["position"][:, 1], mode='lines', line=dict(color = colors[estimator]), name=f'{estimator}_2dMotion_xy'))

        # Update layout
        fig_traj_2d.update_layout(
            xaxis_title='x',
            yaxis_title='y',
            hovermode='x',
            title=f"{scriptName}: 2D trajectories"
        )
        fig_traj_2d.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )
        # Show the interactive plot
        fig_traj_2d.show()

        fig_traj_3d = go.Figure()

        for estimator in estimatorsList:
            x_min = min((kinematics[estimator][mocapBody]["position"][:, 0]).min(), (kinematics[estimator][mocapBody]["position"][:, 0]).min(), (kinematics[estimator][mocapBody]["position"][:, 0]).min())
            y_min = min((kinematics[estimator][mocapBody]["position"][:,1]).min(), (kinematics[estimator][mocapBody]["position"][:,1]).min(), (kinematics[estimator][mocapBody]["position"][:,1]).min())
            z_min = min((kinematics[estimator][mocapBody]["position"][:,2]).min(), (kinematics[estimator][mocapBody]["position"][:,2]).min(), (kinematics[estimator][mocapBody]["position"][:,2]).min())

            x_max = max((kinematics[estimator][mocapBody]["position"][:,0]).max(), (kinematics[estimator][mocapBody]["position"][:,0]).max(), (kinematics[estimator][mocapBody]["position"][:,0]).max())
            y_max = max((kinematics[estimator][mocapBody]["position"][:,1]).max(), (kinematics[estimator][mocapBody]["position"][:,1]).max(), (kinematics[estimator][mocapBody]["position"][:,1]).max())
            z_max = max((kinematics[estimator][mocapBody]["position"][:,2]).max(), (kinematics[estimator][mocapBody]["position"][:,2]).max(), (kinematics[estimator][mocapBody]["position"][:,2]).max())

            # Add traces
            fig_traj_3d.add_trace(go.Scatter3d(
                x=kinematics[estimator][mocapBody]["position"][:,0], 
                y=kinematics[estimator][mocapBody]["position"][:,1], 
                z=kinematics[estimator][mocapBody]["position"][:,2],
                mode='lines', line=dict(color = colors[estimator]),
                name=f'{estimator}'
            ))

        x_min = x_min - np.abs(x_min*0.2)
        y_min = y_min - np.abs(y_min*0.2)
        z_min = z_min - np.abs(z_min*0.2)
        x_max = x_max + np.abs(x_max*0.2)
        y_max = y_max + np.abs(y_max*0.2)
        z_max = z_max + np.abs(z_max*0.2)


        # Update layout
        fig_traj_3d.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                xaxis=dict(range=[x_min, x_max]),
                yaxis=dict(range=[y_min, y_max]),
                zaxis=dict(range=[z_min, z_max]),
                aspectmode='data'
            ),
            legend=dict(
                x=0,
                y=1
            )
            , title=f"{scriptName}: 3D trajectories"
        )

        # Show the plot
        fig_traj_3d.show()


        fig_gyroBias = go.Figure()

        # Get the last time in data_df['t']
        last_time = data_df['t'].max()

        # Filter the data for the first 3 seconds
        df_first_3s = data_df[(data_df['t'] >= 0) & (data_df['t'] <= 3)]

        # Compute the average bias over the first 3 seconds
        average_bias_x_3 = df_first_3s['Accelerometer_angularVelocity_x'].mean()
        average_bias_x_tuple_3 = tuple(average_bias_x_3 for _ in range(len(data_df)))
        average_bias_y_3 = df_first_3s['Accelerometer_angularVelocity_y'].mean()
        average_bias_y_tuple_3 = tuple(average_bias_y_3 for _ in range(len(data_df)))
        average_bias_z_3 = df_first_3s['Accelerometer_angularVelocity_z'].mean()
        average_bias_z_tuple_3 = tuple(average_bias_z_3 for _ in range(len(data_df)))

        # Filter the data for the last 3 seconds
        df_last_3s = data_df[(data_df['t'] >= last_time - 3) & (data_df['t'] <= last_time)]

        # Compute the average bias over the last 3 seconds
        average_bias_x_last = df_last_3s['Accelerometer_angularVelocity_x'].mean()
        average_bias_x_tuple_last = tuple(average_bias_x_last for _ in range(len(data_df)))
        average_bias_y_last = df_last_3s['Accelerometer_angularVelocity_y'].mean()
        average_bias_y_tuple_last = tuple(average_bias_y_last for _ in range(len(data_df)))
        average_bias_z_last = df_last_3s['Accelerometer_angularVelocity_z'].mean()
        average_bias_z_tuple_last = tuple(average_bias_z_last for _ in range(len(data_df)))

        # Plotting the original data and the computed biases
        fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=data_df['Accelerometer_angularVelocity_x'], mode='lines', name='measured_angVel_x'))
        fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=data_df['Accelerometer_angularVelocity_y'], mode='lines', name='measured_angVel_y'))
        fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=data_df['Accelerometer_angularVelocity_z'], mode='lines', name='measured_angVel_z'))
        fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=average_bias_x_tuple_3, mode='lines', name='measured_GyroBias_beginning_x'))
        fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=average_bias_y_tuple_3, mode='lines', name='measured_GyroBias_beginning_y'))
        fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=average_bias_z_tuple_3, mode='lines', name='measured_GyroBias_beginning_z'))
        fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=average_bias_x_tuple_last, mode='lines', name='measured_GyroBias_end_x'))
        fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=average_bias_y_tuple_last, mode='lines', name='measured_GyroBias_end_y'))
        fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=average_bias_z_tuple_last, mode='lines', name='measured_GyroBias_end_z'))

        obs = []
        with open(f'{path_to_project}/../../observersInfos.yaml', 'r') as file:
            try:
                observersInfos_str = file.read()
                observersInfos_yamlData = yaml.safe_load(observersInfos_str)
                for observer in observersInfos_yamlData['observers']:
                    if 'IMU' in observer['kinematics'] and 'gyroBias' in observer['kinematics']['IMU']:
                        obs.append(observer["abbreviation"])
            except yaml.YAMLError as exc:
                print(exc)

        for estimator in estimatorsList:
            if estimator in obs:
                fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=data_df[f'{estimator}_IMU_gyroBias_x'], mode='lines', name=f'{estimator}_gyroBias_x'))
                fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=data_df[f'{estimator}_IMU_gyroBias_y'], mode='lines', name=f'{estimator}_gyroBias_y'))
                fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=data_df[f'{estimator}_IMU_gyroBias_z'], mode='lines', name=f'{estimator}_gyroBias_z'))
        # Update layout
        fig_gyroBias.update_layout(
            title='Gyrometer biases',
            xaxis_title='x',
            yaxis_title='y',
            hovermode='x'
        )

        # Show the interactive plot
        fig_gyroBias.show()



    ###############################  Criterias based on the local linear velocity  ###############################

    with open(f'{path_to_project}/projectConfig.yaml', 'r') as file:
        try:
            projectConfig_str = file.read()
            projectConfig_yamlData = yaml.safe_load(projectConfig_str)
            vel_eval_body = projectConfig_yamlData.get('Body_vel_eval')
        except yaml.YAMLError as exc:
            print(exc)
    
    d = dict()
    zeros_row = np.zeros((1, 3))
    
    if("Mocap" in estimatorsList):
        kinematics['Mocap']['IMU'] = dict()
        kinematics['Mocap'][mocapBody]['linVel'] = np.diff(kinematics['Mocap'][mocapBody]['position'], axis=0)/timeStep_s
        kinematics['Mocap'][mocapBody]['linVel'] = np.vstack((zeros_row,kinematics['Mocap'][mocapBody]['linVel']))

        kinematics['Mocap']['IMU']['position'] =  kinematics['Mocap'][mocapBody]['position'] + kinematics['Mocap'][mocapBody]['R'].apply(posFbImu)
        kinematics['Mocap']['IMU']['R'] = kinematics['Mocap'][mocapBody]['R'] * rImuFb.inv()

        kinematics['Mocap']['IMU']['linVel'] = np.diff(kinematics['Mocap']['IMU']['position'], axis=0)/timeStep_s
        kinematics['Mocap']['IMU']['linVel'] = np.vstack((zeros_row,kinematics['Mocap']['IMU']['linVel']))
        
        kinematics['Mocap'][mocapBody]['locLinVel'] = kinematics['Mocap'][mocapBody]['R'].apply(kinematics['Mocap'][mocapBody]['linVel'], inverse=True)
        kinematics['Mocap']['IMU']['locLinVel'] = kinematics['Mocap']['IMU']['R'].apply(kinematics['Mocap'][mocapBody]['linVel'], inverse=True)

        #b, a = butter(2, 0.15, analog=False)

        #kinematics['Mocap'][mocapBody]['locLinVel'] = filtfilt(b, a, kinematics['Mocap'][mocapBody]['locLinVel'], axis=0)
        #kinematics['Mocap']['IMU']['locLinVel'] = filtfilt(b, a, kinematics['Mocap']['IMU']['locLinVel'], axis=0)
        d['Mocap'] = {'llve': {}, 'estimate': {}}
        d['Mocap']['llve'] = {'x': kinematics['Mocap'][mocapBody]['locLinVel'][:, 0], 'y': kinematics['Mocap'][mocapBody]['locLinVel'][:, 1], 'z': kinematics['Mocap'][mocapBody]['locLinVel'][:, 2]}
        d['Mocap']['estimate'] = {'x': kinematics['Mocap'][vel_eval_body]['locLinVel'][:, 0], 'y': kinematics['Mocap'][vel_eval_body]['locLinVel'][:, 1], 'z': kinematics['Mocap'][vel_eval_body]['locLinVel'][:, 2]}
        with open(f'{path_to_project}/output_data/mocap_loc_vel.pickle', 'wb') as f:
            pickle.dump(d['Mocap'], f)


        if(displayLogs):
            figLocLinVels = go.Figure()
            figLocLinVels.add_trace(go.Scatter(x=data_df["t"], y=kinematics['Mocap'][mocapBody]['locLinVel'][:,0], mode='lines', line=dict(color = colors["Mocap"]), name='locLinvel_Mocap_x'))
            figLocLinVels.add_trace(go.Scatter(x=data_df["t"], y=kinematics['Mocap'][mocapBody]['locLinVel'][:,1], mode='lines', line=dict(color = colors["Mocap"]), name='locLinvel_Mocap_y'))
            figLocLinVels.add_trace(go.Scatter(x=data_df["t"], y=kinematics['Mocap'][mocapBody]['locLinVel'][:,2], mode='lines', line=dict(color = colors["Mocap"]), name='locLinvel_Mocap_z'))

            figLocLinVels.update_layout(title=f"{scriptName}: Linear velocities")  

        for estimator in estimatorsList:
            if estimator != "RI-EKF":
                d[estimator] = dict()

                kinematics[estimator][mocapBody]['llve'] = np.diff(kinematics['Mocap'][mocapBody]['position'], axis=0)/timeStep_s
                kinematics[estimator][mocapBody]['llve'] = np.vstack((zeros_row,kinematics[estimator][mocapBody]['llve']))
                kinematics[estimator][mocapBody]['llve'] = kinematics[estimator][mocapBody]["R"].apply(kinematics[estimator][mocapBody]['llve'], inverse=True)

                d[estimator]['llve'] = {'x': kinematics[estimator][mocapBody]["llve"][:, 0], 'y': kinematics[estimator][mocapBody]["llve"][:, 1], 'z': kinematics[estimator][mocapBody]["llve"][:, 2]}

                with open(f'{path_to_project}/../../observersInfos.yaml', 'r') as file:
                    try:
                        observersInfos_str = file.read()
                        observersInfos_yamlData = yaml.safe_load(observersInfos_str)
                    except yaml.YAMLError as exc:
                        print(exc)

                for observer in observersInfos_yamlData['observers']:
                    if estimator == observer["abbreviation"]:
                        if mocapBody != vel_eval_body:
                            prefix = f'{estimator}_{vel_eval_body}'
                            kinematics[estimator][vel_eval_body] = dict()
                        else:
                            prefix = f'{estimator}'
                        if f'{prefix}_locLinVel_x' in data_df.columns:
                            kinematics[estimator][vel_eval_body]["locLinVel"] = data_df[[prefix + '_locLinVel_x', prefix + '_locLinVel_y', prefix + '_locLinVel_z']].to_numpy()
                        elif f'{prefix}_linVel_x' in data_df.columns and f'{prefix}_orientation_x' in data_df.columns:
                            kinematics[estimator][vel_eval_body]["linVel"] = data_df[[prefix + '_linVel_x', prefix + '_linVel_y', prefix + '_linVel_z']].to_numpy()
                            kinematics[estimator][vel_eval_body]["locLinVel"] = kinematics[prefix][vel_eval_body]["R"].apply(kinematics[estimator][vel_eval_body]["linVel"], inverse=True)
                        else:
                            continue

                        if(displayLogs):
                            figLocLinVels.add_trace(go.Scatter(x=data_df["t"], y=kinematics[estimator][vel_eval_body]["locLinVel"][:,0], mode='lines', line=dict(color = colors[estimator]), name=f'locLinVel_{vel_eval_body}_{estimator}_x'))
                            figLocLinVels.add_trace(go.Scatter(x=data_df["t"], y=kinematics[estimator][vel_eval_body]["locLinVel"][:,1], mode='lines', line=dict(color = colors[estimator]), name=f'locLinVel_{vel_eval_body}_{estimator}_y'))
                            figLocLinVels.add_trace(go.Scatter(x=data_df["t"], y=kinematics[estimator][vel_eval_body]["locLinVel"][:,2], mode='lines', line=dict(color = colors[estimator]), name=f'locLinVel_{vel_eval_body}_{estimator}_z'))

                        d[estimator]['estimate'] = {'x': kinematics[estimator][vel_eval_body]["locLinVel"][:, 0], 'y': kinematics[estimator][vel_eval_body]["locLinVel"][:, 1], 'z': kinematics[estimator][vel_eval_body]["locLinVel"][:, 2]}
            
        if("RI-EKF" in estimatorsList):
            d['RI-EKF'] = dict()
            kinematics["RI-EKF"][mocapBody]["linVel"] = np.diff(kinematics["RI-EKF"][mocapBody]["position"], axis=0)/timeStep_s
            kinematics["RI-EKF"][mocapBody]["linVel"] = np.vstack((zeros_row,kinematics["RI-EKF"][mocapBody]["linVel"])) # Velocity obtained by finite differences

            kinematics["RI-EKF"][mocapBody]["locLinVel"] = kinematics["RI-EKF"][mocapBody]["R"].apply(kinematics["RI-EKF"][mocapBody]["linVel"], inverse=True)

            kinematics["RI-EKF"]["IMU"]["linVel"] = data_df[['RI-EKF_IMU_linVel_x', 'RI-EKF_IMU_linVel_y', 'RI-EKF_IMU_linVel_z']].to_numpy()
            kinematics["RI-EKF"]["IMU"]["locLinVel"] = kinematics["RI-EKF"]["IMU"]["R"].apply(kinematics["RI-EKF"]["IMU"]["linVel"], inverse=True)

            if(displayLogs):
                figLocLinVels.add_trace(go.Scatter(x=data_df["t"], y=kinematics["RI-EKF"][vel_eval_body]["locLinVel"][:,0], mode='lines', line=dict(color = colors["RI-EKF"]), name=f'locLinVel_{vel_eval_body}_{"RI-EKF"}_x'))
                figLocLinVels.add_trace(go.Scatter(x=data_df["t"], y=kinematics["RI-EKF"][vel_eval_body]["locLinVel"][:,1], mode='lines', line=dict(color = colors["RI-EKF"]), name=f'locLinVel_{vel_eval_body}_{"RI-EKF"}_y'))
                figLocLinVels.add_trace(go.Scatter(x=data_df["t"], y=kinematics["RI-EKF"][vel_eval_body]["locLinVel"][:,2], mode='lines', line=dict(color = colors["RI-EKF"]), name=f'locLinVel_{vel_eval_body}_{"RI-EKF"}_z'))

            d['RI-EKF']['llve'] = {'x': kinematics["RI-EKF"][mocapBody]["locLinVel"][:, 0], 'y': kinematics["RI-EKF"][mocapBody]["locLinVel"][:, 1], 'z': kinematics["RI-EKF"][mocapBody]["locLinVel"][:, 2]}
            d['RI-EKF']['estimate'] = {'x': kinematics["RI-EKF"]["IMU"]["locLinVel"][:, 0], 'y': kinematics["RI-EKF"]["IMU"]["locLinVel"][:, 1], 'z': kinematics["RI-EKF"]["IMU"]["locLinVel"][:, 2]}

    if(displayLogs):
        figLocLinVels.show()
    if(writeFormattedData):
       for est in d.keys():
        with open(f'{path_to_project}/output_data/{est}_loc_vel.pickle', 'wb') as f:
            pickle.dump(d[est], f)



if __name__ == '__main__':
    displayLogs = True
    writeFormattedData = False

    path_to_project = ".."


    displayLogs = sys.argv[1].lower() == 'true'          
    path_to_project = sys.argv[2]
    writeFormattedData = sys.argv[3].lower() == 'true'

    run(displayLogs, writeFormattedData, path_to_project)