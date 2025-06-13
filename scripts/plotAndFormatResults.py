import math
import pickle
import signal
import sys
import numpy as np
import pandas as pd
import yaml
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

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

    # Read the CSV file into a DataFrame

    dfObservers = pd.read_csv(f'{path_to_project}/output_data/observerResultsCSV.csv', delimiter=';')

    if(estimatorsList == None):
        with open(f'{path_to_project}/output_data/observers_infos.yaml', 'r') as file:
            try:
                observers_infos_str = file.read()
                observers_infos_yamlData = yaml.safe_load(observers_infos_str)
                estimatorsList = set(observers_infos_yamlData.get('observers'))
                mocapBody = observers_infos_yamlData.get('mocapBody')
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
        dfObservers_overlap = dfObservers[dfObservers["Mocap_datasOverlapping"] == "Datas overlap"]
        df_mocap_toIgnore = dfObservers[dfObservers["Mocap_datasOverlapping"] != "Datas overlap"]

        kinematics["Mocap"] = dict()
        kinematics["Mocap"][mocapBody] = dict()
        kinematics["Mocap"][mocapBody]["position_overlap"] = dfObservers_overlap[['Mocap_position_x', 'Mocap_position_y', 'Mocap_position_z']].to_numpy()
        kinematics["Mocap"][mocapBody]["quaternions_overlap"] = dfObservers_overlap[['Mocap_orientation_x', 'Mocap_orientation_y', 'Mocap_orientation_z', 'Mocap_orientation_w']].to_numpy()
        kinematics["Mocap"][mocapBody]["R_overlap"] = R.from_quat(kinematics["Mocap"][mocapBody]["quaternions_overlap"])
        euler_angles_Mocap_overlap = kinematics["Mocap"][mocapBody]["R_overlap"].as_euler('xyz')
        kinematics["Mocap"][mocapBody]["euler_angles_overlap"] = continuous_euler(euler_angles_Mocap_overlap)

        if(len(df_mocap_toIgnore) > 0):
            posMocap_mocap_toIgnore = df_mocap_toIgnore[['Mocap_position_x', 'Mocap_position_y', 'Mocap_position_z']].to_numpy()
            quaternionsMocap_mocap_toIgnore = df_mocap_toIgnore[['Mocap_orientation_x', 'Mocap_orientation_y', 'Mocap_orientation_z', 'Mocap_orientation_w']].to_numpy()
            rMocap_mocap_toIgnore = R.from_quat(quaternionsMocap_mocap_toIgnore)
            euler_angles_Mocap_mocap_toIgnore = rMocap_mocap_toIgnore.as_euler('xyz')
            euler_angles_Mocap_mocap_toIgnore = continuous_euler(euler_angles_Mocap_mocap_toIgnore)

    
    for estimator in estimatorsList:
        if estimator != "RI-EKF" and estimator != "Mocap":
            kinematics[estimator] = dict()
            kinematics[estimator][mocapBody] = dict()
            kinematics[estimator][mocapBody]["position"] = dfObservers[[estimator + '_position_x', estimator + '_position_y', estimator + '_position_z']].to_numpy()
            kinematics[estimator][mocapBody]["quaternions"] = dfObservers[[estimator + '_orientation_x', estimator + '_orientation_y', estimator + '_orientation_z', estimator + '_orientation_w']].to_numpy()
            rot = R.from_quat(kinematics[estimator][mocapBody]["quaternions"])
            kinematics[estimator][mocapBody]["R"] = rot
            euler_angles = rot.as_euler('xyz')
            euler_angles = continuous_euler(euler_angles)
            kinematics[estimator][mocapBody]["euler_angles"] = euler_angles

            if("Mocap" in estimatorsList):
                kinematics[estimator][mocapBody]["position_overlap"] = dfObservers_overlap[[estimator + '_position_x', estimator + '_position_y', estimator + '_position_z']].to_numpy()
                kinematics[estimator][mocapBody]["quaternions_overlap"] = dfObservers_overlap[[estimator + '_orientation_x', estimator + '_orientation_y', estimator + '_orientation_z', estimator + '_orientation_w']].to_numpy()
                kinematics[estimator][mocapBody]["R_overlap"] = R.from_quat(kinematics[estimator][mocapBody]["quaternions_overlap"])
                euler_angles_overlap = kinematics[estimator][mocapBody]["R_overlap"].as_euler('xyz')
                kinematics[estimator][mocapBody]["euler_angles_overlap"] = continuous_euler(euler_angles_overlap)

    if("RI-EKF" in estimatorsList and "Mocap" in estimatorsList):
        kinematics["RI-EKF"] = dict()
        kinematics["RI-EKF"][mocapBody] = dict()
        kinematics["RI-EKF"][mocapBody]["position"] = dfObservers[['RI-EKF_position_x', 'RI-EKF_position_y', 'RI-EKF_position_z']].to_numpy()
        kinematics["RI-EKF"][mocapBody]["quaternions"] = dfObservers[['RI-EKF_orientation_x', 'RI-EKF_orientation_y', 'RI-EKF_orientation_z', 'RI-EKF_orientation_w']].to_numpy()
        kinematics["RI-EKF"][mocapBody]["R"] = R.from_quat(kinematics["RI-EKF"][mocapBody]["quaternions"])

        euler_angles_Hartley = kinematics["RI-EKF"][mocapBody]["R"].as_euler('xyz')
        kinematics["RI-EKF"][mocapBody]["euler-angles"] = continuous_euler(euler_angles_Hartley)

        if("Mocap" in estimatorsList):
            kinematics["RI-EKF"][mocapBody]["position_overlap"] = dfObservers_overlap[['RI-EKF_position_x', 'RI-EKF_position_y', 'RI-EKF_position_z']].to_numpy()
            kinematics["RI-EKF"][mocapBody]["quaternions_overlap"] = dfObservers_overlap[['RI-EKF_orientation_x', 'RI-EKF_orientation_y', 'RI-EKF_orientation_z', 'RI-EKF_orientation_w']].to_numpy()
            kinematics["RI-EKF"][mocapBody]["R_overlap"] = R.from_quat(kinematics["RI-EKF"][mocapBody]["quaternions_overlap"])
            euler_angles_Hartley_overlap = kinematics["RI-EKF"][mocapBody]["R_overlap"].as_euler('xyz')
            kinematics["RI-EKF"][mocapBody]["euler_angles_overlap"] = continuous_euler(euler_angles_Hartley_overlap)

            kinematics["RI-EKF"]["IMU"] = dict()
            kinematics["RI-EKF"]["IMU"]["position_overlap"] = dfObservers_overlap[['RI-EKF_IMU_position_x', 'RI-EKF_IMU_position_y', 'RI-EKF_IMU_position_z']].to_numpy()
            kinematics["RI-EKF"]["IMU"]["quaternions_overlap"] = dfObservers_overlap[['RI-EKF_IMU_orientation_x', 'RI-EKF_IMU_orientation_y', 'RI-EKF_IMU_orientation_z', 'RI-EKF_IMU_orientation_w']].to_numpy()
            kinematics["RI-EKF"]["IMU"]["R_overlap"] = R.from_quat(kinematics["RI-EKF"]["IMU"]["quaternions_overlap"])
            euler_angles_Hartley_overlap = kinematics["RI-EKF"]["IMU"]["R_overlap"].as_euler('xyz')
            kinematics["RI-EKF"]["IMU"]["euler_angles_overlap"] = continuous_euler(euler_angles_Hartley_overlap)
            
            
            posImuFb_overlap = dfObservers_overlap[['HartleyIEKF_imuFbKine_position_x', 'HartleyIEKF_imuFbKine_position_y', 'HartleyIEKF_imuFbKine_position_z']].to_numpy()
            quaternions_rImuFb_overlap = dfObservers_overlap[['HartleyIEKF_imuFbKine_ori_x', 'HartleyIEKF_imuFbKine_ori_y', 'HartleyIEKF_imuFbKine_ori_z', 'HartleyIEKF_imuFbKine_ori_w']].to_numpy()
            rImuFb_overlap = R.from_quat(quaternions_rImuFb_overlap)

            linVelImuFb_overlap = dfObservers_overlap[['HartleyIEKF_imuFbKine_linVel_x', 'HartleyIEKF_imuFbKine_linVel_y', 'HartleyIEKF_imuFbKine_linVel_z']].to_numpy()
            angVelImuFb_overlap = dfObservers_overlap[['HartleyIEKF_imuFbKine_angVel_x', 'HartleyIEKF_imuFbKine_angVel_y', 'HartleyIEKF_imuFbKine_angVel_z']].to_numpy()
            posFbImu_overlap = - rImuFb_overlap.apply(posImuFb_overlap, inverse=True)
            linVelFbImu_overlap = rImuFb_overlap.apply(np.cross(angVelImuFb_overlap, posImuFb_overlap), inverse=True) - rImuFb_overlap.apply(linVelImuFb_overlap, inverse=True)



    ###################################################### Format data ######################################################

    if(writeFormattedData):
        if("Mocap" in estimatorsList):
            dfMocapPose = pd.DataFrame(columns=['#', 'timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
            dfMocapPose['timestamp'] = dfObservers_overlap['t']
            dfMocapPose['tx'] = kinematics["Mocap"][mocapBody]["position_overlap"][:,0]
            dfMocapPose['ty'] = kinematics["Mocap"][mocapBody]["position_overlap"][:,1]
            dfMocapPose['tz'] = kinematics["Mocap"][mocapBody]["position_overlap"][:,2]
            dfMocapPose['qx'] = kinematics["Mocap"][mocapBody]["quaternions_overlap"][:,0]
            dfMocapPose['qy'] = kinematics["Mocap"][mocapBody]["quaternions_overlap"][:,1]
            dfMocapPose['qz'] = kinematics["Mocap"][mocapBody]["quaternions_overlap"][:,2]
            dfMocapPose['qw'] = kinematics["Mocap"][mocapBody]["quaternions_overlap"][:,3]

            dfMocapPose = dfMocapPose[dfObservers["Mocap_datasOverlapping"] == "Datas overlap"]

            txtOutput = f'{path_to_project}/output_data/formattedMocap_Traj.txt'
            dfMocapPose.to_csv(txtOutput, header=None, index=None, sep=' ')

            line = '# timestamp tx ty tz qx qy qz qw' 
            with open(txtOutput, 'r+') as file: 
                file_data = file.read() 
                file.seek(0, 0) 
                file.write(line + '\n' + file_data) 

            d = {'x': kinematics["Mocap"][mocapBody]["position_overlap"][:, 0], 'y': kinematics["Mocap"][mocapBody]["position_overlap"][:, 1], 'z': kinematics["Mocap"][mocapBody]["position_overlap"][:, 2]}
            with open(f'{path_to_project}/output_data/mocap_x_y_z_traj.pickle', 'wb') as f:
                pickle.dump(d, f)

            for estimator in estimatorsList:
                dfPose_overlap = pd.DataFrame(columns=['#', 'timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
                dfPose_overlap['timestamp'] = dfObservers_overlap['t']
                dfPose_overlap['tx'] = kinematics[estimator][mocapBody]["position_overlap"][:,0]
                dfPose_overlap['ty'] = kinematics[estimator][mocapBody]["position_overlap"][:,1]
                dfPose_overlap['tz'] = kinematics[estimator][mocapBody]["position_overlap"][:,2]
                dfPose_overlap['qx'] = kinematics[estimator][mocapBody]["quaternions_overlap"][:,0]
                dfPose_overlap['qy'] = kinematics[estimator][mocapBody]["quaternions_overlap"][:,1]
                dfPose_overlap['qz'] = kinematics[estimator][mocapBody]["quaternions_overlap"][:,2]
                dfPose_overlap['qw'] = kinematics[estimator][mocapBody]["quaternions_overlap"][:,3]

                txtOutput = f'{path_to_project}/output_data/formatted_{estimator}_Traj.txt'
                dfPose_overlap.to_csv(txtOutput, header=None, index=None, sep=' ')

                line = '# timestamp tx ty tz qx qy qz qw' 
                with open(txtOutput, 'r+') as file: 
                    file_data = file.read() 
                    file.seek(0, 0) 
                    file.write(line + '\n' + file_data) 

                d = {'x': dfPose_overlap['tx'], 'y': dfPose_overlap['ty'], 'z': dfPose_overlap['tz']}
                with open(f'{path_to_project}/output_data/{estimator}_x_y_z_traj.pickle', 'wb') as f:
                    pickle.dump(d, f)


    ###################################################### Display logs ######################################################


    if(displayLogs):
        # Create the figure
        fig_pose = go.Figure()
        for estimator in estimatorsList:
            # Add traces for each plot
            fig_pose.add_trace(go.Scatter(x=dfObservers['t'], y=kinematics[estimator][mocapBody]["position_overlap"][:, 0], mode='lines', line=dict(color = colors[estimator]), name=f'{estimator}_Position_x'))
            fig_pose.add_trace(go.Scatter(x=dfObservers['t'], y=kinematics[estimator][mocapBody]["position_overlap"][:, 1], mode='lines', line=dict(color = colors[estimator]), name=f'{estimator}_Position_y'))
            fig_pose.add_trace(go.Scatter(x=dfObservers['t'], y=kinematics[estimator][mocapBody]["position_overlap"][:, 2], mode='lines', line=dict(color = colors[estimator]), name=f'{estimator}_Position_z'))
            fig_pose.add_trace(go.Scatter(x=dfObservers['t'], y=kinematics[estimator][mocapBody]["euler_angles_overlap"][:, 0], mode='lines', line=dict(color = colors[estimator]), name=f'{estimator}_Roll'))
            fig_pose.add_trace(go.Scatter(x=dfObservers['t'], y=kinematics[estimator][mocapBody]["euler_angles_overlap"][:, 1], mode='lines', line=dict(color = colors[estimator]), name=f'{estimator}_Pitch'))
            fig_pose.add_trace(go.Scatter(x=dfObservers['t'], y=kinematics[estimator][mocapBody]["euler_angles_overlap"][:, 2], mode='lines', line=dict(color = colors[estimator]), name=f'{estimator}_Yaw'))


        # Update layout
        fig_pose.update_layout(
            title= f'{scriptName}: Pose over time',
            xaxis_title='Time',
            yaxis_title='Value',
            hovermode='x'
        )
        # Show the interactive plot
        # fig_pose.show()

        fig_traj_2d = go.Figure()

        for estimator in estimatorsList:
            fig_traj_2d.add_trace(go.Scatter(x=kinematics[estimator][mocapBody]["position_overlap"][:, 0], y=kinematics[estimator][mocapBody]["position_overlap"][:, 1], mode='lines', line=dict(color = colors[estimator]), name=f'{estimator}_2dMotion_xy'))

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
        # fig_traj_2d.show()

        fig_traj_3d = go.Figure()

        for estimator in estimatorsList:
            x_min = min((kinematics[estimator][mocapBody]["position_overlap"][:, 0]).min(), (kinematics[estimator][mocapBody]["position_overlap"][:, 0]).min(), (kinematics[estimator][mocapBody]["position_overlap"][:, 0]).min())
            y_min = min((kinematics[estimator][mocapBody]["position_overlap"][:,1]).min(), (kinematics[estimator][mocapBody]["position_overlap"][:,1]).min(), (kinematics[estimator][mocapBody]["position_overlap"][:,1]).min())
            z_min = min((kinematics[estimator][mocapBody]["position_overlap"][:,2]).min(), (kinematics[estimator][mocapBody]["position_overlap"][:,2]).min(), (kinematics[estimator][mocapBody]["position_overlap"][:,2]).min())

            x_max = max((kinematics[estimator][mocapBody]["position_overlap"][:,0]).max(), (kinematics[estimator][mocapBody]["position_overlap"][:,0]).max(), (kinematics[estimator][mocapBody]["position_overlap"][:,0]).max())
            y_max = max((kinematics[estimator][mocapBody]["position_overlap"][:,1]).max(), (kinematics[estimator][mocapBody]["position_overlap"][:,1]).max(), (kinematics[estimator][mocapBody]["position_overlap"][:,1]).max())
            z_max = max((kinematics[estimator][mocapBody]["position_overlap"][:,2]).max(), (kinematics[estimator][mocapBody]["position_overlap"][:,2]).max(), (kinematics[estimator][mocapBody]["position_overlap"][:,2]).max())

            # Add traces
            fig_traj_3d.add_trace(go.Scatter3d(
                x=kinematics[estimator][mocapBody]["position_overlap"][:,0], 
                y=kinematics[estimator][mocapBody]["position_overlap"][:,1], 
                z=kinematics[estimator][mocapBody]["position_overlap"][:,2],
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
        # fig_traj_3d.show()


        fig_gyroBias = go.Figure()

        # Get the last time in dfObservers['t']
        last_time = dfObservers['t'].max()

        # Filter the data for the first 3 seconds
        df_first_3s = dfObservers[(dfObservers['t'] >= 0) & (dfObservers['t'] <= 3)]

        # Compute the average bias over the first 3 seconds
        average_bias_x_3 = df_first_3s['Accelerometer_angularVelocity_x'].mean()
        average_bias_x_tuple_3 = tuple(average_bias_x_3 for _ in range(len(dfObservers)))
        average_bias_y_3 = df_first_3s['Accelerometer_angularVelocity_y'].mean()
        average_bias_y_tuple_3 = tuple(average_bias_y_3 for _ in range(len(dfObservers)))
        average_bias_z_3 = df_first_3s['Accelerometer_angularVelocity_z'].mean()
        average_bias_z_tuple_3 = tuple(average_bias_z_3 for _ in range(len(dfObservers)))

        # Filter the data for the last 3 seconds
        df_last_3s = dfObservers[(dfObservers['t'] >= last_time - 3) & (dfObservers['t'] <= last_time)]

        # Compute the average bias over the last 3 seconds
        average_bias_x_last = df_last_3s['Accelerometer_angularVelocity_x'].mean()
        average_bias_x_tuple_last = tuple(average_bias_x_last for _ in range(len(dfObservers)))
        average_bias_y_last = df_last_3s['Accelerometer_angularVelocity_y'].mean()
        average_bias_y_tuple_last = tuple(average_bias_y_last for _ in range(len(dfObservers)))
        average_bias_z_last = df_last_3s['Accelerometer_angularVelocity_z'].mean()
        average_bias_z_tuple_last = tuple(average_bias_z_last for _ in range(len(dfObservers)))

        # Plotting the original data and the computed biases
        fig_gyroBias.add_trace(go.Scatter(x=dfObservers['t'], y=dfObservers['Accelerometer_angularVelocity_x'], mode='lines', name='measured_angVel_x'))
        fig_gyroBias.add_trace(go.Scatter(x=dfObservers['t'], y=dfObservers['Accelerometer_angularVelocity_y'], mode='lines', name='measured_angVel_y'))
        fig_gyroBias.add_trace(go.Scatter(x=dfObservers['t'], y=dfObservers['Accelerometer_angularVelocity_z'], mode='lines', name='measured_angVel_z'))
        fig_gyroBias.add_trace(go.Scatter(x=dfObservers['t'], y=average_bias_x_tuple_3, mode='lines', name='measured_GyroBias_beginning_x'))
        fig_gyroBias.add_trace(go.Scatter(x=dfObservers['t'], y=average_bias_y_tuple_3, mode='lines', name='measured_GyroBias_beginning_y'))
        fig_gyroBias.add_trace(go.Scatter(x=dfObservers['t'], y=average_bias_z_tuple_3, mode='lines', name='measured_GyroBias_beginning_z'))
        fig_gyroBias.add_trace(go.Scatter(x=dfObservers['t'], y=average_bias_x_tuple_last, mode='lines', name='measured_GyroBias_end_x'))
        fig_gyroBias.add_trace(go.Scatter(x=dfObservers['t'], y=average_bias_y_tuple_last, mode='lines', name='measured_GyroBias_end_y'))
        fig_gyroBias.add_trace(go.Scatter(x=dfObservers['t'], y=average_bias_z_tuple_last, mode='lines', name='measured_GyroBias_end_z'))

        obs = []
        with open('../observersInfos.yaml', 'r') as file:
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
                fig_gyroBias.add_trace(go.Scatter(x=dfObservers['t'], y=dfObservers[f'{estimator}_IMU_gyroBias_x'], mode='lines', name=f'{estimator}_gyroBias_x'))
                fig_gyroBias.add_trace(go.Scatter(x=dfObservers['t'], y=dfObservers[f'{estimator}_IMU_gyroBias_y'], mode='lines', name=f'{estimator}_gyroBias_y'))
                fig_gyroBias.add_trace(go.Scatter(x=dfObservers['t'], y=dfObservers[f'{estimator}_IMU_gyroBias_z'], mode='lines', name=f'{estimator}_gyroBias_z'))
        # Update layout
        fig_gyroBias.update_layout(
            title='Gyrometer biases',
            xaxis_title='x',
            yaxis_title='y',
            hovermode='x'
        )

        # Show the interactive plot
        # fig_gyroBias.show()



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
        kinematics['Mocap'][mocapBody]['linVel_overlap'] = np.diff(kinematics['Mocap'][mocapBody]['position_overlap'], axis=0)/timeStep_s
        kinematics['Mocap'][mocapBody]['linVel_overlap'] = np.vstack((zeros_row,kinematics['Mocap'][mocapBody]['linVel_overlap']))

        kinematics['Mocap']['IMU']['position_overlap'] =  kinematics['Mocap'][mocapBody]['position_overlap'] + kinematics['Mocap'][mocapBody]['R_overlap'].apply(posFbImu_overlap)
        kinematics['Mocap']['IMU']['R_overlap'] = kinematics['Mocap'][mocapBody]['R_overlap'] * rImuFb_overlap.inv()

        kinematics['Mocap']['IMU']['linVel_overlap'] = np.diff(kinematics['Mocap']['IMU']['position_overlap'], axis=0)/timeStep_s
        kinematics['Mocap']['IMU']['linVel_overlap'] = np.vstack((zeros_row,kinematics['Mocap']['IMU']['linVel_overlap']))
        
        kinematics['Mocap'][mocapBody]['locLinVel_overlap'] = kinematics['Mocap'][mocapBody]['R_overlap'].apply(kinematics['Mocap'][mocapBody]['linVel_overlap'], inverse=True)
        kinematics['Mocap']['IMU']['locLinVel_overlap'] = kinematics['Mocap']['IMU']['R_overlap'].apply(kinematics['Mocap'][mocapBody]['linVel_overlap'], inverse=True)

        b, a = butter(2, 0.15, analog=False)

        kinematics['Mocap'][mocapBody]['locLinVel_overlap'] = filtfilt(b, a, kinematics['Mocap'][mocapBody]['locLinVel_overlap'], axis=0)
        kinematics['Mocap']['IMU']['locLinVel_overlap'] = filtfilt(b, a, kinematics['Mocap']['IMU']['locLinVel_overlap'], axis=0)
        d['Mocap'] = {'llve': {}, 'estimate': {}}
        d['Mocap']['llve'] = {'x': kinematics['Mocap'][mocapBody]['locLinVel_overlap'][:, 0], 'y': kinematics['Mocap'][mocapBody]['locLinVel_overlap'][:, 1], 'z': kinematics['Mocap'][mocapBody]['locLinVel_overlap'][:, 2]}
        d['Mocap']['estimate'] = {'x': kinematics['Mocap'][vel_eval_body]['locLinVel_overlap'][:, 0], 'y': kinematics['Mocap'][vel_eval_body]['locLinVel_overlap'][:, 1], 'z': kinematics['Mocap'][vel_eval_body]['locLinVel_overlap'][:, 2]}
        with open(f'{path_to_project}/output_data/mocap_loc_vel.pickle', 'wb') as f:
            pickle.dump(d['Mocap'], f)


        if(displayLogs):
            figLocLinVels = go.Figure()
            figLocLinVels.add_trace(go.Scatter(x=dfObservers_overlap["t"], y=kinematics['Mocap'][mocapBody]['locLinVel_overlap'][:,0], mode='lines', line=dict(color = colors["Mocap"]), name='locLinvel_Mocap_x'))
            figLocLinVels.add_trace(go.Scatter(x=dfObservers_overlap["t"], y=kinematics['Mocap'][mocapBody]['locLinVel_overlap'][:,1], mode='lines', line=dict(color = colors["Mocap"]), name='locLinvel_Mocap_y'))
            figLocLinVels.add_trace(go.Scatter(x=dfObservers_overlap["t"], y=kinematics['Mocap'][mocapBody]['locLinVel_overlap'][:,2], mode='lines', line=dict(color = colors["Mocap"]), name='locLinvel_Mocap_z'))

            figLocLinVels.update_layout(title=f"{scriptName}: Linear velocities")  

        for estimator in estimatorsList:
            if estimator != "RI-EKF":
                d[estimator] = dict()

                kinematics[estimator][mocapBody]['llve'] = np.diff(kinematics['Mocap'][mocapBody]['position_overlap'], axis=0)/timeStep_s
                kinematics[estimator][mocapBody]['llve'] = np.vstack((zeros_row,kinematics[estimator][mocapBody]['llve']))
                kinematics[estimator][mocapBody]['llve'] = kinematics[estimator][mocapBody]["R_overlap"].apply(kinematics[estimator][mocapBody]['llve'], inverse=True)

                d[estimator]['llve'] = {'x': kinematics[estimator][mocapBody]["llve"][:, 0], 'y': kinematics[estimator][mocapBody]["llve"][:, 1], 'z': kinematics[estimator][mocapBody]["llve"][:, 2]}

                with open('../observersInfos.yaml', 'r') as file:
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
                        if f'{prefix}_locLinVel_x' in dfObservers.columns:
                            kinematics[estimator][vel_eval_body]["locLinVel_overlap"] = dfObservers[[prefix + '_locLinVel_x', prefix + '_locLinVel_y', prefix + '_locLinVel_z']].to_numpy()
                        elif f'{prefix}_LinVel_x' in dfObservers.columns and f'{prefix}_orientation_x' in dfObservers.columns:
                            kinematics[estimator][vel_eval_body]["linVel_overlap"] = dfObservers[[prefix + '_linVel_x', prefix + '_linVel_y', prefix + '_linVel_z']].to_numpy()
                            kinematics[estimator][vel_eval_body]["locLinVel_overlap"] = kinematics[prefix][vel_eval_body]["R_overlap"].apply(kinematics[estimator][vel_eval_body]["linVel_overlap"], inverse=True)
                        else:
                            continue

                        if(displayLogs):
                            figLocLinVels.add_trace(go.Scatter(x=dfObservers_overlap["t"], y=kinematics[estimator][vel_eval_body]["locLinVel_overlap"][:,0], mode='lines', line=dict(color = colors[estimator]), name=f'locLinVel_{vel_eval_body}_{estimator}_x'))
                            figLocLinVels.add_trace(go.Scatter(x=dfObservers_overlap["t"], y=kinematics[estimator][vel_eval_body]["locLinVel_overlap"][:,1], mode='lines', line=dict(color = colors[estimator]), name=f'locLinVel_{vel_eval_body}_{estimator}_y'))
                            figLocLinVels.add_trace(go.Scatter(x=dfObservers_overlap["t"], y=kinematics[estimator][vel_eval_body]["locLinVel_overlap"][:,2], mode='lines', line=dict(color = colors[estimator]), name=f'locLinVel_{vel_eval_body}_{estimator}_z'))

                        d[estimator]['estimate'] = {'x': kinematics[estimator][vel_eval_body]["locLinVel_overlap"][:, 0], 'y': kinematics[estimator][vel_eval_body]["locLinVel_overlap"][:, 1], 'z': kinematics[estimator][vel_eval_body]["locLinVel_overlap"][:, 2]}
            
        if("RI-EKF" in estimatorsList):
            d['RI-EKF'] = dict()
            kinematics["RI-EKF"][mocapBody]["linVel_overlap"] = np.diff(kinematics["RI-EKF"][mocapBody]["position_overlap"], axis=0)/timeStep_s
            kinematics["RI-EKF"][mocapBody]["linVel_overlap"] = np.vstack((zeros_row,kinematics["RI-EKF"][mocapBody]["linVel_overlap"])) # Velocity obtained by finite differences

            kinematics["RI-EKF"][mocapBody]["locLinVel_overlap"] = kinematics["RI-EKF"][mocapBody]["R_overlap"].apply(kinematics["RI-EKF"][mocapBody]["linVel_overlap"], inverse=True)

            kinematics["RI-EKF"]["IMU"]["linVel_overlap"] = dfObservers_overlap[['RI-EKF_IMU_linVel_x', 'RI-EKF_IMU_linVel_y', 'RI-EKF_IMU_linVel_z']].to_numpy()
            kinematics["RI-EKF"]["IMU"]["locLinVel_overlap"] = kinematics["RI-EKF"]["IMU"]["R_overlap"].apply(kinematics["RI-EKF"]["IMU"]["linVel_overlap"], inverse=True)

            if(displayLogs):
                figLocLinVels.add_trace(go.Scatter(x=dfObservers_overlap["t"], y=kinematics["RI-EKF"][vel_eval_body]["locLinVel_overlap"][:,0], mode='lines', line=dict(color = colors["RI-EKF"]), name=f'locLinVel_{vel_eval_body}_{"RI-EKF"}_x'))
                figLocLinVels.add_trace(go.Scatter(x=dfObservers_overlap["t"], y=kinematics["RI-EKF"][vel_eval_body]["locLinVel_overlap"][:,1], mode='lines', line=dict(color = colors["RI-EKF"]), name=f'locLinVel_{vel_eval_body}_{"RI-EKF"}_y'))
                figLocLinVels.add_trace(go.Scatter(x=dfObservers_overlap["t"], y=kinematics["RI-EKF"][vel_eval_body]["locLinVel_overlap"][:,2], mode='lines', line=dict(color = colors["RI-EKF"]), name=f'locLinVel_{vel_eval_body}_{"RI-EKF"}_z'))

            d['RI-EKF']['llve'] = {'x': kinematics["RI-EKF"][mocapBody]["locLinVel_overlap"][:, 0], 'y': kinematics["RI-EKF"][mocapBody]["locLinVel_overlap"][:, 1], 'z': kinematics["RI-EKF"][mocapBody]["locLinVel_overlap"][:, 2]}
            d['RI-EKF']['estimate'] = {'x': kinematics["RI-EKF"]["IMU"]["locLinVel_overlap"][:, 0], 'y': kinematics["RI-EKF"]["IMU"]["locLinVel_overlap"][:, 1], 'z': kinematics["RI-EKF"]["IMU"]["locLinVel_overlap"][:, 2]}

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


    if(len(sys.argv) > 1):
        timeStepInput = sys.argv[1]
        if(len(sys.argv) > 2):
            displayLogs = sys.argv[2].lower() == 'true'
        if(len(sys.argv) > 3):
            path_to_project = sys.argv[3]
        if(len(sys.argv) > 4):
            writeFormattedData = sys.argv[4].lower() == 'true'
    else:
        timeStepInput = input("Please enter the timestep of the controller in milliseconds: ")

    try:
        # Check if the timestep was given in milliseconds
        if(timeStepInput.isdigit()):
            timeStep_ms = int(timeStepInput)
            timeStep_s = float(timeStep_ms)/1000.0
        else:
            timeStep_s = float(timeStepInput)
            timeStep_ms = int(timeStep_s*1000.0)
        resample_str = f'{timeStep_ms}ms'
    except ValueError:
        print(f"The input timestep is not valid: {timeStepInput}")
        sys.exit(1)

    run(displayLogs, writeFormattedData, path_to_project)