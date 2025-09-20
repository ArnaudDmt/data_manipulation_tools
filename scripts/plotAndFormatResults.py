import math
import pickle
import signal
import sys
import numpy as np
import pandas as pd
import yaml
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from collections import deque

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
        kinematics["Mocap"][mocapBody]["quaternions"] = data_df[['Mocap_orientation_x', 'Mocap_orientation_y', 'Mocap_orientation_z', 'Mocap_orientation_w']].to_numpy()
        kinematics["Mocap"][mocapBody]["R"] = R.from_quat(kinematics["Mocap"][mocapBody]["quaternions"])
        euler_angles_Mocap = kinematics["Mocap"][mocapBody]["R"].as_euler('xyz')
        kinematics["Mocap"][mocapBody]["euler_angles"] = continuous_euler(euler_angles_Mocap)
    
    for estimator in estimatorsList:
        if estimator != "Hartley" and estimator != "Mocap":
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

    if("Hartley" in estimatorsList and "Mocap" in estimatorsList):
        kinematics["Hartley"] = dict()
        kinematics["Hartley"][mocapBody] = dict()
        kinematics["Hartley"][mocapBody]["position"] = data_df[['Hartley_position_x', 'Hartley_position_y', 'Hartley_position_z']].to_numpy()
        kinematics["Hartley"][mocapBody]["quaternions"] = data_df[['Hartley_orientation_x', 'Hartley_orientation_y', 'Hartley_orientation_z', 'Hartley_orientation_w']].to_numpy()
        kinematics["Hartley"][mocapBody]["R"] = R.from_quat(kinematics["Hartley"][mocapBody]["quaternions"])

        euler_angles_Hartley = kinematics["Hartley"][mocapBody]["R"].as_euler('xyz')
        kinematics["Hartley"][mocapBody]["euler-angles"] = continuous_euler(euler_angles_Hartley)

        if("Mocap" in estimatorsList):
            kinematics["Hartley"][mocapBody]["position"] = data_df[['Hartley_position_x', 'Hartley_position_y', 'Hartley_position_z']].to_numpy()
            kinematics["Hartley"][mocapBody]["quaternions"] = data_df[['Hartley_orientation_x', 'Hartley_orientation_y', 'Hartley_orientation_z', 'Hartley_orientation_w']].to_numpy()
            kinematics["Hartley"][mocapBody]["R"] = R.from_quat(kinematics["Hartley"][mocapBody]["quaternions"])
            euler_angles_Hartley = kinematics["Hartley"][mocapBody]["R"].as_euler('xyz')
            kinematics["Hartley"][mocapBody]["euler_angles"] = continuous_euler(euler_angles_Hartley)

            kinematics["Hartley"]["IMU"] = dict()
            kinematics["Hartley"]["IMU"]["position"] = data_df[['Hartley_IMU_position_x', 'Hartley_IMU_position_y', 'Hartley_IMU_position_z']].to_numpy()
            kinematics["Hartley"]["IMU"]["quaternions"] = data_df[['Hartley_IMU_orientation_x', 'Hartley_IMU_orientation_y', 'Hartley_IMU_orientation_z', 'Hartley_IMU_orientation_w']].to_numpy()
            kinematics["Hartley"]["IMU"]["R"] = R.from_quat(kinematics["Hartley"]["IMU"]["quaternions"])
            euler_angles_Hartley = kinematics["Hartley"]["IMU"]["R"].as_euler('xyz')
            kinematics["Hartley"]["IMU"]["euler_angles"] = continuous_euler(euler_angles_Hartley)
            
            
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
                if estimator == "Mocap":
                    continue
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

        # fig_traj_2d = go.Figure()

        # for estimator in estimatorsList:
        #     fig_traj_2d.add_trace(go.Scatter(x=kinematics[estimator][mocapBody]["position"][:, 0], y=kinematics[estimator][mocapBody]["position"][:, 1], mode='lines', line=dict(color = colors[estimator]), name=f'{estimator}_2dMotion_xy'))

        # # Update layout
        # fig_traj_2d.update_layout(
        #     xaxis_title='x',
        #     yaxis_title='y',
        #     hovermode='x',
        #     title=f"{scriptName}: 2D trajectories",
        #     updatemenus=[dict(type="buttons",
        #                                buttons=[dict(label="Play",
        #                                              method="animate",
        #                                              args=[None])])],
        # )
        # fig_traj_2d.update_yaxes(
        #     scaleanchor="x",
        #     scaleratio=1,
            
        # )
        
        # # Show the interactive plot
        # fig_traj_2d.show()

        # fig_traj_2d_animated = go.Figure()

        # # Initial frame (empty traces)
        # for estimator in estimatorsList:
        #     fig_traj_2d_animated.add_trace(go.Scatter(
        #         x=[], y=[],
        #         mode='lines',
        #         line=dict(color=colors[estimator]),
        #         name=f'{estimator}_2dMotion_xy'
        #     ))

        # # Number of steps (assumes all estimators have same length)
        # n_steps = min([kinematics[estimator][mocapBody]["position"].shape[0] for estimator in estimatorsList])

        # # Create frames
        # frames = []
        # for i in range(n_steps):
        #     frame_data = []
        #     for estimator in estimatorsList:
        #         pos = kinematics[estimator][mocapBody]["position"]
        #         frame_data.append(go.Scatter(
        #             x=pos[:i+1, 0],
        #             y=pos[:i+1, 1],
        #             mode='lines',
        #             line=dict(color=colors[estimator]),
        #             name=f'{estimator}_2dMotion_xy'
        #         ))
        #     frames.append(go.Frame(data=frame_data, name=str(i)))

        # fig_traj_2d_animated.frames = frames

        # # Layout with play button and slider
        # fig_traj_2d_animated.update_layout(
        #     xaxis_title='x',
        #     yaxis_title='y',
        #     hovermode='x',
        #     title=f"{scriptName}: 2D trajectories",
        #     updatemenus=[dict(
        #         type="buttons",
        #         showactive=False,
        #         buttons=[dict(label="Play", method="animate", args=[None])]
        #     )],
        #     sliders=[dict(
        #         steps=[dict(method="animate", args=[[str(k)]], label=str(k)) for k in range(n_steps)],
        #         transition=dict(duration=0),
        #         x=0.1,
        #         len=0.9
        #     )]
        # )

        # fig_traj_2d_animated.update_yaxes(scaleanchor="x", scaleratio=1)

        # fig_traj_2d_animated.show()

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
        # fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=data_df['Accelerometer_angularVelocity_x'], mode='lines', name='measured_angVel_x'))
        # fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=data_df['Accelerometer_angularVelocity_y'], mode='lines', name='measured_angVel_y'))
        # fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=data_df['Accelerometer_angularVelocity_z'], mode='lines', name='measured_angVel_z'))
        # fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=average_bias_x_tuple_3, mode='lines', name='measured_GyroBias_beginning_x'))
        # fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=average_bias_y_tuple_3, mode='lines', name='measured_GyroBias_beginning_y'))
        # fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=average_bias_z_tuple_3, mode='lines', name='measured_GyroBias_beginning_z'))
        # fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=average_bias_x_tuple_last, mode='lines', name='measured_GyroBias_end_x'))
        # fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=average_bias_y_tuple_last, mode='lines', name='measured_GyroBias_end_y'))
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
                # fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=data_df[f'{estimator}_IMU_gyroBias_x'], mode='lines', name=f'{estimator}_gyroBias_x'))
                # fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=data_df[f'{estimator}_IMU_gyroBias_y'], mode='lines', name=f'{estimator}_gyroBias_y'))
                fig_gyroBias.add_trace(go.Scatter(x=data_df['t'], y=data_df[f'{estimator}_IMU_gyroBias_z'], mode='lines', name=f'{estimator}_gyroBias_z'))
        # Update layout
        fig_gyroBias.update_layout(
            title='Gyrometer biases',
            xaxis_title='Time [seconds]',
            yaxis_title='Gyrometer bias (rad/s)',
            hovermode='x'
        )

        # Show the interactive plot
        fig_gyroBias.show()
        fig_gyroBias.write_image(f'/tmp/biases.svg')


    ###############################  Criteria based on the step sequence  ###############################

    # contact_columns = [col for col in data_df.columns if col.endswith('_isSet')]

    contact_columns = {
        col.split('_')[-2]: col
        for col in data_df.columns
        if col.endswith('_isSet')
        }
    if(writeFormattedData and len(contact_columns.keys()) > 0):
        

        contact_transitions_init = {}

        for contact, col in contact_columns.items():
            series = data_df[col].fillna("").astype(str)
            prev = series.shift(1).fillna("").astype(str)
            transitions = (prev == "Set") & (series != "Set")
            contact_transitions_init[contact] = deque(series.index[transitions].tolist())

        contacts = list(contact_transitions_init.keys())
        intervals = []
        
        contact_transitions = contact_transitions_init.copy()

        # First phase: find when all contacts have been removed once
        removalIndex_contact_map = dict()
        for c in contacts:
            if not contact_transitions[c]:
                break  # if any contact has no transition, we can't proceed
            removalIndex_contact_map[contact_transitions[c][0]] = c

        if len(removalIndex_contact_map) < len(contacts):
            raise ValueError("Not all contacts have at least one transition.")

        end_time = max(removalIndex_contact_map.keys())
        intervals.append({
            'start_time': 0,
            'end_time': end_time,
            'reference': None
        })

        reference = removalIndex_contact_map[end_time]
        
        for c in contacts:
            while  contact_transitions[c][0] < end_time:
                contact_transitions[c].popleft()

        currently_set_contacts = [
                c for c in contacts
                if data_df[contact_columns[c]].iloc[0] == "Set"
            ]

        while True:
            start_time = end_time
            # removalIndex_contact_map = {end_time: removalIndex_contact_map[end_time]}

            removalIndex_contact_map = dict()
            success = False
            if reference is not None and contact_transitions[reference]:
                currently_set_contacts = [
                c for c in contacts
                if data_df[contact_columns[c]].iloc[end_time] == "Set"
            ]
                
                index_found = None
                success = False
                other_contacts = [c for c in currently_set_contacts if c != reference]
                if other_contacts:
                    for i, t in enumerate(contact_transitions[reference]):
                        if t <= start_time:
                            continue  # Skip transitions before or at start_time

                        all_removed = True
                        for c in other_contacts:
                            transitions = contact_transitions.get(c, [])
                            # Check if there is any transition for contact c strictly after start_time and at or before t
                            if not any(start_time < tc <= t for tc in transitions):
                                all_removed = False
                                break

                        if all_removed:
                            success = True
                            index_found = i
                            break  # Found a valid time for the reference contact

                if success:
                    end_time = contact_transitions[reference][index_found]


            skipIter = False
            if reference == None or success == False:
                for c in currently_set_contacts:
                    if len(contact_transitions[c]) > 2:
                        removalIndex_contact_map[contact_transitions[c][0]] = c
                    else:
                        break
                if removalIndex_contact_map:
                    start_time = min(removalIndex_contact_map.keys())
                    end_time = start_time
                    reference = removalIndex_contact_map[start_time]
                    success = True
                    skipIter = True
            
            if skipIter:
                continue

            if not success and not skipIter:
                break
            

            intervals.append({
                'start_time': start_time,
                'end_time': end_time,
                'reference': reference
            })

            nbRemaining = 0
            
            for c in contacts:
                while contact_transitions[c] and contact_transitions[c][0] < end_time:
                    contact_transitions[c].popleft()
                if not contact_transitions[c]:
                    reference = None
                nbRemaining += len(contact_transitions[c])
            if nbRemaining == 0:
                break

        fig = go.Figure()

        def get_invariant_orthogonal_vector(Rhat: np.ndarray, Rtez: np.ndarray):
                epsilon = 2.2204460492503131e-16
                Rhat_Rtez = np.dot(Rhat, Rtez)
                if np.all(np.abs(Rhat_Rtez[:2]) < epsilon):
                    return np.array([1, 0, 0])
                else:
                    return np.array([Rhat_Rtez[1], -Rhat_Rtez[0], 0])
                
        def merge_tilt_with_yaw_axis_agnostic(Rtez: np.ndarray, R2: np.ndarray):
            ez = np.array([0, 0, 1])
            v1 = Rtez
        
            m = get_invariant_orthogonal_vector(R2, Rtez)
            m = m / np.linalg.norm(m)

            ml = np.dot(R2.T, m)

            R_temp1 = np.column_stack((np.cross(m, ez), m, ez))

            R_temp2 = np.vstack((np.cross(ml, v1).T, ml.T, v1.T))

            return np.dot(R_temp1, R_temp2)
        
        from plotly.subplots import make_subplots

        def _merge_consecutive_intervals(_intervals, k):
            out = []
            if k <= 1:
                return _intervals[:]
            for i in range(len(_intervals) - k + 1):
                out.append({
                    'start_time': _intervals[i]['start_time'],
                    'end_time': _intervals[i + k - 1]['end_time'],
                    'reference': _intervals[i + k - 1]['reference']
                })
            return out

        final_position_errors = {estimator: [] for estimator in estimatorsList}

        # Ensure d exists and seed for 1, 2, 3
        if 'd' not in locals():
            d = {1: {}}
        elif 1 not in d:
            d[1] = {}

        for _mult in [1, 2, 3]:
            if _mult not in d:
                d[_mult] = {}
            for estimator in estimatorsList:
                if estimator != "Mocap" and estimator not in d[_mult]:
                    d[_mult][estimator] = {'pos': [], 'tilt': [], 'yaw': []}

        cycle_mults = [1, 2, 3]
        intervals_by_mult = {
            1: intervals,
            2: _merge_consecutive_intervals(intervals, 2),
            3: _merge_consecutive_intervals(intervals, 3),
        }

        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            subplot_titles=["Contact states", "1 cycle", "2 cycles", "3 cycles"]
        )

        import plotly.colors as pc
        interval_colors = pc.qualitative.Pastel

        # Contact states subplot (row 1)
        for i_cs, (name, col) in enumerate(contact_columns.items()):
            raw_state = (data_df[col] != "Set").astype(float)
            y_max_cs = 1.0# - i_cs * 0.1
            scaled_state = raw_state * y_max_cs
            fig.add_trace(
                go.Scatter(
                    x=data_df.index,
                    y=scaled_state,
                    mode="lines",
                    name=name,
                    legendgroup="contacts",
                    showlegend=True,
                    line=dict(dash="longdashdot") 
                ),
                row=1, col=1
            )

        # Same colored areas across ALL subplots, based on base 'intervals'
        for i_rect, interval in enumerate(intervals):
            for r in [1, 2, 3, 4]:
                fig.add_vrect(
                    x0=interval["start_time"],
                    x1=interval["end_time"],
                    fillcolor=interval_colors[i_rect % len(interval_colors)],
                    opacity=0.3,
                    line_width=0,
                    row=r,
                    col=1
                )
        

        show_legend_traj = True
        row_for_mult = {1: 2, 2: 3, 3: 4} 

        for mult in cycle_mults:
            poses = {}
            for estimator in estimatorsList:
                poses[estimator] = {"tx": [], "ty": [], "tz": [], "rz": [], "rx": [], "ry": []}
            idx_range = []
            min_y = float("inf")
            max_y = -float("inf")

            for i_int, interval in enumerate(intervals_by_mult[mult]):
                start_time = interval["start_time"]
                end_time = interval["end_time"]

                R_mocap = kinematics["Mocap"][mocapBody]["R"][start_time:end_time]
                pos_mocap = kinematics["Mocap"][mocapBody]["position"][start_time:end_time]

                aligned_init_ori_mat_mocap = R.from_matrix(merge_tilt_with_yaw_axis_agnostic(
                    R_mocap[0].apply([0, 0, 1], inverse=True), R.identity().as_matrix()
                ))

                R_aligned_mocap = aligned_init_ori_mat_mocap * R_mocap[0].inv() * R_mocap
                p_aligned_mocap = np.array([0, 0, 0]) + \
                    (aligned_init_ori_mat_mocap * R_mocap[0].inv()).apply(pos_mocap - pos_mocap[0])

                idx_range.extend(range(start_time, end_time))
                poses["Mocap"]["tx"].extend(p_aligned_mocap[:, 0])
                poses["Mocap"]["ty"].extend(p_aligned_mocap[:, 1])
                poses["Mocap"]["tz"].extend(p_aligned_mocap[:, 2])
                euler_m = R_aligned_mocap.as_euler("zxy")
                poses["Mocap"]["rz"].extend(euler_m[:, 0])
                poses["Mocap"]["rx"].extend(euler_m[:, 1])
                poses["Mocap"]["ry"].extend(euler_m[:, 2])

                min_y = min(min_y, np.min(p_aligned_mocap))
                max_y = max(max_y, np.max(p_aligned_mocap))

                for estimator in estimatorsList:
                    if estimator == "Mocap":
                        continue

                    R_est = kinematics[estimator][mocapBody]["R"][start_time:end_time]
                    pos_est = kinematics[estimator][mocapBody]["position"][start_time:end_time]

                    aligned_init_ori_mat_est = R.from_matrix(merge_tilt_with_yaw_axis_agnostic(
                        R_est[0].apply([0, 0, 1], inverse=True), R.identity().as_matrix()
                    ))

                    R_aligned_est = aligned_init_ori_mat_est * R_est[0].inv() * R_est
                    p_aligned_est = np.array([0, 0, 0]) + \
                        (aligned_init_ori_mat_est * R_est[0].inv()).apply(pos_est - pos_est[0])

                    poses[estimator]["tx"].extend(p_aligned_est[:, 0])
                    poses[estimator]["ty"].extend(p_aligned_est[:, 1])
                    poses[estimator]["tz"].extend(p_aligned_est[:, 2])
                    euler = R_aligned_est.as_euler("zxy")
                    poses[estimator]["rz"].extend(euler[:, 0])
                    poses[estimator]["rx"].extend(euler[:, 1])
                    poses[estimator]["ry"].extend(euler[:, 2])

                    error_pos = p_aligned_est[-1] - p_aligned_mocap[-1]
                    final_position_errors[estimator].append(error_pos)

                    scalar_product = np.dot(
                        R_aligned_mocap[-1].apply([0, 0, 1], inverse=True),
                        R_aligned_est[-1].apply([0, 0, 1], inverse=True)
                    )
                    tilt_error = np.arccos(scalar_product)
                    if tilt_error == "nan":
                        tilt_error = 0
                    R_error = R_aligned_mocap[-1] * R_aligned_est[-1].inv()

                    min_y = min(min_y, np.min(euler))
                    max_y = max(max_y, np.max(euler))
                    min_y = min(min_y, np.min(p_aligned_est))
                    max_y = max(max_y, np.max(p_aligned_est))

                    tilt_error = np.rad2deg(tilt_error)
                    d[mult][estimator]["pos"].append(error_pos)
                    d[mult][estimator]["tilt"].append(tilt_error)
                    yaw_error = np.array(abs(R_error.as_euler("zxy", degrees=True)[0]))
                    d[mult][estimator]["yaw"].append(yaw_error) 

                # Separator to avoid straight lines across windows
                idx_range.append(None)
                for est in estimatorsList:
                    for k in ["tx", "ty", "tz", "rz", "rx", "ry"]:
                        poses[est][k].append(None)

            y_low = min_y * 1.05
            y_high = max_y * 1.05
            xv = []
            yv = []
            for iv in intervals_by_mult[mult]:
                xv += [iv["end_time"], iv["end_time"], None]
                yv += [y_low, y_high, None]

            r = row_for_mult[mult]
            for iv in intervals_by_mult[mult]:
                fig.add_shape(
                    type="line",
                    x0=iv["start_time"],
                    x1=iv["start_time"],
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="y domain",
                    line=dict(dash="dash", width=1, color="black"),
                    layer="above",
                    row=r,
                    col=1
                )

            r = row_for_mult[mult]
            for estimator in estimatorsList:
                leg = show_legend_traj
                fig.add_trace(
                    go.Scatter(x=idx_range, y=poses[estimator]["tx"], mode="lines",
                            line=dict(color=colors[estimator]), name=f"{estimator}",
                            showlegend=leg),
                    row=r, col=1
                )
                # fig.add_trace(
                #     go.Scatter(x=idx_range, y=poses[estimator]["ty"], mode="lines",
                #             line=dict(color=colors[estimator]), name=f"{estimator} aligned Y",
                #             showlegend=False),
                #     row=r, col=1
                # )
                # fig.add_trace(
                #     go.Scatter(x=idx_range, y=poses[estimator]["tz"], mode="lines",
                #             line=dict(color=colors[estimator]), name=f"{estimator} aligned Z",
                #             showlegend=False),
                #     row=r, col=1
                # )
                # fig.add_trace(
                #     go.Scatter(x=idx_range, y=poses[estimator]["rx"], mode="lines",
                #             line=dict(color=colors[estimator]), name=f"{estimator} aligned Roll",
                #             showlegend=False),
                #     row=r, col=1
                # )
                # fig.add_trace(
                #     go.Scatter(x=idx_range, y=poses[estimator]["ry"], mode="lines",
                #             line=dict(color=colors[estimator]), name=f"{estimator} aligned Pitch",
                #             showlegend=False),
                #     row=r, col=1
                # )
                # fig.add_trace(
                #     go.Scatter(x=idx_range, y=poses[estimator]["rz"], mode="lines",
                #             line=dict(color=colors[estimator]), name=f"{estimator} aligned Yaw",
                #             showlegend=False),
                #     row=r, col=1
                # )
            fig.update_yaxes(title_text="Contact Bands", showticklabels=False, row=r, col=1)
            show_legend_traj = False

        fig.update_layout(
            title="Contact states and trajectories across cycle multiples",
            xaxis=dict(title="Time Index"),
            legend=dict(title="Trajectories"),
            height=250 + 350 * 3
        )

        fig.show()


        with open(f'{path_to_project}/output_data/evals/error_walk_cycle.pickle', 'wb') as f:
            pickle.dump(d, f)


    ###############################  Criteria based on the local linear velocity  ###############################

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
        b,a = butter(N=2, Wn=15/(0.5*200), btype='low')

        kinematics['Mocap']['IMU'] = dict()
        kinematics['Mocap'][mocapBody]['linVel'] = filtfilt(b, a, kinematics['Mocap'][mocapBody]['position'], axis=0)
        kinematics['Mocap'][mocapBody]['linVel'] = np.diff(kinematics['Mocap'][mocapBody]['linVel'], axis=0) / timeStep_s
        kinematics['Mocap'][mocapBody]['linVel'] = np.vstack((zeros_row,kinematics['Mocap'][mocapBody]['linVel']))

        kinematics['Mocap']['IMU']['position'] =  kinematics['Mocap'][mocapBody]['position'] + kinematics['Mocap'][mocapBody]['R'].apply(posFbImu)
        kinematics['Mocap']['IMU']['R'] = kinematics['Mocap'][mocapBody]['R'] * rImuFb.inv()
        kinematics['Mocap']['IMU']['linVel'] = filtfilt(b, a, kinematics['Mocap']['IMU']['position'], axis=0)
        kinematics['Mocap']['IMU']['linVel'] = np.diff(kinematics['Mocap']['IMU']['linVel'], axis=0) / timeStep_s
        kinematics['Mocap']['IMU']['linVel'] = np.vstack((zeros_row,kinematics['Mocap']['IMU']['linVel']))


        kinematics['Mocap'][mocapBody]['locLinVel'] = kinematics['Mocap'][mocapBody]['R'].apply(kinematics['Mocap'][mocapBody]['linVel'], inverse=True)
        kinematics['Mocap']['IMU']['locLinVel'] = kinematics['Mocap']['IMU']['R'].apply(kinematics['Mocap'][mocapBody]['linVel'], inverse=True)

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
            if estimator != "Hartley":
                d[estimator] = dict()

                kinematics[estimator][mocapBody]['llve'] = filtfilt(b, a, kinematics[estimator][mocapBody]['position'], axis=0)
                kinematics[estimator][mocapBody]['llve'] = np.diff(kinematics[estimator][mocapBody]['llve'], axis=0)/timeStep_s
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
            
        if("Hartley" in estimatorsList):
            d['Hartley'] = dict()
            kinematics["Hartley"][mocapBody]['linVel'] = filtfilt(b, a, kinematics["Hartley"][mocapBody]['position'], axis=0)
            kinematics["Hartley"][mocapBody]['linVel'] = np.diff(kinematics["Hartley"][mocapBody]['linVel'], axis=0)/timeStep_s
            kinematics["Hartley"][mocapBody]['linVel'] = np.vstack((zeros_row,kinematics["Hartley"][mocapBody]['linVel']))
            kinematics["Hartley"][mocapBody]['linVel'] = kinematics["Hartley"][mocapBody]["R"].apply(kinematics["Hartley"][mocapBody]['linVel'], inverse=True)

            kinematics["Hartley"][mocapBody]["locLinVel"] = kinematics["Hartley"][mocapBody]["R"].apply(kinematics["Hartley"][mocapBody]["linVel"], inverse=True)

            kinematics["Hartley"]["IMU"]["linVel"] = data_df[['Hartley_IMU_linVel_x', 'Hartley_IMU_linVel_y', 'Hartley_IMU_linVel_z']].to_numpy()
            kinematics["Hartley"]["IMU"]["locLinVel"] = kinematics["Hartley"]["IMU"]["R"].apply(kinematics["Hartley"]["IMU"]["linVel"], inverse=True)

            if(displayLogs):
                figLocLinVels.add_trace(go.Scatter(x=data_df["t"], y=kinematics["Hartley"][vel_eval_body]["locLinVel"][:,0], mode='lines', line=dict(color = colors["Hartley"]), name=f'locLinVel_{vel_eval_body}_{"Hartley"}_x'))
                figLocLinVels.add_trace(go.Scatter(x=data_df["t"], y=kinematics["Hartley"][vel_eval_body]["locLinVel"][:,1], mode='lines', line=dict(color = colors["Hartley"]), name=f'locLinVel_{vel_eval_body}_{"Hartley"}_y'))
                figLocLinVels.add_trace(go.Scatter(x=data_df["t"], y=kinematics["Hartley"][vel_eval_body]["locLinVel"][:,2], mode='lines', line=dict(color = colors["Hartley"]), name=f'locLinVel_{vel_eval_body}_{"Hartley"}_z'))

            d['Hartley']['llve'] = {'x': kinematics["Hartley"][mocapBody]["locLinVel"][:, 0], 'y': kinematics["Hartley"][mocapBody]["locLinVel"][:, 1], 'z': kinematics["Hartley"][mocapBody]["locLinVel"][:, 2]}
            d['Hartley']['estimate'] = {'x': kinematics["Hartley"]["IMU"]["locLinVel"][:, 0], 'y': kinematics["Hartley"]["IMU"]["locLinVel"][:, 1], 'z': kinematics["Hartley"]["IMU"]["locLinVel"][:, 2]}

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