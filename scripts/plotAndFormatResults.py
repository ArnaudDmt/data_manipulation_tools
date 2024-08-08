import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

import os.path


###############################  Main variables initialization  ###############################


displayLogs = True

withKO = False
withVanyte = False
withTilt = False
withMocap = False
withController = False
withHartley = False

path_to_project = ".."

if(len(sys.argv) > 1):
    displayLogs = sys.argv[2].lower() == 'true'
    if(len(sys.argv) > 2):
        path_to_project = sys.argv[2]


# Read the CSV file into a DataFrame

dfObservers = pd.read_csv(f'{path_to_project}/output_data/observerResultsCSV.csv', delimiter=';')

if os.path.isfile(f'{path_to_project}/output_data/HartleyOutput.csv') and 'HartleyIEKF_imuFbKine_position_x' in dfObservers.columns:
    dfHartley = pd.read_csv(f'{path_to_project}/output_data/HartleyOutput.csv', delimiter=';')
    withHartley = True
else:
    sys.exit(0)




dfObservers.rename(columns=lambda x: x.replace('Observers_MainObserverPipeline_MCKineticsObserver_mcko_fb_posW', 'KO'), inplace=True)
dfObservers.rename(columns=lambda x: x.replace('MCKineticsObserver_globalWorldCentroidState', 'KoState'), inplace=True)
dfObservers.rename(columns=lambda x: x.replace('Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world_pose', 'Vanyte'), inplace=True)    
dfObservers.rename(columns=lambda x: x.replace('Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world_pose', 'Vanyte'), inplace=True)
dfObservers.rename(columns=lambda x: x.replace('Observers_MainObserverPipeline_Tilt_FloatingBase_world_pose', 'Tilt'), inplace=True)
dfObservers.rename(columns=lambda x: x.replace('Observers_MainObserverPipeline_MocapVisualizer_mocap', 'Mocap'), inplace=True)
dfObservers.rename(columns=lambda x: x.replace('Observers_MainObserverPipeline_MocapVisualizer_worldFb', 'Mocap'), inplace=True)
dfObservers.rename(columns=lambda x: x.replace('ff', 'Controller'), inplace=True)



if 'KO_tx' in dfObservers.columns:
    withKO = True
if 'Vanyte_tx' in dfObservers.columns:
    withVanyte = True
if 'Tilt_tx' in dfObservers.columns:
    withTilt = True
if 'Mocap_pos_x' in dfObservers.columns:
    withMocap = True
if 'Controller_tx' in dfObservers.columns:
    withController = True


###############################  Definition  ###############################

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


###############################  Plots  ###############################



if(withMocap):
    mocap = dfObservers[dfObservers["Mocap_datasOverlapping"] == "Datas overlap"]
    df_mocap_toIgnore = dfObservers[dfObservers["Mocap_datasOverlapping"] != "Datas overlap"]

    posMocap_overlap = mocap[['Mocap_pos_x', 'Mocap_pos_y', 'Mocap_pos_z']].to_numpy()
    quaternionsMocap_overlap = mocap[['Mocap_ori_x', 'Mocap_ori_y', 'Mocap_ori_z', 'Mocap_ori_w']].to_numpy()
    # Compute the conjugate of the quaternions
    quaternionsMocap_overlap_conjugate = quaternionsMocap_overlap.copy()
    quaternionsMocap_overlap_conjugate[:, :3] *= -1
    # Convert the conjugate quaternions to Euler angles
    rMocap_overlap = R.from_quat(quaternionsMocap_overlap_conjugate)
    euler_angles_Mocap_overlap = rMocap_overlap.as_euler('xyz')
    euler_angles_Mocap_overlap = continuous_euler(euler_angles_Mocap_overlap)

    dfMocapPose = pd.DataFrame(columns=['#', 'timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    dfMocapPose['timestamp'] = dfObservers['t']
    dfMocapPose['tx'] = posMocap_overlap[:,0]
    dfMocapPose['ty'] = posMocap_overlap[:,1]
    dfMocapPose['tz'] = posMocap_overlap[:,2]
    dfMocapPose['qx'] = quaternionsMocap_overlap_conjugate[:,0]
    dfMocapPose['qy'] = quaternionsMocap_overlap_conjugate[:,1]
    dfMocapPose['qz'] = quaternionsMocap_overlap_conjugate[:,2]
    dfMocapPose['qw'] = quaternionsMocap_overlap_conjugate[:,3]

    dfMocapPose = dfMocapPose[dfObservers["Mocap_datasOverlapping"] == "Datas overlap"]

    txtOutput = f'{path_to_project}/output_data/formattedMocapTraj.txt'
    dfMocapPose.to_csv(txtOutput, header=None, index=None, sep=' ')

    line = '# timestamp tx ty tz qx qy qz qw' 
    with open(txtOutput, 'r+') as file: 
        file_data = file.read() 
        file.seek(0, 0) 
        file.write(line + '\n' + file_data) 

    if(len(df_mocap_toIgnore) > 0):
        posMocap_mocap_toIgnore = df_mocap_toIgnore[['Mocap_pos_x', 'Mocap_pos_y', 'Mocap_pos_z']].to_numpy()
        quaternionsMocap_mocap_toIgnore = df_mocap_toIgnore[['Mocap_ori_x', 'Mocap_ori_y', 'Mocap_ori_z', 'Mocap_ori_w']].to_numpy()
        # Compute the conjugate of the quaternions
        quaternionsMocap_mocap_toIgnore_conjugate = quaternionsMocap_mocap_toIgnore.copy()
        quaternionsMocap_mocap_toIgnore_conjugate[:, :3] *= -1
        # Convert the conjugate quaternions to Euler angles
        rMocap_mocap_toIgnore = R.from_quat(quaternionsMocap_mocap_toIgnore_conjugate)
        euler_angles_Mocap_mocap_toIgnore = rMocap_mocap_toIgnore.as_euler('xyz')
        euler_angles_Mocap_mocap_toIgnore = continuous_euler(euler_angles_Mocap_mocap_toIgnore)
        
if(withKO):
    posKO = dfObservers[['KO_tx', 'KO_ty', 'KO_tz']].to_numpy()
    quaternionsKO = dfObservers[['KO_qx', 'KO_qy', 'KO_qz', 'KO_qw']].to_numpy()
    # Compute the conjugate of the quaternions
    quaternionsKO_conjugate = quaternionsKO.copy()
    quaternionsKO_conjugate[:, :3] *= -1
    # Convert the conjugate quaternions to Euler angles
    rKO = R.from_quat(quaternionsKO_conjugate)
    euler_angles_KO = rKO.as_euler('xyz')
    euler_angles_KO = continuous_euler(euler_angles_KO)

    rKo_fb_quat = rKO.as_quat()

    dfKoPose = pd.DataFrame(columns=['#', 'timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    dfKoPose['timestamp'] = dfObservers['t']
    dfKoPose['tx'] = posKO[:,0]
    dfKoPose['ty'] = posKO[:,1]
    dfKoPose['tz'] = posKO[:,2]
    dfKoPose['qx'] = rKo_fb_quat[:,0]
    dfKoPose['qy'] = rKo_fb_quat[:,1]
    dfKoPose['qz'] = rKo_fb_quat[:,2]
    dfKoPose['qw'] = rKo_fb_quat[:,3]

    dfKoPose = dfKoPose[dfObservers["Mocap_datasOverlapping"] == "Datas overlap"]

    txtOutput = f'{path_to_project}/output_data/formattedKoTraj.txt'
    dfKoPose.to_csv(txtOutput, header=None, index=None, sep=' ')

    line = '# timestamp tx ty tz qx qy qz qw' 
    with open(txtOutput, 'r+') as file: 
        file_data = file.read() 
        file.seek(0, 0) 
        file.write(line + '\n' + file_data) 

if(withVanyte):
    posVanyte = dfObservers[['Vanyte_tx', 'Vanyte_ty', 'Vanyte_tz']].to_numpy()
    quaternionsVanyte = dfObservers[['Vanyte_qx', 'Vanyte_qy', 'Vanyte_qz', 'Vanyte_qw']].to_numpy()
    # Compute the conjugate of the quaternions
    quaternionsVanyte_conjugate = quaternionsVanyte.copy()
    quaternionsVanyte_conjugate[:, :3] *= -1
    # Convert the conjugate quaternions to Euler angles
    rVanyte = R.from_quat(quaternionsVanyte_conjugate)
    euler_angles_vanyte = rVanyte.as_euler('xyz')
    euler_angles_vanyte = continuous_euler(euler_angles_vanyte)

    rVanyte_fb_quat = rVanyte.as_quat()

    dfVanytePose = pd.DataFrame(columns=['#', 'timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    dfVanytePose['timestamp'] = dfObservers['t']
    dfVanytePose['tx'] = posVanyte[:,0]
    dfVanytePose['ty'] = posVanyte[:,1]
    dfVanytePose['tz'] = posVanyte[:,2]
    dfVanytePose['qx'] = rVanyte_fb_quat[:,0]
    dfVanytePose['qy'] = rVanyte_fb_quat[:,1]
    dfVanytePose['qz'] = rVanyte_fb_quat[:,2]
    dfVanytePose['qw'] = rVanyte_fb_quat[:,3]

    dfVanytePose = dfVanytePose[dfObservers["Mocap_datasOverlapping"] == "Datas overlap"]

    txtOutput = f'{path_to_project}/output_data/formattedVanyteTraj.txt'
    dfVanytePose.to_csv(txtOutput, header=None, index=None, sep=' ')

    line = '# timestamp tx ty tz qx qy qz qw' 
    with open(txtOutput, 'r+') as file: 
        file_data = file.read() 
        file.seek(0, 0) 
        file.write(line + '\n' + file_data) 

if(withTilt):
    posTilt = dfObservers[['Tilt_tx', 'Tilt_ty', 'Tilt_tz']].to_numpy()
    quaternionsTilt = dfObservers[['Tilt_qx', 'Tilt_qy', 'Tilt_qz', 'Tilt_qw']].to_numpy()
    # Compute the conjugate of the quaternions
    quaternionsTilt_conjugate = quaternionsTilt.copy()
    quaternionsTilt_conjugate[:, :3] *= -1
    # Convert the conjugate quaternions to Euler angles
    rTilt = R.from_quat(quaternionsTilt_conjugate)
    euler_angles_Tilt = rTilt.as_euler('xyz')
    euler_angles_Tilt = continuous_euler(euler_angles_Tilt)

    rTilt_fb_quat = rTilt.as_quat()

    dfTiltPose = pd.DataFrame(columns=['#', 'timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    dfTiltPose['timestamp'] = dfObservers['t']
    dfTiltPose['tx'] = posTilt[:,0]
    dfTiltPose['ty'] = posTilt[:,1]
    dfTiltPose['tz'] = posTilt[:,2]
    dfTiltPose['qx'] = rTilt_fb_quat[:,0]
    dfTiltPose['qy'] = rTilt_fb_quat[:,1]
    dfTiltPose['qz'] = rTilt_fb_quat[:,2]
    dfTiltPose['qw'] = rTilt_fb_quat[:,3]

    dfTiltPose = dfTiltPose[dfObservers["Mocap_datasOverlapping"] == "Datas overlap"]

    txtOutput = f'{path_to_project}/output_data/formattedTiltTraj.txt'
    dfTiltPose.to_csv(txtOutput, header=None, index=None, sep=' ')

    line = '# timestamp tx ty tz qx qy qz qw' 
    with open(txtOutput, 'r+') as file: 
        file_data = file.read() 
        file.seek(0, 0) 
        file.write(line + '\n' + file_data) 

if(withController):
    posController = dfObservers[['Controller_tx', 'Controller_ty', 'Controller_tz']].to_numpy()
    quaternionsController = dfObservers[['Controller_qx', 'Controller_qy', 'Controller_qz', 'Controller_qw']].to_numpy()
    # Compute the conjugate of the quaternions
    quaternionsController_conjugate = quaternionsController.copy()
    quaternionsController_conjugate[:, :3] *= -1
    # Convert the conjugate quaternions to Euler angles
    rController = R.from_quat(quaternionsController_conjugate)
    euler_angles_Controller = rController.as_euler('xyz')
    euler_angles_Controller = continuous_euler(euler_angles_Controller)

    rController_fb_quat = rController.as_quat()

    dfControllerPose = pd.DataFrame(columns=['#', 'timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    dfControllerPose['timestamp'] = dfObservers['t']
    dfControllerPose['tx'] = posController[:,0]
    dfControllerPose['ty'] = posController[:,1]
    dfControllerPose['tz'] = posController[:,2]
    dfControllerPose['qx'] = rController_fb_quat[:,0]
    dfControllerPose['qy'] = rController_fb_quat[:,1]
    dfControllerPose['qz'] = rController_fb_quat[:,2]
    dfControllerPose['qw'] = rController_fb_quat[:,3]

    dfControllerPose = dfControllerPose[dfObservers["Mocap_datasOverlapping"] == "Datas overlap"]

    txtOutput = f'{path_to_project}/output_data/formattedControllerTraj.txt'
    dfControllerPose.to_csv(txtOutput, header=None, index=None, sep=' ')

    line = '# timestamp tx ty tz qx qy qz qw' 
    with open(txtOutput, 'r+') as file: 
        file_data = file.read() 
        file.seek(0, 0) 
        file.write(line + '\n' + file_data) 


if(withHartley):
    # Convert quaternion to Euler angles (roll, pitch, yaw)
    posHartley_imu = dfHartley[['Position_x', 'Position_y', 'Position_z']].to_numpy()
    quaternionsHartley_imu = dfHartley[['Orientation_x', 'Orientation_y', 'Orientation_z', 'Orientation_w']].to_numpy()
    rHartley_imu = R.from_quat(quaternionsHartley_imu)

    posImuFb = dfObservers[['HartleyIEKF_imuFbKine_position_x', 'HartleyIEKF_imuFbKine_position_y', 'HartleyIEKF_imuFbKine_position_z']].to_numpy()
    quaternions_rImuFb = dfObservers[['HartleyIEKF_imuFbKine_ori_x', 'HartleyIEKF_imuFbKine_ori_y', 'HartleyIEKF_imuFbKine_ori_z', 'HartleyIEKF_imuFbKine_ori_w']].to_numpy()
    rImuFb = R.from_quat(quaternions_rImuFb)

    rHartley_fb = rHartley_imu * rImuFb
    posHartley_fb = posHartley_imu + rHartley_imu.apply(posImuFb)

    euler_angles_Hartley = rHartley_fb.as_euler('xyz')
    euler_angles_Hartley = continuous_euler(euler_angles_Hartley)

    rHartley_fb_quat = rHartley_fb.as_quat()

    dfHartley = pd.DataFrame(columns=['#', 'timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
    dfHartley['timestamp'] = dfObservers['t']
    dfHartley['tx'] = posHartley_fb[:,0]
    dfHartley['ty'] = posHartley_fb[:,1]
    dfHartley['tz'] = posHartley_fb[:,2]
    dfHartley['qx'] = rHartley_fb_quat[:,0]
    dfHartley['qy'] = rHartley_fb_quat[:,1]
    dfHartley['qz'] = rHartley_fb_quat[:,2]
    dfHartley['qw'] = rHartley_fb_quat[:,3]

    dfHartley = dfHartley[dfObservers["Mocap_datasOverlapping"] == "Datas overlap"]

    dfHartley.to_csv(f'{path_to_project}/output_data/formattedHartleyTraj.txt', header=None, index=None, sep=' ')

    line = '# timestamp tx ty tz qx qy qz qw' 
    with open(f'{path_to_project}/output_data/formattedHartleyTraj.txt', 'r+') as file: 
        file_data = file.read() 
        file.seek(0, 0) 
        file.write(line + '\n' + file_data) 




if(displayLogs):
    # Create the figure
    fig = go.Figure()


    # Add traces for each plot
    fig.add_trace(go.Scatter(x=dfHartley['t'], y=posHartley_fb[:, 0], mode='lines', name='Hartley_Position_x'))
    fig.add_trace(go.Scatter(x=dfHartley['t'], y=posHartley_fb[:, 1], mode='lines', name='Hartley_Position_y'))
    fig.add_trace(go.Scatter(x=dfHartley['t'], y=posHartley_fb[:, 2], mode='lines', name='Hartley_Position_z'))
    fig.add_trace(go.Scatter(x=dfHartley['t'], y=euler_angles_Hartley[:, 0], mode='lines', name='Hartley_Roll'))
    fig.add_trace(go.Scatter(x=dfHartley['t'], y=euler_angles_Hartley[:, 1], mode='lines', name='Hartley_Pitch'))
    fig.add_trace(go.Scatter(x=dfHartley['t'], y=euler_angles_Hartley[:, 2], mode='lines', name='Hartley_Yaw'))

    if(withVanyte):
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=posVanyte[:, 0], mode='lines', name='Vanyte_Position_x'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=posVanyte[:, 1], mode='lines', name='Vanyte_Position_y'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=posVanyte[:, 2], mode='lines', name='Vanyte_Position_z'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=euler_angles_vanyte[:, 0], mode='lines', name='Vanyte_Roll'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=euler_angles_vanyte[:, 1], mode='lines', name='Vanyte_Pitch'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=euler_angles_vanyte[:, 2], mode='lines', name='Vanyte_Yaw'))

    if(withKO):
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=posKO[:, 0], mode='lines', name='KineticsObserver_Position_x'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=posKO[:, 1], mode='lines', name='KineticsObserver_Position_y'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=posKO[:, 2], mode='lines', name='KineticsObserver_Position_z'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=euler_angles_KO[:, 0], mode='lines', name='KineticsObserver_Roll'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=euler_angles_KO[:, 1], mode='lines', name='KineticsObserver_Pitch'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=euler_angles_KO[:, 2], mode='lines', name='KineticsObserver_Yaw'))

    if(withTilt):
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=posTilt[:, 0], mode='lines', name='Tilt_Position_x'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=posTilt[:, 1], mode='lines', name='Tilt_Position_y'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=posTilt[:, 2], mode='lines', name='Tilt_Position_z'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=euler_angles_Tilt[:, 0], mode='lines', name='Tilt_Roll'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=euler_angles_Tilt[:, 1], mode='lines', name='Tilt_Pitch'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=euler_angles_Tilt[:, 2], mode='lines', name='Tilt_Yaw'))

    if(withMocap):
        fig.add_trace(go.Scatter(x=mocap['t'], y=posMocap_overlap[:, 0], mode='lines', name='Mocap_Position_x'))
        fig.add_trace(go.Scatter(x=mocap['t'], y=posMocap_overlap[:, 1], mode='lines', name='Mocap_Position_y'))
        fig.add_trace(go.Scatter(x=mocap['t'], y=posMocap_overlap[:, 2], mode='lines', name='Mocap_Position_z'))
        fig.add_trace(go.Scatter(x=mocap['t'], y=euler_angles_Mocap_overlap[:, 0], mode='lines', name='Mocap_Roll'))
        fig.add_trace(go.Scatter(x=mocap['t'], y=euler_angles_Mocap_overlap[:, 1], mode='lines', name='Mocap_Pitch'))
        fig.add_trace(go.Scatter(x=mocap['t'], y=euler_angles_Mocap_overlap[:, 2], mode='lines', name='Mocap_Yaw'))

        if(len(df_mocap_toIgnore) > 0):
            fig.add_trace(go.Scatter(x=mocap['t'], y=posMocap_mocap_toIgnore[:, 0], mode='lines', name='Mocap_mocap_toIgnore_Position_x'))
            fig.add_trace(go.Scatter(x=mocap['t'], y=posMocap_mocap_toIgnore[:, 1], mode='lines', name='Mocap_mocap_toIgnore_Position_y'))
            fig.add_trace(go.Scatter(x=mocap['t'], y=posMocap_mocap_toIgnore[:, 2], mode='lines', name='Mocap_mocap_toIgnore_Position_z'))
            fig.add_trace(go.Scatter(x=mocap['t'], y=euler_angles_Mocap_mocap_toIgnore[:, 0], mode='lines', name='Mocap_mocap_toIgnore_Roll'))
            fig.add_trace(go.Scatter(x=mocap['t'], y=euler_angles_Mocap_mocap_toIgnore[:, 1], mode='lines', name='Mocap_mocap_toIgnore_Pitch'))
            fig.add_trace(go.Scatter(x=mocap['t'], y=euler_angles_Mocap_mocap_toIgnore[:, 2], mode='lines', name='Mocap_mocap_toIgnore_Yaw'))

    if(withController):
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=posController[:, 0], mode='lines', name='Controller_Position_x'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=posController[:, 1], mode='lines', name='Controller_Position_y'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=posController[:, 2], mode='lines', name='Controller_Position_z'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=euler_angles_Controller[:, 0], mode='lines', name='Controller_Roll'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=euler_angles_Controller[:, 1], mode='lines', name='Controller_Pitch'))
        fig.add_trace(go.Scatter(x=dfObservers['t'], y=euler_angles_Controller[:, 2], mode='lines', name='Controller_Yaw'))


    # Update layout
    fig.update_layout(
        title='Position and Euler Angles (Roll, Pitch, Yaw) over Time',
        xaxis_title='Time',
        yaxis_title='Value',
        hovermode='x'
    )
    # Show the interactive plot
    fig.show()


    fig2 = go.Figure()


    if(withVanyte):
        fig2.add_trace(go.Scatter(x=posVanyte[:, 0], y=posVanyte[:, 1], mode='lines', name='Vanyte_2dMotion_xy'))
    if(withKO):
        fig2.add_trace(go.Scatter(x=posKO[:, 0], y=posKO[:, 1], mode='lines', name='KineticsObserver_2dMotion_xy'))
    if(withTilt):
        fig2.add_trace(go.Scatter(x=posTilt[:, 0], y=posTilt[:, 1], mode='lines', name='Tilt_2dMotion_xy'))
    if(withMocap):
        fig2.add_trace(go.Scatter(x=posMocap_overlap[:, 0], y=posMocap_overlap[:, 1], mode='lines', name='Mocap_2dMotion_xy'))
        if(len(df_mocap_toIgnore) > 0):
            fig2.add_trace(go.Scatter(x=posMocap_mocap_toIgnore[:, 0], y=posMocap_mocap_toIgnore[:, 1], mode='lines', name='Mocap_mocap_toIgnore_2dMotion_xy'))
    if(withController):
        fig2.add_trace(go.Scatter(x=posController[:, 0], y=posController[:, 1], mode='lines', name='Controller_2dMotion_xy'))
    fig2.add_trace(go.Scatter(x=posHartley_fb[:,0], y=posHartley_fb[:,1], mode='lines', name='Hartley_2dMotion_xy'))

    # Update layout
    fig2.update_layout(
        title='Position x vs y',
        xaxis_title='x',
        yaxis_title='y',
        hovermode='x'
    )
    fig2.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    # Show the interactive plot
    fig2.show()


    fig3 = go.Figure()

    # Find the index where 't' is closest to 0
    index_0 = (dfObservers['t'] - 0).abs().idxmin()
    # Find the index where 't' is closest to 80
    index_80 = (dfObservers['t'] - 80).abs().idxmin()
    # Compute the average bias between these indexes
    average_bias_x_80 = dfObservers.loc[index_0:index_80, 'Accelerometer_angularVelocity_x'].mean()
    average_bias_x_tuple_80 = tuple(average_bias_x_80 for _ in range(len(dfObservers)))
    average_bias_y_80 = dfObservers.loc[index_0:index_80, 'Accelerometer_angularVelocity_y'].mean()
    average_bias_y_tuple_80 = tuple(average_bias_y_80 for _ in range(len(dfObservers)))
    average_bias_z_80 = dfObservers.loc[index_0:index_80, 'Accelerometer_angularVelocity_z'].mean()
    average_bias_z_tuple_80 = tuple(average_bias_z_80 for _ in range(len(dfObservers)))

    # Find the index where 't' is closest to 360
    index_360 = (dfObservers['t'] - 360).abs().idxmin()
    # Find the index where 't' is closest to 450
    index_450 = (dfObservers['t'] - 450).abs().idxmin()
    # Compute the average bias between these indexes
    average_bias_x_360 = dfObservers.loc[index_360:index_450, 'Accelerometer_angularVelocity_x'].mean()
    average_bias_x_tuple_360 = tuple(average_bias_x_360 for _ in range(len(dfObservers)))
    average_bias_y_360 = dfObservers.loc[index_360:index_450, 'Accelerometer_angularVelocity_y'].mean()
    average_bias_y_tuple_360 = tuple(average_bias_y_360 for _ in range(len(dfObservers)))
    average_bias_z_360 = dfObservers.loc[index_360:index_450, 'Accelerometer_angularVelocity_z'].mean()
    average_bias_z_tuple_360 = tuple(average_bias_z_360 for _ in range(len(dfObservers)))

    fig3.add_trace(go.Scatter(x=dfObservers['t'], y=dfObservers['Accelerometer_angularVelocity_x'], mode='lines', name='measured_angVel_x'))
    fig3.add_trace(go.Scatter(x=dfObservers['t'], y=dfObservers['Accelerometer_angularVelocity_y'], mode='lines', name='measured_angVel_y'))
    fig3.add_trace(go.Scatter(x=dfObservers['t'], y=dfObservers['Accelerometer_angularVelocity_z'], mode='lines', name='measured_angVel_z'))
    fig3.add_trace(go.Scatter(x=dfObservers['t'], y=average_bias_x_tuple_80, mode='lines', name='measured_GyroBias_beginning_x'))
    fig3.add_trace(go.Scatter(x=dfObservers['t'], y=average_bias_y_tuple_80, mode='lines', name='measured_GyroBias_beginning_y'))
    fig3.add_trace(go.Scatter(x=dfObservers['t'], y=average_bias_z_tuple_80, mode='lines', name='measured_GyroBias_beginning_z'))
    fig3.add_trace(go.Scatter(x=dfObservers['t'], y=average_bias_x_tuple_360, mode='lines', name='measured_GyroBias_secondStop_x'))
    fig3.add_trace(go.Scatter(x=dfObservers['t'], y=average_bias_y_tuple_360, mode='lines', name='measured_GyroBias_secondStop_y'))
    fig3.add_trace(go.Scatter(x=dfObservers['t'], y=average_bias_z_tuple_360, mode='lines', name='measured_GyroBias_secondStop_z'))
    fig3.add_trace(go.Scatter(x=dfHartley['t'], y=dfHartley['GyroBias_x'], mode='lines', name='Hartley_GyroBias_x'))
    fig3.add_trace(go.Scatter(x=dfHartley['t'], y=dfHartley['GyroBias_y'], mode='lines', name='Hartley_GyroBias_y'))
    fig3.add_trace(go.Scatter(x=dfHartley['t'], y=dfHartley['GyroBias_z'], mode='lines', name='Hartley_GyroBias_z'))
    if(withKO):
        fig3.add_trace(go.Scatter(x=dfObservers['t'], y=dfObservers['KoState_gyroBias_Accelerometer_x'], mode='lines', name='KO_GyroBias_x'))
        fig3.add_trace(go.Scatter(x=dfObservers['t'], y=dfObservers['KoState_gyroBias_Accelerometer_y'], mode='lines', name='KO_GyroBias_y'))
        fig3.add_trace(go.Scatter(x=dfObservers['t'], y=dfObservers['KoState_gyroBias_Accelerometer_z'], mode='lines', name='KO_GyroBias_z'))


    # Update layout
    fig3.update_layout(
        title='Gyrometer biases',
        xaxis_title='x',
        yaxis_title='y',
        hovermode='x'
    )

    # Show the interactive plot
    fig3.show()




    x_min = min((posHartley_fb[:,0]).min(), (posHartley_fb[:,0]).min(), (posHartley_fb[:,0]).min())
    y_min = min((posHartley_fb[:,1]).min(), (posHartley_fb[:,1]).min(), (posHartley_fb[:,1]).min())
    z_min = min((posHartley_fb[:,2]).min(), (posHartley_fb[:,2]).min(), (posHartley_fb[:,2]).min())

    x_max = max((posHartley_fb[:,0]).max(), (posHartley_fb[:,0]).max(), (posHartley_fb[:,0]).max())
    y_max = max((posHartley_fb[:,1]).max(), (posHartley_fb[:,1]).max(), (posHartley_fb[:,1]).max())
    z_max = max((posHartley_fb[:,2]).max(), (posHartley_fb[:,2]).max(), (posHartley_fb[:,2]).max())

    fig3d = go.Figure()

    # Add traces
    fig3d.add_trace(go.Scatter3d(
        x=posHartley_fb[:,0], 
        y=posHartley_fb[:,1], 
        z=posHartley_fb[:,2],
        mode='lines',
        line=dict(color='darkred'),
        name='Hartley'
    ))

    if(withVanyte):
        x_min = min(x_min, (posVanyte[:,0]).min())
        y_min = min(y_min, (posVanyte[:,1]).min())
        z_min = min(z_min, (posVanyte[:,2]).min())

        x_max = max(x_max, (posVanyte[:,0]).max())
        y_max = max(y_max, (posVanyte[:,1]).max())
        z_max = max(z_max, (posVanyte[:,2]).max())

        fig3d.add_trace(go.Scatter3d(
            x=posVanyte[:,0], 
            y=posVanyte[:,1], 
            z=posVanyte[:,2],
            mode='lines',
            line=dict(color='darkblue'),
            name='Vanyt-e'
        ))
    if(withKO):
        x_min = min(x_min, (posKO[:,0]).min())
        y_min = min(y_min, (posKO[:,1]).min())
        z_min = min(z_min, (posKO[:,2]).min())

        x_max = max(x_max, (posKO[:,0]).max())
        y_max = max(y_max, (posKO[:,1]).max())
        z_max = max(z_max, (posKO[:,2]).max())

        fig3d.add_trace(go.Scatter3d(
            x=posKO[:,0], 
            y=posKO[:,1], 
            z=posKO[:,2],
            mode='lines',
            line=dict(color='darkgreen'),
            name='Kinetics Observer'
        ))

    if(withTilt):
        x_min = min(x_min, (posTilt[:,0]).min())
        y_min = min(y_min, (posTilt[:,1]).min())
        z_min = min(z_min, (posTilt[:,2]).min())

        x_max = max(x_max, (posTilt[:,0]).max())
        y_max = max(y_max, (posTilt[:,1]).max())
        z_max = max(z_max, (posTilt[:,2]).max())

        fig3d.add_trace(go.Scatter3d(
            x=posTilt[:,0], 
            y=posTilt[:,1], 
            z=posTilt[:,2],
            mode='lines',
            line=dict(color='darkcyan'),
            name='Tilt Observer'
        ))

    if(withMocap):
        x_min = min(x_min, (posMocap_overlap[:,0]).min())
        y_min = min(y_min, (posMocap_overlap[:,1]).min())
        z_min = min(z_min, (posMocap_overlap[:,2]).min())

        x_max = max(x_max, (posMocap_overlap[:,0]).max())
        y_max = max(y_max, (posMocap_overlap[:,1]).max())
        z_max = max(z_max, (posMocap_overlap[:,2]).max())

        fig3d.add_trace(go.Scatter3d(
            x=posMocap_overlap[:,0], 
            y=posMocap_overlap[:,1], 
            z=posMocap_overlap[:,2],
            mode='lines',
            line=dict(color='darkviolet'),
            name='Motion capture'
        ))
        
        if(len(df_mocap_toIgnore) > 0):
            x_min = min(x_min, (posMocap_mocap_toIgnore[:,0]).min())
            y_min = min(y_min, (posMocap_mocap_toIgnore[:,1]).min())
            z_min = min(z_min, (posMocap_mocap_toIgnore[:,2]).min())

            x_max = max(x_max, (posMocap_mocap_toIgnore[:,0]).max())
            y_max = max(y_max, (posMocap_mocap_toIgnore[:,1]).max())
            z_max = max(z_max, (posMocap_mocap_toIgnore[:,2]).max())

            fig3d.add_trace(go.Scatter3d(
                x=posMocap_mocap_toIgnore[:,0], 
                y=posMocap_mocap_toIgnore[:,1], 
                z=posMocap_mocap_toIgnore[:,2],
                mode='lines',
                line=dict(width=0.5, color='darkviolet'),
                name='Motion capture to ignore'
            ))

    if(withController):
        x_min = min(x_min, (posController[:,0]).min())
        y_min = min(y_min, (posController[:,1]).min())
        z_min = min(z_min, (posController[:,2]).min())

        x_max = max(x_max, (posController[:,0]).max())
        y_max = max(y_max, (posController[:,1]).max())
        z_max = max(z_max, (posController[:,2]).max())

        fig3d.add_trace(go.Scatter3d(
            x=posController[:,0], 
            y=posController[:,1], 
            z=posController[:,2],
            mode='lines',
            line=dict(color='darkorange'),
            name='Controller'
        ))


    x_min = x_min - np.abs(x_min*0.2)
    y_min = y_min - np.abs(y_min*0.2)
    z_min = z_min - np.abs(z_min*0.2)
    x_max = x_max + np.abs(x_max*0.2)
    y_max = y_max + np.abs(y_max*0.2)
    z_max = z_max + np.abs(z_max*0.2)


    # Update layout
    fig3d.update_layout(
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
    )

    # Show the plot
    fig3d.show()


