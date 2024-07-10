import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.spatial.transform import Rotation as R




###############################  User input for the timestep  ###############################

timeStepInput = input("Please enter the timestep of the controller in milliseconds: ")

# Convert the input to a double
try:
    timeStepInt = int(timeStepInput)
    timeStepFloat = float(timeStepInput)*1000.0
    resample_str = f'{timeStepInt}ms'
except ValueError:
    print("That's not a valid int!")




###############################  Main variables initialization  ###############################

output_csv_file_path = 'realignedMocapHeadData.csv'



###############################  Function definitions  ###############################


def convert_mm_to_m(dataframe):
    for col in dataframe.columns:
        if 'pos' in col:
            dataframe[col] = dataframe[col] / 1000
    return dataframe

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


# Load the CSV files into pandas dataframes
observer_data = pd.read_csv('lightData.csv')
mocapData = pd.read_csv('resampledMocapData.csv', delimiter=',')

print(len(observer_data))
print(len(mocapData))

import warnings
if(len(observer_data) != len(mocapData)):
    warnings.warn("The size of the two logs is different, there might be an issue.")

###############################  Poses retrieval  ###############################

# Extracting the poses related to the mocap
world_mocapRigidBody_Pos = np.array([mocapData['RigidBody001_tX'], mocapData['RigidBody001_tY'], mocapData['RigidBody001_tZ']]).T
world_RigidBody_Ori_R = R.from_quat(mocapData[["RigidBody001_qX", "RigidBody001_qY", "RigidBody001_qZ", "RigidBody001_qW"]].values)

world_mocapHead_Pos = np.array([mocapData['world_Head_Pos_x'], mocapData['world_Head_Pos_y'], mocapData['world_Head_Pos_z']]).T
world_mocapHead_Ori_R = R.from_quat(mocapData[["world_Head_Ori_qx", "world_Head_Ori_qy", "world_Head_Ori_qz", "world_Head_Ori_qw"]].values)

headFb_Pos = np.array([observer_data['Observers_MainObserverPipeline_MocapVisualizer_mocap_bodyFbPose_pos_x'], observer_data['Observers_MainObserverPipeline_MocapVisualizer_mocap_bodyFbPose_pos_y'], observer_data['Observers_MainObserverPipeline_MocapVisualizer_mocap_bodyFbPose_pos_z']]).T
headFb_Ori_R = R.from_quat(observer_data[["Observers_MainObserverPipeline_MocapVisualizer_mocap_bodyFbPose_ori_x", "Observers_MainObserverPipeline_MocapVisualizer_mocap_bodyFbPose_ori_y", "Observers_MainObserverPipeline_MocapVisualizer_mocap_bodyFbPose_ori_z", "Observers_MainObserverPipeline_MocapVisualizer_mocap_bodyFbPose_ori_w"]].values)

world_mocapFb_Pos = world_mocapHead_Pos + world_mocapHead_Ori_R.apply(headFb_Pos)
world_mocapFb_Ori_R = world_mocapHead_Ori_R * headFb_Ori_R


# Extracting the poses coming from mc_rtc
world_VanyteFb_Pos = np.array([observer_data['Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world_pose_tx'], observer_data['Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world_pose_ty'], observer_data['Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world_pose_tz']]).T
world_VanyteFb_Ori_R = R.from_quat(observer_data[["Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world_pose_qx", "Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world_pose_qy", "Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world_pose_qz", "Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world_pose_qw"]].values)
# We get the inverse of the orientation as the inverse quaternion was stored
world_VanyteFb_Ori_R = world_VanyteFb_Ori_R.inv()



###############################  Visualization of the extracted poses  ###############################


# Plot of the resulting positions
figPositions = go.Figure()

figPositions.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world_pose_tx"], mode='lines', name='world_Vanyte_fb_pos_x'))
figPositions.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world_pose_ty"], mode='lines', name='world_Vanyte_fb_pos_y'))
figPositions.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world_pose_tz"], mode='lines', name='world_Vanyte_fb_pos_z'))


figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapHead_Pos[:,0], mode='lines', name='world_mocapHead_Pos_x'))
figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapHead_Pos[:,1], mode='lines', name='world_mocapHead_Pos_y'))
figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapHead_Pos[:,2], mode='lines', name='world_mocapHead_Pos_z'))

figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=headFb_Pos[:,0], mode='lines', name='headFb_Pos_x'))
figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=headFb_Pos[:,1], mode='lines', name='headFb_Pos_y'))
figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=headFb_Pos[:,2], mode='lines', name='headFb_Pos_z'))

figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapFb_Pos[:,0], mode='lines', name='world_mocapFb_Pos_x'))
figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapFb_Pos[:,1], mode='lines', name='world_mocapFb_Pos_y'))
figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapFb_Pos[:,2], mode='lines', name='world_mocapFb_Pos_z'))

figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=mocapData["RigidBody001_tX"], mode='lines', name='world_RigidBody_pos_x'))
figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=mocapData["RigidBody001_tY"], mode='lines', name='world_RigidBody_pos_y'))
figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=mocapData["RigidBody001_tZ"], mode='lines', name='world_RigidBody_pos_z'))

figPositions.update_layout(title="Resulting positions")

# Show the plotly figures
# figPositions.show()


# Plot of the resulting orientations
figOrientations = go.Figure()

world_VanyteFb_Ori_euler = world_VanyteFb_Ori_R.as_euler("xyz")
world_mocapFb_Ori_euler = world_mocapFb_Ori_R.as_euler("xyz")
world_mocapHead_Ori_euler = world_mocapHead_Ori_R.as_euler("xyz")
world_RigidBody_Ori_euler = world_RigidBody_Ori_R.as_euler("xyz")
headFb_Ori_euler = headFb_Ori_R.as_euler("xyz")

world_VanyteFb_Ori_euler_continuous = continuous_euler(world_VanyteFb_Ori_euler)
world_mocapFb_Ori_euler_continuous = continuous_euler(world_mocapFb_Ori_euler)
world_mocapHead_Ori_euler_continuous = continuous_euler(world_mocapHead_Ori_euler)
world_RigidBody_Ori_euler_continuous = continuous_euler(world_RigidBody_Ori_euler)
headFb_Ori_euler_continuous = continuous_euler(headFb_Ori_euler)

figOrientations.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteFb_Ori_euler_continuous[:,0], mode='lines', name='world_Vanyte_fb_ori_roll'))
figOrientations.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteFb_Ori_euler_continuous[:,1], mode='lines', name='world_Vanyte_fb_ori_pitch'))
figOrientations.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteFb_Ori_euler_continuous[:,2], mode='lines', name='world_Vanyte_fb_ori_yaw'))

figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapHead_Ori_euler_continuous[:,0], mode='lines', name='world_mocapHead_Ori_roll'))
figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapHead_Ori_euler_continuous[:,1], mode='lines', name='world_mocapHead_Ori_pitch'))
figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapHead_Ori_euler_continuous[:,2], mode='lines', name='world_mocapHead_Ori_yaw'))

figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=headFb_Ori_euler_continuous[:,0], mode='lines', name='headFb_Ori_roll'))
figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=headFb_Ori_euler_continuous[:,1], mode='lines', name='headFb_Ori_pitch'))
figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=headFb_Ori_euler_continuous[:,2], mode='lines', name='headFb_Ori_yaw'))

figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapFb_Ori_euler_continuous[:,0], mode='lines', name='world_mocapFb_Ori_roll'))
figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapFb_Ori_euler_continuous[:,1], mode='lines', name='world_mocapFb_Ori_pitch'))
figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapFb_Ori_euler_continuous[:,2], mode='lines', name='world_mocapFb_Ori_yaw'))

figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,0], mode='lines', name='world_RigidBody_ori_roll'))
figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,1], mode='lines', name='world_RigidBody_ori_pitch'))
figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,2], mode='lines', name='world_RigidBody_ori_yaw'))


error_Vanyte_Vs_Mocap_fb_R = world_VanyteFb_Ori_R.inv() * world_mocapFb_Ori_R
error_Vanyte_Vs_Mocap_fb_euler = error_Vanyte_Vs_Mocap_fb_R.as_euler("xyz")

figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=error_Vanyte_Vs_Mocap_fb_euler[:,0], mode='lines', name='error_Vanyte_Vs_Mocap_fb_roll'))
figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=error_Vanyte_Vs_Mocap_fb_euler[:,1], mode='lines', name='error_Vanyte_Vs_Mocap_fb_pitch'))
figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=error_Vanyte_Vs_Mocap_fb_euler[:,2], mode='lines', name='error_Vanyte_Vs_Mocap_fb_yaw'))

figOrientations.update_layout(title="Resulting orientations")

# Show the plotly figures
figOrientations.show()



###############################  Local linear velocity of the head in the world  ###############################

# We compute the velocity of the head in the world
world_mocapFb_Vel_x = np.diff(world_mocapFb_Pos[:,0])/timeStepFloat*1000.0
world_mocapFb_Vel_y = np.diff(world_mocapFb_Pos[:,1])/timeStepFloat*1000.0
world_mocapFb_Vel_z = np.diff(world_mocapFb_Pos[:,2])/timeStepFloat*1000.0
world_mocapFb_Vel_x = np.insert(world_mocapFb_Vel_x, 0, 0.0, axis=0)
world_mocapFb_Vel_y = np.insert(world_mocapFb_Vel_y, 0, 0.0, axis=0)
world_mocapFb_Vel_z = np.insert(world_mocapFb_Vel_z, 0, 0.0, axis=0)
world_mocapFb_Vel = np.stack((world_mocapFb_Vel_x, world_mocapFb_Vel_y, world_mocapFb_Vel_z), axis = 1)

# We compute the velocity of the mocap's rigid body in the world
world_RigidBody_Vel_x = np.diff(world_mocapRigidBody_Pos[:,0])/timeStepFloat*1000.0
world_RigidBody_Vel_y = np.diff(world_mocapRigidBody_Pos[:,1])/timeStepFloat*1000.0
world_RigidBody_Vel_z = np.diff(world_mocapRigidBody_Pos[:,2])/timeStepFloat*1000.0
world_RigidBody_Vel_x = np.insert(world_RigidBody_Vel_x, 0, 0.0, axis=0)
world_RigidBody_Vel_y = np.insert(world_RigidBody_Vel_y, 0, 0.0, axis=0)
world_RigidBody_Vel_z = np.insert(world_RigidBody_Vel_z, 0, 0.0, axis=0)
world_RigidBody_Vel = np.stack((world_RigidBody_Vel_x, world_RigidBody_Vel_y, world_RigidBody_Vel_z), axis = 1)

# We compute the velocity estimated by the Vanyte in the world
world_VanyteFb_Vel_x = np.diff(world_VanyteFb_Pos[:,0])/timeStepFloat*1000.0
world_VanyteFb_Vel_y = np.diff(world_VanyteFb_Pos[:,1])/timeStepFloat*1000.0
world_VanyteFb_Vel_z = np.diff(world_VanyteFb_Pos[:,2])/timeStepFloat*1000.0
world_VanyteFb_Vel_x = np.insert(world_VanyteFb_Vel_x, 0, 0.0, axis=0)
world_VanyteFb_Vel_y = np.insert(world_VanyteFb_Vel_y, 0, 0.0, axis=0)
world_VanyteFb_Vel_z = np.insert(world_VanyteFb_Vel_z, 0, 0.0, axis=0)
world_VanyteFb_Vel = np.stack((world_VanyteFb_Vel_x, world_VanyteFb_Vel_y, world_VanyteFb_Vel_z), axis = 1)


# Now we get the local linear velocity
world_mocapFb_LocVel = world_mocapFb_Ori_R.apply(world_mocapFb_Vel, inverse=True)
world_RigidBody_LocVel = world_RigidBody_Ori_R.apply(world_RigidBody_Vel, inverse=True)
world_VanyteFb_LocVel = world_VanyteFb_Ori_R.apply(world_VanyteFb_Vel, inverse=True)

# Plot of the resulting poses
fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapFb_LocVel[:,0], mode='lines', name='world_mocapFb_LocVel_x'))
fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapFb_LocVel[:,1], mode='lines', name='world_mocapFb_LocVel_y'))
fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapFb_LocVel[:,2], mode='lines', name='world_mocapFb_LocVel_z'))
fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_LocVel[:,0], mode='lines', name='world_RigidBody_LocalLinVel_x'))
fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_LocVel[:,1], mode='lines', name='world_RigidBody_LocalLinVel_y'))
fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_LocVel[:,2], mode='lines', name='world_RigidBody_LocalLinVel_z'))
fig2.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteFb_LocVel[:,0], mode='lines', name='world_VanyteFb_LocVel_x'))
fig2.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteFb_LocVel[:,1], mode='lines', name='world_VanyteFb_LocVel_y'))
fig2.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteFb_LocVel[:,2], mode='lines', name='world_VanyteFb_LocVel_z'))

fig2.update_layout(title="Local linear velocity of the head in the world / vs the one of the rigid body")
# Show the plotly figures
# fig2.show()




###############################  Cross correlation  ###############################

def realignData(data1, data2, data1_name, data2_name):
    fig = go.Figure()
    # Remove the mean from the signals
    data1 = data1 - np.mean(data1)
    data2 = data2 - np.mean(data2)

    # Find the index of the maximum value in the cross-correlation of the two signals
    max_cross_corr = 0
    for i in range(data1.shape[1]):
        crosscorr = np.correlate(data1[:,i], data2[:,i], mode='full')
        if(np.argmax(crosscorr) > max_cross_corr):
            max_index = np.argmax(crosscorr)

    
    # Shift the second observer_data file by the calculated index
    shift = max_index - (data1.shape[0] - 1)
    data2_shifted = np.roll(data2, shift, axis=0)

    fig.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=data1[:,0], mode='lines', name=f'{data1_name}_1'))
    fig.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=data1[:,1], mode='lines', name=f'{data1_name}_2'))
    fig.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=data1[:,2], mode='lines', name=f'{data1_name}_3'))

    fig.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=data2_shifted[:,0], mode='lines', name=f'{data2_name}_1'))
    fig.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=data2_shifted[:,1], mode='lines', name=f'{data2_name}_2'))
    fig.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=data2_shifted[:,2], mode='lines', name=f'{data2_name}_3'))

    fig.show()

    print(f"The observer_data will be shifted by {shift} indexes.")
    return data2, shift


world_mocapFb_LocVel, shift = realignData(world_VanyteFb_LocVel, world_mocapFb_LocVel, "world_mocapFb_LocVel", "world_VanyteFb_LocVel")


# # Version which receives the shift to apply as an input
# def realignData(data1, data2, data1_name, data2_name, shift):
#     fig = go.Figure()

#     data2_shifted = np.roll(data2, shift, axis=0)
#     #data2_shifted = data2_shifted[shift:]

#     fig.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=data1[:,0], mode='lines', name=f'{data1_name}_1'))
#     fig.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=data1[:,1], mode='lines', name=f'{data1_name}_2'))
#     fig.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=data1[:,2], mode='lines', name=f'{data1_name}_3'))

#     fig.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=data2_shifted[:,0], mode='lines', name=f'{data2_name}_shifted_1'))
#     fig.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=data2_shifted[:,1], mode='lines', name=f'{data2_name}_shifted_2'))
#     fig.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=data2_shifted[:,2], mode='lines', name=f'{data2_name}_shifted_3'))

#     fig.show()

#     print(f"The observer_data will be shifted by {shift} indexes.")
#     return data2

# world_mocapFb_Ori_euler_continuous = realignData(world_VanyteFb_Ori_euler_continuous, world_mocapFb_Ori_euler_continuous, "world_VanyteFb_Ori_euler_continuous", "world_mocapFb_Ori_euler_continuous", shift)



# Version which receives the shift to apply as an input
def realignData(data_to_shift, data2,  shift):
    fig = go.Figure()

    # data_shifted = np.roll(observer_data, shift, axis=0)
    data_shifted = data_to_shift.shift(shift)

    #data_shifted = data_shifted[shift:]
    # Remove rows until the new index of the first row after the shift
    # data_shifted = data_shifted.iloc[shift:]
    # Set the values until the index of the first row after the shift to the value at index shift

    for col in data_shifted.columns:
        if(shift > 0):
            data_shifted.loc[:shift, col] = data_shifted.loc[shift, col]
        else:
            if (shift < 0):
                data_shifted.loc[-shift:, col] = data_shifted.loc[-shift, col]
    
    data_shifted["Time(Seconds)"] = data2["t"]

    print(f"The observer_data will be shifted by {shift} indexes.")
    return data_shifted

realignedMocapData = realignData(mocapData, observer_data, shift)


###############################  Shifted poses retrieval  ###############################

# Extracting the poses related to the realignedMocapData
world_mocapRigidBody_Pos = np.array([realignedMocapData['RigidBody001_tX'], realignedMocapData['RigidBody001_tY'], realignedMocapData['RigidBody001_tZ']]).T
world_RigidBody_Ori_R = R.from_quat(realignedMocapData[["RigidBody001_qX", "RigidBody001_qY", "RigidBody001_qZ", "RigidBody001_qW"]].values)

world_mocapHead_Pos = np.array([realignedMocapData['world_Head_Pos_x'], realignedMocapData['world_Head_Pos_y'], realignedMocapData['world_Head_Pos_z']]).T
world_mocapHead_Ori_R = R.from_quat(realignedMocapData[["world_Head_Ori_qx", "world_Head_Ori_qy", "world_Head_Ori_qz", "world_Head_Ori_qw"]].values)

world_mocapFb_Pos = world_mocapHead_Pos + world_mocapHead_Ori_R.apply(headFb_Pos)
world_mocapFb_Ori_R = world_mocapHead_Ori_R * headFb_Ori_R


###############################  Visualization of the extracted poses  ###############################


# Plot of the resulting positions
figPositions_realigned = go.Figure()

figPositions_realigned.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world_pose_tx"], mode='lines', name='world_Vanyte_fb_pos_x_realigned'))
figPositions_realigned.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world_pose_ty"], mode='lines', name='world_Vanyte_fb_pos_y_realigned'))
figPositions_realigned.add_trace(go.Scatter(x=observer_data["t"], y=observer_data["Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world_pose_tz"], mode='lines', name='world_Vanyte_fb_pos_z_realigned'))

figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapHead_Pos[:,0], mode='lines', name='world_mocapHead_Pos_x_realigned'))
figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapHead_Pos[:,1], mode='lines', name='world_mocapHead_Pos_y_realigned'))
figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapHead_Pos[:,2], mode='lines', name='world_mocapHead_Pos_z_realigned'))

figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapFb_Pos[:,0], mode='lines', name='world_mocapFb_Pos_x_realigned'))
figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapFb_Pos[:,1], mode='lines', name='world_mocapFb_Pos_y_realigned'))
figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapFb_Pos[:,2], mode='lines', name='world_mocapFb_Pos_z_realigned'))

figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=realignedMocapData["RigidBody001_tX"], mode='lines', name='world_RigidBody_pos_x_realigned'))
figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=realignedMocapData["RigidBody001_tY"], mode='lines', name='world_RigidBody_pos_y_realigned'))
figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=realignedMocapData["RigidBody001_tZ"], mode='lines', name='world_RigidBody_pos_z_realigned'))

figPositions_realigned.update_layout(title="Realigned positions")

# Show the plotly figures
figPositions_realigned.show()


# Plot of the resulting orientations
figOrientations_realigned = go.Figure()

world_VanyteFb_Ori_euler = world_VanyteFb_Ori_R.as_euler("xyz")
world_mocapFb_Ori_euler = world_mocapFb_Ori_R.as_euler("xyz")
world_mocapHead_Ori_euler = world_mocapHead_Ori_R.as_euler("xyz")
world_RigidBody_Ori_euler = world_RigidBody_Ori_R.as_euler("xyz")
headFb_Ori_euler = headFb_Ori_R.as_euler("xyz")

world_VanyteFb_Ori_euler_continuous = continuous_euler(world_VanyteFb_Ori_euler)
world_mocapFb_Ori_euler_continuous = continuous_euler(world_mocapFb_Ori_euler)
world_mocapHead_Ori_euler_continuous = continuous_euler(world_mocapHead_Ori_euler)
world_RigidBody_Ori_euler_continuous = continuous_euler(world_RigidBody_Ori_euler)
headFb_Ori_euler_continuous = continuous_euler(headFb_Ori_euler)

figOrientations_realigned.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteFb_Ori_euler_continuous[:,0], mode='lines', name='world_Vanyte_fb_ori_roll_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteFb_Ori_euler_continuous[:,1], mode='lines', name='world_Vanyte_fb_ori_pitch_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteFb_Ori_euler_continuous[:,2], mode='lines', name='world_Vanyte_fb_ori_yaw_realigned'))

figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapHead_Ori_euler_continuous[:,0], mode='lines', name='world_mocapHead_Ori_roll_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapHead_Ori_euler_continuous[:,1], mode='lines', name='world_mocapHead_Ori_pitch_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapHead_Ori_euler_continuous[:,2], mode='lines', name='world_mocapHead_Ori_yaw_realigned'))

figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=headFb_Ori_euler_continuous[:,0], mode='lines', name='headFb_Ori_roll_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=headFb_Ori_euler_continuous[:,1], mode='lines', name='headFb_Ori_pitch_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=headFb_Ori_euler_continuous[:,2], mode='lines', name='headFb_Ori_yaw_realigned'))

figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapFb_Ori_euler_continuous[:,0], mode='lines', name='world_mocapFb_Ori_roll_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapFb_Ori_euler_continuous[:,1], mode='lines', name='world_mocapFb_Ori_pitch_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapFb_Ori_euler_continuous[:,2], mode='lines', name='world_mocapFb_Ori_yaw_realigned'))

figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,0], mode='lines', name='world_RigidBody_ori_roll_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,1], mode='lines', name='world_RigidBody_ori_pitch_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,2], mode='lines', name='world_RigidBody_ori_yaw_realigned'))


error_Vanyte_Vs_Mocap_fb_R = world_VanyteFb_Ori_R.inv() * world_mocapFb_Ori_R
error_Vanyte_Vs_Mocap_fb_euler = error_Vanyte_Vs_Mocap_fb_R.as_euler("xyz")

figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=error_Vanyte_Vs_Mocap_fb_euler[:,0], mode='lines', name='error_Vanyte_Vs_Mocap_fb_roll'))
figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=error_Vanyte_Vs_Mocap_fb_euler[:,1], mode='lines', name='error_Vanyte_Vs_Mocap_fb_pitch'))
figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=error_Vanyte_Vs_Mocap_fb_euler[:,2], mode='lines', name='error_Vanyte_Vs_Mocap_fb_yaw'))

figOrientations_realigned.update_layout(title="Realigned orientations")

# Show the plotly figures
figOrientations_realigned.show()






###############################  Orientation difference wrt the initial orientation  ###############################


world_mocapFb_Ori_R_transfo = world_mocapFb_Ori_R[0].inv() * world_mocapFb_Ori_R
world_mocapHead_Ori_R_transfo = world_mocapHead_Ori_R[0].inv() * world_mocapHead_Ori_R
world_RigidBody_Ori_R_transfo = world_RigidBody_Ori_R[0].inv() * world_RigidBody_Ori_R
world_VanyteFb_Ori_R_transfo = world_VanyteFb_Ori_R[0].inv() * world_VanyteFb_Ori_R

world_mocapFb_Ori_transfo_euler = world_mocapFb_Ori_R_transfo.as_euler("xyz")
world_mocapHead_Ori_transfo_euler = world_mocapHead_Ori_R_transfo.as_euler("xyz")
world_RigidBody_Ori_transfo_euler = world_RigidBody_Ori_R_transfo.as_euler("xyz")
world_VanyteFb_Ori_transfo_euler = world_VanyteFb_Ori_R_transfo.as_euler("xyz")

world_mocapFb_Ori_transfo_euler_continuous = continuous_euler(world_mocapFb_Ori_transfo_euler)
world_mocapHead_Ori_transfo_euler_continuous = continuous_euler(world_mocapHead_Ori_transfo_euler)
world_RigidBody_Ori_transfo_euler_continuous = continuous_euler(world_RigidBody_Ori_transfo_euler)
world_VanyteFb_Ori_transfo_euler_continuous = continuous_euler(world_VanyteFb_Ori_transfo_euler)


fig3 = go.Figure()

fig3.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapFb_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_mocapFb_Ori_transfo_roll'))
fig3.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapFb_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_mocapFb_Ori_transfo_pitch'))
fig3.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapFb_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_mocapFb_Ori_transfo_yaw'))
fig3.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapHead_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_mocapHead_Ori_transfo_roll'))
fig3.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapHead_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_mocapHead_Ori_transfo_pitch'))
fig3.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapHead_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_mocapHead_Ori_transfo_yaw'))
fig3.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_RigidBody_Ori_transfo_roll'))
fig3.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_RigidBody_Ori_transfo_pitch'))
fig3.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_RigidBody_Ori_transfo_yaw'))
fig3.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteFb_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_VanyteFb_Ori_transfo_roll'))
fig3.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteFb_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_VanyteFb_Ori_transfo_pitch'))
fig3.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteFb_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_VanyteFb_Ori_transfo_yaw'))

fig3.update_layout(title="Orientation transformations")

# Show the plotly figures
fig3.show()




R_zero = R.from_rotvec(np.array([0, 0, 0]))
finalR = R_zero * world_mocapHead_Ori_R_transfo
finalR_euler = finalR.as_euler("xyz")

finalR_euler_continuous = continuous_euler(finalR_euler)

figTest = go.Figure()

figTest.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=finalR_euler_continuous[:,0], mode='lines', name='finalR_roll'))
figTest.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=finalR_euler_continuous[:,1], mode='lines', name='finalR_pitch'))
figTest.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=finalR_euler_continuous[:,2], mode='lines', name='finalR_yaw'))

figTest.update_layout(title="With R identity at start")

# Show the plotly figures
figTest.show()



world_mocapHead_Ori_quat = world_mocapHead_Ori_R.as_quat()
output_df = pd.DataFrame({'t': observer_data['t'], 'realignedWorldHeadPos_x': world_mocapHead_Pos[:,0], 'realignedWorldHeadPos_y': world_mocapHead_Pos[:,1], 'realignedWorldHeadPos_z': world_mocapHead_Pos[:,2], 'realignedWorldHeadOri_x': world_mocapHead_Ori_quat[:,0], 'realignedWorldHeadOri_y': world_mocapHead_Ori_quat[:,1], 'realignedWorldHeadOri_z': world_mocapHead_Ori_quat[:,2], 'realignedWorldHeadOri_w': world_mocapHead_Ori_quat[:,3]})



# Save the DataFrame to a new CSV file
save_csv = input("Do you want to save the data as a CSV file? (y/n): ")

if save_csv.lower() == 'y':
    output_df.to_csv(output_csv_file_path, index=False)
    print("Output CSV file has been saved to ", output_csv_file_path)
else:
    print("Data not saved.")

