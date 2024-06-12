import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

import csv

import plotly.graph_objects as go


from scipy.spatial.transform import Rotation as R

from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


###############################  User input for the timestep  ###############################

timeStepInput = input("Please enter the timestep of the controller in milliseconds: ")

# Convert the input to a double
try:
    timeStepInt = int(timeStepInput)
    timeStepFloat = float(timeStepInput)*1000.0
    resample_str = f'{timeStepInt}ms'
    print(f"Resampling the MoCap data at {timeStepInt} ms")
except ValueError:
    print("That's not a valid int!")



###############################  Main variables initialization  ###############################

csv_file_path = 'ExpeRhps1.csv'
output_csv_file_path = 'resampledMocapData.csv'
# Define a list of patterns you want to match
pattern1 = ['Time(Seconds)','Marker1', 'Marker2', 'Marker3']  # Add more patterns as needed
pattern2 = r'RigidBody(?!.*Marker)'
# Position of the markers in the head frame
head_P1_pos = np.array([-114.6, 1.4, 191.3])
head_P2_pos = np.array([95.2, -49.9, 202.6])
head_P3_pos = np.array([46.6, 71.1, 4.9])



###############################  Function definitions  ###############################

# Filter columns based on the predefined patterns
def filterColumns(dataframe, pattern1, pattern2):
    filtered_columns = []
    for col in dataframe.columns:
        if any(pattern in col for pattern in pattern1):
            filtered_columns.append(col)
        elif re.search(pattern2, col):
            filtered_columns.append(col)
    return filtered_columns

def convert_mm_to_m(dataframe):
    for col in dataframe.columns:
        if ('Pos' in col) or ('RigidBody001_t' in col):
            dataframe[col] = dataframe[col] / 1000
    return dataframe

def rename_columns(dataframe):
    for col in dataframe.columns:
        if 'Marker' in col:
            dataframe = dataframe.rename(columns={col: col.replace('RigidBody001:', '')})
    return dataframe

def time_to_seconds(time_value):
    return time_value.hour * 3600 + time_value.minute * 60 + time_value.second + time_value.microsecond / 1000000.

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



###############################  Data handling and resampling  ###############################

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path, header=[2, 4, 5])
df1 = df.drop(df.columns[[0, 1]], axis=1)
# Combine the multi-level header into a single level header
df1.columns = ['_'.join(map(str, col)).strip() for col in df1.columns.values]
df_time = pd.read_csv(csv_file_path, header=5, usecols=['Time (Seconds)'])
df2 = pd.concat([df_time,df1], axis=1, sort=False)

df2.columns = df2.columns.str.replace('Position_', 't') 
df2.columns = df2.columns.str.replace('Rotation_', 'q') 
df2.columns = df2.columns.str.replace(' ', '') 

filtered_columns = filterColumns(df2, pattern1, pattern2)

# Filter the DataFrame to keep only the specified columns
df_filtered = df2[filtered_columns].copy()
df_filtered = rename_columns(df_filtered)

# Convert the float column to datetime
df_filtered.loc[:, 'Time(Seconds)'] = pd.to_datetime(df_filtered['Time(Seconds)'], unit='s')
df_filtered.set_index('Time(Seconds)', inplace=True, drop=True)

resampled_df = df_filtered.resample(resample_str).interpolate(method='polynomial', order=2, limit_direction='both').bfill().ffill()

resampled_df = resampled_df.reset_index()
#resampled_df = df_filtered.resample('5ms').interpolate(method='linear').bfill()
# Extract the time component from the datetime column using the '.dt.time' attribute
resampled_df.loc[:, 'Time(Seconds)'] = resampled_df['Time(Seconds)'].dt.time
# Apply the 'time_to_seconds' function to the 'Time (Seconds)' column to convert it to seconds including milliseconds
resampled_df.loc[:, 'Time(Seconds)'] = resampled_df['Time(Seconds)'].apply(time_to_seconds).astype('float')


# Original data for the comparison plots
df_filtered = df_filtered.reset_index()
df_filtered.loc[:, 'Time(Seconds)'] = df_filtered['Time(Seconds)'].dt.time
# Apply the 'time_to_seconds' function to the 'Time (Seconds)' column to convert it to seconds including milliseconds
df_filtered.loc[:, 'Time(Seconds)'] = df_filtered['Time(Seconds)'].apply(time_to_seconds).astype('float')


plotOriginalVsResampled = input("Do you want to compare the resampled data to the original one? (y/n): ")

if plotOriginalVsResampled.lower() == 'y':
    # Plot the original and resampled data
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=df_filtered["Time(Seconds)"], y=df_filtered["Marker1_tX"], mode='lines', name='Initial_Marker1_tX'))
    fig1.add_trace(go.Scatter(x=df_filtered["Time(Seconds)"], y=df_filtered["Marker2_tX"], mode='lines', name='Initial_Marker2_tX'))
    fig1.add_trace(go.Scatter(x=df_filtered["Time(Seconds)"], y=df_filtered["Marker3_tX"], mode='lines', name='Initial_Marker3_tX'))
    fig1.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["Marker1_tX"], mode='lines', name='Resampled_Marker1_tX'))
    fig1.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["Marker2_tX"], mode='lines', name='Resampled_Marker2_tX'))
    fig1.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["Marker3_tX"], mode='lines', name='Resampled_Marker3_tX'))

    fig1.add_trace(go.Scatter(x=df_filtered["Time(Seconds)"], y=df_filtered["Marker1_tY"], mode='lines', name='Initial_Marker1_tY'))
    fig1.add_trace(go.Scatter(x=df_filtered["Time(Seconds)"], y=df_filtered["Marker2_tY"], mode='lines', name='Initial_Marker2_tY'))
    fig1.add_trace(go.Scatter(x=df_filtered["Time(Seconds)"], y=df_filtered["Marker3_tY"], mode='lines', name='Initial_Marker3_tY'))
    fig1.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["Marker1_tY"], mode='lines', name='Resampled_Marker1_tY'))
    fig1.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["Marker2_tY"], mode='lines', name='Resampled_Marker2_tY'))
    fig1.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["Marker3_tY"], mode='lines', name='Resampled_Marker3_tY'))

    fig1.add_trace(go.Scatter(x=df_filtered["Time(Seconds)"], y=df_filtered["Marker1_tZ"], mode='lines', name='Initial_Marker1_tZ'))
    fig1.add_trace(go.Scatter(x=df_filtered["Time(Seconds)"], y=df_filtered["Marker2_tZ"], mode='lines', name='Initial_Marker2_tZ'))
    fig1.add_trace(go.Scatter(x=df_filtered["Time(Seconds)"], y=df_filtered["Marker3_tZ"], mode='lines', name='Initial_Marker3_tZ'))
    fig1.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["Marker1_tZ"], mode='lines', name='Resampled_Marker1_tZ'))
    fig1.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["Marker2_tZ"], mode='lines', name='Resampled_Marker2_tZ'))
    fig1.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["Marker3_tZ"], mode='lines', name='Resampled_Marker3_tZ'))

    fig1.update_layout(title="Resampled data vs original one")
    
    # Show the plotly figures
    fig1.show()




###############################  Pose of the head in the world  ###############################


# We retrieve directly the orientation the mocap's rigid body in the world frame
worldRigidBodyOri_R = R.from_quat(resampled_df[["RigidBody001_qX", "RigidBody001_qY", "RigidBody001_qZ", "RigidBody001_qW"]].values)
rRigidBody_mats = worldRigidBodyOri_R.as_dcm()

# Initialization of the DataFrame's columns
resampled_df['worldHeadPos_x'] = np.nan
resampled_df['worldHeadPos_y'] = np.nan
resampled_df['worldHeadPos_z'] = np.nan
resampled_df['worldHeadOri_qx'] = np.nan
resampled_df['worldHeadOri_qy'] = np.nan
resampled_df['worldHeadOri_qz'] = np.nan
resampled_df['worldHeadOri_qw'] = np.nan

resampled_df['world_threePointsFrame_x'] = np.nan
resampled_df['world_threePointsFrame_y'] = np.nan
resampled_df['world_threePointsFrame_z'] = np.nan
resampled_df['world_threePointsFrame_qx'] = np.nan
resampled_df['world_threePointsFrame_qy'] = np.nan
resampled_df['world_threePointsFrame_qz'] = np.nan
resampled_df['world_threePointsFrame_qw'] = np.nan


# Pose of the frame defined by the three points in the head frame.
head_threePointsFrame_pos = (head_P1_pos + head_P2_pos + head_P3_pos) / 3
head_threePointsFrame_x = (head_P2_pos - head_P1_pos)
head_threePointsFrame_x = head_threePointsFrame_x / np.linalg.norm(head_threePointsFrame_x)
head_threePointsFrame_y = np.cross(head_threePointsFrame_x, head_P3_pos - head_P1_pos)
head_threePointsFrame_y = head_threePointsFrame_y / np.linalg.norm(head_threePointsFrame_y)
head_threePointsFrame_z = np.cross(head_threePointsFrame_x, head_threePointsFrame_y)
head_threePointsFrame_z = head_threePointsFrame_z / np.linalg.norm(head_threePointsFrame_z)
head_threePointsFrame_ori = np.column_stack((head_threePointsFrame_x, head_threePointsFrame_y, head_threePointsFrame_z))
threePointsFrame_head_ori = head_threePointsFrame_ori.T
threePointsFrame_head_pos = -np.matmul(threePointsFrame_head_ori, head_threePointsFrame_pos)
threePointsFrame_head_R = R.from_dcm(threePointsFrame_head_ori)

# Update world_P1_pos, world_P2_pos, and world_P3_pos from the current row
world_P1_pos = np.array([resampled_df['Marker1_tX'], resampled_df['Marker1_tY'], resampled_df['Marker1_tZ']]).T
world_P2_pos = np.array([resampled_df['Marker2_tX'], resampled_df['Marker2_tY'], resampled_df['Marker2_tZ']]).T
world_P3_pos = np.array([resampled_df['Marker3_tX'], resampled_df['Marker3_tY'], resampled_df['Marker3_tZ']]).T

# We compute the unit vectors of a frame defined by the three points
worldThreePointsFramePos = (world_P1_pos + world_P2_pos + world_P3_pos) / 3
world_threePointsFrame_x = (world_P2_pos - world_P1_pos)
norms_x = np.linalg.norm(world_threePointsFrame_x, axis=1)
# Reshape norms to enable broadcasting
norms_x = norms_x.reshape(-1, 1)
world_threePointsFrame_x = world_threePointsFrame_x / norms_x
world_threePointsFrame_y = np.cross(world_threePointsFrame_x, world_P3_pos - world_P1_pos)
norms_y = np.linalg.norm(world_threePointsFrame_y, axis=1)
# Reshape norms to enable broadcasting
norms_y = norms_y.reshape(-1, 1)
world_threePointsFrame_y = world_threePointsFrame_y / norms_y
world_threePointsFrame_z = np.cross(world_threePointsFrame_x, world_threePointsFrame_y)
norms_z = np.linalg.norm(world_threePointsFrame_z, axis=1)
# Reshape norms to enable broadcasting
norms_z = norms_z.reshape(-1, 1)
world_threePointsFrame_z = world_threePointsFrame_z / norms_z

# We stack the unit vectors to obtain the rotation matrix
world_threePointsFrame_ori_mat = np.stack((world_threePointsFrame_x, world_threePointsFrame_y, world_threePointsFrame_z), axis=-1)
# The rotation matrix is transformed to a Rotation object
world_threePointsFrame_R = R.from_dcm(world_threePointsFrame_ori_mat)


# We finally get the head's pose in the world by composing the ones of the two frames.
worldHeadOri_R = world_threePointsFrame_R * threePointsFrame_head_R
worldHeadPos = worldThreePointsFramePos + world_threePointsFrame_R.apply(threePointsFrame_head_pos)


# Converting the orientations in the world back to quaternions for the plots and storage
worldHeadOri_quat = worldHeadOri_R.as_quat()
world_threePointsFrame_ori_quat = world_threePointsFrame_R.as_quat()
# if(index > 0 ):
#     if (np.dot(worldHeadOri_quat, previous_head_quaternion) < 0):
#         worldHeadOri_quat = -worldHeadOri_quat
#     if (np.dot(world_threePointsFrame_ori_quat, previous_tpf_quaternion) < 0):
#         world_threePointsFrame_ori_quat = -world_threePointsFrame_ori_quat
# previous_head_quaternion = worldHeadOri_quat
# previous_tpf_quaternion = world_threePointsFrame_ori_quat

resampled_df['worldHeadPos_x'] = worldHeadPos[:,0]
resampled_df['worldHeadPos_y'] = worldHeadPos[:,1]
resampled_df['worldHeadPos_z'] = worldHeadPos[:,2]
resampled_df['worldHeadOri_qx'] = worldHeadOri_quat[:,0]
resampled_df['worldHeadOri_qy'] = worldHeadOri_quat[:,1]
resampled_df['worldHeadOri_qz'] = worldHeadOri_quat[:,2]
resampled_df['worldHeadOri_qw'] = worldHeadOri_quat[:,3]

resampled_df['world_threePointsFrame_x'] = worldThreePointsFramePos[:,0]
resampled_df['world_threePointsFrame_y'] = worldThreePointsFramePos[:,1]
resampled_df['world_threePointsFrame_z'] = worldThreePointsFramePos[:,2]
resampled_df['world_threePointsFrame_qx'] = world_threePointsFrame_ori_quat[:,0]
resampled_df['world_threePointsFrame_qy'] = world_threePointsFrame_ori_quat[:,1]
resampled_df['world_threePointsFrame_qz'] = world_threePointsFrame_ori_quat[:,2]
resampled_df['world_threePointsFrame_qw'] = world_threePointsFrame_ori_quat[:,3]


# Plot of the resulting poses
fig2 = go.Figure()

world_threePointsFrame_ori_euler = world_threePointsFrame_R.as_euler("xyz")
worldHeadOri_euler = worldHeadOri_R.as_euler("xyz")
worldRigidBodyOri_euler = worldRigidBodyOri_R.as_euler("xyz")
# world_threePointsFrame_ori_euler_continuous = continuous_euler(world_threePointsFrame_ori_euler)
# worldHeadOri_euler_continuous = continuous_euler(worldHeadOri_euler)
# worldRigidBodyOri_euler_continuous = continuous_euler(worldRigidBodyOri_euler)

world_threePointsFrame_ori_euler_continuous = world_threePointsFrame_ori_euler
worldHeadOri_euler_continuous = worldHeadOri_euler
worldRigidBodyOri_euler_continuous = worldRigidBodyOri_euler

fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_threePointsFrame_ori_euler_continuous[:,0], mode='lines', name='world_threePointsFrame_ori_roll'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_threePointsFrame_ori_euler_continuous[:,1], mode='lines', name='world_threePointsFrame_ori_pitch'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_threePointsFrame_ori_euler_continuous[:,2], mode='lines', name='world_threePointsFrame_ori_yaw'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=worldHeadOri_euler_continuous[:,0], mode='lines', name='world_Head_ori_roll'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=worldHeadOri_euler_continuous[:,1], mode='lines', name='world_Head_ori_pitch'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=worldHeadOri_euler_continuous[:,2], mode='lines', name='world_Head_ori_yaw'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=worldRigidBodyOri_euler_continuous[:,0], mode='lines', name='world_RigidBody_ori_roll'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=worldRigidBodyOri_euler_continuous[:,1], mode='lines', name='world_RigidBody_ori_pitch'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=worldRigidBodyOri_euler_continuous[:,2], mode='lines', name='world_RigidBody_ori_yaw'))

fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["world_threePointsFrame_x"], mode='lines', name='world_threePointsFrame_pos_x'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["world_threePointsFrame_y"], mode='lines', name='world_threePointsFrame_pos_y'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["world_threePointsFrame_z"], mode='lines', name='world_threePointsFrame_pos_z'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["worldHeadPos_x"], mode='lines', name='world_Head_pos_x'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["worldHeadPos_y"], mode='lines', name='world_Head_pos_y'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["worldHeadPos_z"], mode='lines', name='world_Head_pos_z'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["RigidBody001_tX"], mode='lines', name='world_RigidBody_pos_x'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["RigidBody001_tY"], mode='lines', name='world_RigidBody_pos_y'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["RigidBody001_tZ"], mode='lines', name='world_RigidBody_pos_z'))

errorWithRigidBody_R = worldHeadOri_R.inv() * worldRigidBodyOri_R
errorWithRigidBody_euler = errorWithRigidBody_R.as_euler("xyz")

fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=errorWithRigidBody_euler[:,0], mode='lines', name='errorWithRigidBody_roll'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=errorWithRigidBody_euler[:,1], mode='lines', name='errorWithRigidBody_pitch'))
fig2.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=errorWithRigidBody_euler[:,2], mode='lines', name='errorWithRigidBody_yaw'))

fig2.update_layout(title="Resulting poses in the world")

# Show the plotly figures
fig2.show()





###############################  Local linear velocity of the head in the world  ###############################

# We compute the velocity of the head in the world
worldHeadVel_x = resampled_df['worldHeadPos_x'].diff()/timeStepFloat*1000.0
worldHeadVel_y = resampled_df['worldHeadPos_y'].diff()/timeStepFloat*1000.0
worldHeadVel_z = resampled_df['worldHeadPos_z'].diff()/timeStepFloat*1000.0
worldHeadVel_x[0] = 0.0
worldHeadVel_y[0] = 0.0
worldHeadVel_z[0] = 0.0
worldHeadVel = np.stack((worldHeadVel_x, worldHeadVel_y, worldHeadVel_z), axis = 1)
# We compute the velocity of the mocap's rigid body in the world
worldRigidBodyVel_x = resampled_df['RigidBody001_tX'].diff()/timeStepFloat*1000.0
worldRigidBodyVel_y = resampled_df['RigidBody001_tY'].diff()/timeStepFloat*1000.0
worldRigidBodyVel_z = resampled_df['RigidBody001_tZ'].diff()/timeStepFloat*1000.0
worldRigidBodyVel_x[0] = 0.0
worldRigidBodyVel_y[0] = 0.0
worldRigidBodyVel_z[0] = 0.0

worldRigidBodyVel = np.stack((worldRigidBodyVel_x, worldRigidBodyVel_y, worldRigidBodyVel_z), axis = 1)

worldHeadLocVel = worldHeadOri_R.apply(worldHeadVel, inverse=True)
worldRigidBodyLocVel = worldRigidBodyOri_R.apply(worldRigidBodyVel, inverse=True)

# Plot of the resulting poses
fig3 = go.Figure()

fig3.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=worldHeadLocVel[:,0], mode='lines', name='world_Head_LocalLinVel_x'))
fig3.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=worldHeadLocVel[:,1], mode='lines', name='world_Head_LocalLinVel_y'))
fig3.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=worldHeadLocVel[:,2], mode='lines', name='world_Head_LocalLinVel_z'))
fig3.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=worldRigidBodyLocVel[:,0], mode='lines', name='world_RigidBody_LocalLinVel_x'))
fig3.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=worldRigidBodyLocVel[:,1], mode='lines', name='world_RigidBody_LocalLinVel_y'))
fig3.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=worldRigidBodyLocVel[:,2], mode='lines', name='world_RigidBody_LocalLinVel_z'))

fig3.update_layout(title="Local linear velocity of the head in the world / vs the one of the rigid body")
# Show the plotly figures
fig3.show()






###############################  3d trajectory  ###############################


plot3dTrajectory = input("Do you want to plot the resulting 3d trajectory? (y/n): ")

if plot3dTrajectory.lower() == 'y':


    # Create a 3D plot
    fig, ax = plt.subplots(1, 1)
    ax = fig.add_subplot(111, projection='3d')

    worldHeadOri_mat = worldHeadOri_R.as_dcm()

    x_min = min(resampled_df["worldHeadPos_x"].min(), resampled_df["RigidBody001_tX"].min())
    y_min = min(resampled_df["worldHeadPos_y"].min(), resampled_df["RigidBody001_tY"].min())
    z_min = min(resampled_df["worldHeadPos_z"].min(), resampled_df["RigidBody001_tZ"].min())
    x_min = x_min - np.abs(x_min*0.2)
    y_min = y_min - np.abs(y_min*0.2)
    z_min = z_min - np.abs(z_min*0.2)

    x_max = max(resampled_df["worldHeadPos_x"].max(), resampled_df["RigidBody001_tX"].max())
    y_max = max(resampled_df["worldHeadPos_y"].max(), resampled_df["RigidBody001_tY"].max())
    z_max = max(resampled_df["worldHeadPos_z"].max(), resampled_df["RigidBody001_tZ"].max())
    x_max = x_max + np.abs(x_max*0.2)
    y_max = y_max + np.abs(y_max*0.2)
    z_max = z_max + np.abs(z_max*0.2)

    for t in range(0,len(resampled_df), 1000):
        quiverRigidBody = ax.quiver(resampled_df["RigidBody001_tX"][t], resampled_df["RigidBody001_tY"][t], resampled_df["RigidBody001_tZ"][t], *rRigidBody_mats[t,:,0], color='red', linewidth=4, length=70)
        quiverRigidBody = ax.quiver(resampled_df["RigidBody001_tX"][t], resampled_df["RigidBody001_tY"][t], resampled_df["RigidBody001_tZ"][t], *rRigidBody_mats[t,:,1], color='red', linewidth=1, length=50)
        quiverRigidBody = ax.quiver(resampled_df["RigidBody001_tX"][t], resampled_df["RigidBody001_tY"][t], resampled_df["RigidBody001_tZ"][t], *rRigidBody_mats[t,:,2], color='red', linewidth=3, length=50)
        
        quiverRigidBody = ax.quiver(resampled_df["world_threePointsFrame_x"][t], resampled_df["world_threePointsFrame_y"][t], resampled_df["world_threePointsFrame_z"][t], *world_threePointsFrame_ori_mat[t][:,0], color='orange', linewidth=3, length=70)
        quiverRigidBody = ax.quiver(resampled_df["world_threePointsFrame_x"][t], resampled_df["world_threePointsFrame_y"][t], resampled_df["world_threePointsFrame_z"][t], *world_threePointsFrame_ori_mat[t][:,1], color='orange', linewidth=1, length=50)
        quiverRigidBody = ax.quiver(resampled_df["world_threePointsFrame_x"][t], resampled_df["world_threePointsFrame_y"][t], resampled_df["world_threePointsFrame_z"][t], *world_threePointsFrame_ori_mat[t][:,2], color='orange', linewidth=1, length=50)

        quiverHead = ax.quiver(resampled_df["worldHeadPos_x"][t], resampled_df["worldHeadPos_y"][t], resampled_df["worldHeadPos_z"][t], *worldHeadOri_mat[t][:,0], color='blue', linewidth=4, length=70)
        quiverHead = ax.quiver(resampled_df["worldHeadPos_x"][t], resampled_df["worldHeadPos_y"][t], resampled_df["worldHeadPos_z"][t], *worldHeadOri_mat[t][:,1], color='blue', linewidth=1, length=50)
        quiverHead = ax.quiver(resampled_df["worldHeadPos_x"][t], resampled_df["worldHeadPos_y"][t], resampled_df["worldHeadPos_z"][t], *worldHeadOri_mat[t][:,2], color='blue', linewidth=3, length=50)

        quiverHead = ax.quiver(resampled_df["worldHeadPos_x"][t], resampled_df["worldHeadPos_y"][t], resampled_df["worldHeadPos_z"][t], *head_threePointsFrame_ori[:,0], color='green', linewidth=8, length=70)
        quiverHead = ax.quiver(resampled_df["worldHeadPos_x"][t], resampled_df["worldHeadPos_y"][t], resampled_df["worldHeadPos_z"][t], *head_threePointsFrame_ori[:,1], color='green', linewidth=1, length=50)
        quiverHead = ax.quiver(resampled_df["worldHeadPos_x"][t], resampled_df["worldHeadPos_y"][t], resampled_df["worldHeadPos_z"][t], *head_threePointsFrame_ori[:,2], color='green', linewidth=3, length=50)

    # Adding the axes arrows
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)
    arrow_x = Arrow3D([x_min,x_max],[y_min,y_min],[z_min,z_min], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    arrow_y = Arrow3D([x_min,x_min],[y_min,y_max],[z_min,z_min], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    arrow_z = Arrow3D([x_min,x_min],[y_min,y_min],[z_min,z_max], mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
    ax.add_artist(arrow_x)
    ax.add_artist(arrow_y)
    ax.add_artist(arrow_z)

    ax.plot(resampled_df["RigidBody001_tX"], resampled_df["RigidBody001_tY"], resampled_df["RigidBody001_tZ"], color='darkred')
    ax.plot(resampled_df["world_threePointsFrame_x"], resampled_df["world_threePointsFrame_y"], resampled_df["world_threePointsFrame_z"], color='darkorange')
    ax.plot(resampled_df["worldHeadPos_x"], resampled_df["worldHeadPos_y"], resampled_df["worldHeadPos_z"], color='darkblue')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])


    # Show the plot
    plt.show()


# Remove the useless columns

# Specify your patterns
patterns = ['Marker', 'threePoints']
# Get columns that contain any of the patterns
cols_to_drop = resampled_df.columns[resampled_df.columns.str.contains('|'.join(patterns))]
# Drop these columns
resampled_df = resampled_df.drop(columns=cols_to_drop)
resampled_df = convert_mm_to_m(resampled_df)


# Save the DataFrame to a new CSV file
save_csv = input("Do you want to save the data as a CSV file? (y/n): ")

if save_csv.lower() == 'y':
    resampled_df.to_csv(output_csv_file_path, index=False)
    print("Output CSV file has been saved to ", output_csv_file_path)
else:
    print("Data not saved.")

