import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objects as go


from scipy.spatial.transform import Rotation as R

from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

import yaml

import sys



###############################  Main variables initialization  ###############################

path_to_project = ".."
displayLogs = True
# Define a list of patterns you want to match
pattern1 = ['Time(Seconds)','Marker1', 'Marker2', 'Marker3']  # Add more patterns as needed
pattern2 = r'RigidBody(?!.*Marker)'
scriptName = "Resample and Extract MoCap data"


###############################  User inputs  ###############################


if(len(sys.argv) > 1):
    displayLogs = sys.argv[1].lower() == 'true'
    if(len(sys.argv) > 3):
        path_to_project = sys.argv[3]

with open(f'{path_to_project}/output_data/observers_infos.yaml', 'r') as file:
    try:
        infos_yaml_str = file.read()
        infos_yamlData = yaml.safe_load(infos_yaml_str)
        timeStep_s = float(infos_yamlData.get("timeStep_s"))
        timeStep_ms = int(timeStep_s*1000.0)
        resample_str = f'{timeStep_ms}ms'
    except yaml.YAMLError as exc:
        print(exc)

print(f"Preparing to resample the MoCap data at {timeStep_ms} ms")

csv_file_path = f'{path_to_project}/raw_data/mocapData.csv'
output_csv_file_path_mocap = f'{path_to_project}/output_data/resampledMocapData.csv'



###############################  Configuration reading  ###############################

def get_markers(robot_name, body_name):
    # Iterate over the robots
    for robot in markers_yamlData['robots']:
        # If the robot name matches
        if robot['name'] == robot_name:
            # Iterate over the bodies of the robot
            for body in robot['bodies']:
                # If the body name matches
                if body['name'] == body_name:
                    # Return the markers
                    markers = body['markers']
                    marker1 = markers[0]['Marker1']
                    marker2 = markers[1]['Marker2']
                    marker3 = markers[2]['Marker3']
                    marker1_pos = np.array([marker1['x'], marker1['y'], marker1['z']]) / 1000.0
                    marker2_pos = np.array([marker2['x'], marker2['y'], marker2['z']]) / 1000.0
                    marker3_pos = np.array([marker3['x'], marker3['y'], marker3['z']]) / 1000.0
                    return marker1_pos, marker2_pos, marker3_pos

    # If no matching robot and body were found, return None
    print("The pair robot / body given in the file markersPlacements.yaml doesn't have marker positions. Please give a valid pair.")
    sys.exit()

with open('../markersPlacements.yaml', 'r') as file:
    try:
        markers_yaml_str = file.read()
        markers_yamlData = yaml.safe_load(markers_yaml_str)
    except yaml.YAMLError as exc:
        print(exc)

with open(f'{path_to_project}/projectConfig.yaml', 'r') as file:
    try:
        projConf_yaml_str = file.read()
        projConf_yamlData = yaml.safe_load(projConf_yaml_str)
    except yaml.YAMLError as exc:
        print(exc)

# Get the value of EnabledRobot and EnabledBody
enabled_robot = projConf_yamlData.get('EnabledRobot')
enabled_body = projConf_yamlData.get('EnabledBody')

# Check if EnabledRobot exists and is uncommented
if enabled_robot is None:
    print("EnabledRobot does not exist or is commented out.")
    enabled_robot = input("Please enter the name of the robot: ")

# Check if EnabledBody exists and is uncommented
if enabled_body is None:
    print("EnabledBody does not exist or is commented out.")
    enabled_body = input("Please enter the name of the limb: ")
    

# Get the markers and print them
marker1_pos, marker2_pos, marker3_pos = get_markers(enabled_robot, enabled_body)



###############################  Function definitions  ###############################

# # Filter columns based on the predefined patterns
# def filterColumns(dataframe, pattern1, pattern2):
#     filtered_columns = []
#     for col in dataframe.columns:
#         if any(pattern in col for pattern in pattern1):
#             filtered_columns.append(col)
#         elif re.search(pattern2, col):
#             filtered_columns.append(col)
#     return filtered_columns

def convert_mm_to_m(dataframe):
    position_columns = [col for col in dataframe.columns if 'Position' in col]
    for col in position_columns:
        dataframe[col] = dataframe[col] / 1000.0
    return dataframe

def rename_columns(dataframe):
    for col in dataframe.columns:
        if 'Marker' in col:
            dataframe = dataframe.rename(columns={col: col.replace('RigidBody001:', '')})
        if 'Time(Seconds)' in col:
            dataframe = dataframe.rename(columns={col: col.replace('Name__', '')})
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

# Read header rows only
with open(csv_file_path, 'r') as f:
    lines = [next(f) for _ in range(7)]

# Split header lines
line4 = lines[3].strip().split(',')
line6 = lines[5].strip().split(',')
line7 = lines[6].strip().split(',')

final_headers = [
    f"{h4.strip()}_{h6.strip()}_{h7.strip()}"
    for h4, h6, h7 in zip(line4, line6, line7)
]

# Filter columns
usecols = [i for i, (h4, h7) in enumerate(zip(line4, line7))
           if 'Marker1' in h4 or 'Marker2' in h4 or 'Marker3' in h4 or h4 == 'Rigid Body 001' or 'Time (Seconds)' in h7]

# Read only desired columns (use line 7 for actual column parsing)
df = pd.read_csv(csv_file_path, header=5, usecols=usecols)

# Rename columns to combined headers
selected_headers = [final_headers[i] for i in usecols]
df.columns = selected_headers

# df = df.drop(df.columns[[0, 1]], axis=1)

# Combine the multi-level header into a single level header
# df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]


df = convert_mm_to_m(df)


df.columns = df.columns.str.replace('Position_', 't') 
df.columns = df.columns.str.replace('Rotation_', 'q') 
df.columns = df.columns.str.replace(' ', '') 

# filtered_columns = filterColumns(df2, pattern1, pattern2)

# Filter the DataFrame to keep only the specified columns
# df_filtered = df2[filtered_columns].copy()


df_filtered = rename_columns(df)

df_filtered = df_filtered.loc[:,~df.columns.duplicated()].copy()
#Removing potential duplicates

# Convert the float column to datetime
df_filtered.loc[:, 'Time(Seconds)'] = pd.to_datetime(df_filtered['Time(Seconds)'], unit='s')
df_filtered.set_index('Time(Seconds)', inplace=True, drop=True)

print(f"Resampling the MoCap data at {timeStep_ms} ms")
resampled_df = df_filtered.resample(resample_str).interpolate(method='polynomial', order=2, limit_direction='both').bfill().ffill()
print(f"Finished the resampling.")
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

plotOriginalVsResampled = 'n'
if(displayLogs):
    plotOriginalVsResampled = input("Do you want to compare the resampled data to the original one? (y/n): ")


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

fig1.update_layout(title=f"{scriptName}: Resampled data vs original one")



# fig1.write_image(f'{path_to_project}/output_data/scriptResults/resampleAndExtractMocap/resampledMocapMarkers.png')

# if plotOriginalVsResampled.lower() == 'y':
#     # Show the plotly figures
#     fig1.show()




###############################  Pose of the mocapLimb in the world  ###############################


# We retrieve directly the orientation the mocap's rigid body in the world frame
world_RigidBody_Pos = np.array([resampled_df['RigidBody001_tX'], resampled_df['RigidBody001_tY'], resampled_df['RigidBody001_tZ']]).T

world_RigidBody_Ori_R = R.from_quat(resampled_df[["RigidBody001_qX", "RigidBody001_qY", "RigidBody001_qZ", "RigidBody001_qW"]].values)
world_RigidBody_Ori_mat = world_RigidBody_Ori_R.as_matrix()


# Pose of the frame defined by the three points in the mocapLimb frame.
mocapLimb_ThreePointsFrame_Pos = (marker1_pos + marker2_pos + marker3_pos) / 3
mocapLimb_ThreePointsFrame_x = (marker2_pos - marker1_pos)
mocapLimb_ThreePointsFrame_x = mocapLimb_ThreePointsFrame_x / np.linalg.norm(mocapLimb_ThreePointsFrame_x)
mocapLimb_ThreePointsFrame_y = np.cross(mocapLimb_ThreePointsFrame_x, marker3_pos - marker1_pos)
mocapLimb_ThreePointsFrame_y = mocapLimb_ThreePointsFrame_y / np.linalg.norm(mocapLimb_ThreePointsFrame_y)
mocapLimb_ThreePointsFrame_z = np.cross(mocapLimb_ThreePointsFrame_x, mocapLimb_ThreePointsFrame_y)
mocapLimb_ThreePointsFrame_z = mocapLimb_ThreePointsFrame_z / np.linalg.norm(mocapLimb_ThreePointsFrame_z)
mocapLimb_ThreePointsFrame_Ori = np.column_stack((mocapLimb_ThreePointsFrame_x, mocapLimb_ThreePointsFrame_y, mocapLimb_ThreePointsFrame_z))
threePointsFrame_MocapLimb_Ori = mocapLimb_ThreePointsFrame_Ori.T
threePointsFrame_MocapLimb_Pos = -np.matmul(threePointsFrame_MocapLimb_Ori, mocapLimb_ThreePointsFrame_Pos)
threePointsFrame_MocapLimb_Ori_R = R.from_matrix(threePointsFrame_MocapLimb_Ori)


# We get the position of the markers in the world
world_P1_Pos = np.array([resampled_df['Marker1_tX'], resampled_df['Marker1_tY'], resampled_df['Marker1_tZ']]).T
world_P2_Pos = np.array([resampled_df['Marker2_tX'], resampled_df['Marker2_tY'], resampled_df['Marker2_tZ']]).T
world_P3_Pos = np.array([resampled_df['Marker3_tX'], resampled_df['Marker3_tY'], resampled_df['Marker3_tZ']]).T

rigidBody_World_Ori = world_RigidBody_Ori_R.inv()
rigidBody_World_Pos = - rigidBody_World_Ori.apply(world_RigidBody_Pos)

rigidBody_Marker1_Pos = np.mean(rigidBody_World_Pos + rigidBody_World_Ori.apply(world_P1_Pos), axis=0)
rigidBody_Marker2_Pos = np.mean(rigidBody_World_Pos + rigidBody_World_Ori.apply(world_P2_Pos), axis=0)
rigidBody_Marker3_Pos = np.mean(rigidBody_World_Pos + rigidBody_World_Ori.apply(world_P3_Pos), axis=0)

rigidBody_ThreePointsFrame_Pos = (rigidBody_Marker1_Pos + rigidBody_Marker2_Pos + rigidBody_Marker3_Pos) / 3
rigidBody_ThreePointsFrame_x = rigidBody_Marker2_Pos - rigidBody_Marker1_Pos
rigidBody_ThreePointsFrame_x = rigidBody_ThreePointsFrame_x / np.linalg.norm(rigidBody_ThreePointsFrame_x)

rigidBody_ThreePointsFrame_y = np.cross(rigidBody_ThreePointsFrame_x, rigidBody_Marker3_Pos - rigidBody_Marker1_Pos)
rigidBody_ThreePointsFrame_y = rigidBody_ThreePointsFrame_y / np.linalg.norm(rigidBody_ThreePointsFrame_y)
rigidBody_ThreePointsFrame_z = np.cross(rigidBody_ThreePointsFrame_x, rigidBody_ThreePointsFrame_y)
rigidBody_ThreePointsFrame_z = rigidBody_ThreePointsFrame_z / np.linalg.norm(rigidBody_ThreePointsFrame_z)
rigidBody_ThreePointsFrame_Ori_mat = np.column_stack((rigidBody_ThreePointsFrame_x, rigidBody_ThreePointsFrame_y, rigidBody_ThreePointsFrame_z))
rigidBody_ThreePointsFrame_Ori_R = R.from_matrix(rigidBody_ThreePointsFrame_Ori_mat)

rigidBody_MocapLimb_Ori_R = rigidBody_ThreePointsFrame_Ori_R * threePointsFrame_MocapLimb_Ori_R
rigidBody_MocapLimb_Pos = rigidBody_ThreePointsFrame_Pos + rigidBody_ThreePointsFrame_Ori_R.apply(threePointsFrame_MocapLimb_Pos)

world_MocapLimb_Ori_R = world_RigidBody_Ori_R * rigidBody_MocapLimb_Ori_R
world_MocapLimb_Pos = world_RigidBody_Pos + rigidBody_MocapLimb_Ori_R.apply(rigidBody_MocapLimb_Pos)



###############################  Storage of the pose of the mocapLimb in the world  ###############################

# Converting the orientations in the world back to quaternions for the plots and storage
world_ThreePointsFrame_Ori_R = world_RigidBody_Ori_R * rigidBody_ThreePointsFrame_Ori_R
world_ThreePointsFrame_Pos = world_RigidBody_Pos + world_RigidBody_Ori_R.apply(rigidBody_ThreePointsFrame_Pos)

world_MocapLimb_Ori_quat = world_MocapLimb_Ori_R.as_quat()
world_ThreePointsFrame_Ori_quat = world_ThreePointsFrame_Ori_R.as_quat()

# Initialization of the DataFrame's columns
resampled_df['world_MocapLimb_Pos_x'] = np.nan
resampled_df['world_MocapLimb_Pos_y'] = np.nan
resampled_df['world_MocapLimb_Pos_z'] = np.nan
resampled_df['world_MocapLimb_Ori_qx'] = np.nan
resampled_df['world_MocapLimb_Ori_qy'] = np.nan
resampled_df['world_MocapLimb_Ori_qz'] = np.nan
resampled_df['world_MocapLimb_Ori_qw'] = np.nan
resampled_df['world_ThreePointsFrame_x'] = np.nan
resampled_df['world_ThreePointsFrame_y'] = np.nan
resampled_df['world_ThreePointsFrame_z'] = np.nan
resampled_df['world_ThreePointsFrame_qx'] = np.nan
resampled_df['world_ThreePointsFrame_qy'] = np.nan
resampled_df['world_ThreePointsFrame_qz'] = np.nan
resampled_df['world_ThreePointsFrame_qw'] = np.nan


resampled_df['world_MocapLimb_Pos_x'] = world_MocapLimb_Pos[:,0]
resampled_df['world_MocapLimb_Pos_y'] = world_MocapLimb_Pos[:,1]
resampled_df['world_MocapLimb_Pos_z'] = world_MocapLimb_Pos[:,2]
resampled_df['world_MocapLimb_Ori_qx'] = world_MocapLimb_Ori_quat[:,0]
resampled_df['world_MocapLimb_Ori_qy'] = world_MocapLimb_Ori_quat[:,1]
resampled_df['world_MocapLimb_Ori_qz'] = world_MocapLimb_Ori_quat[:,2]
resampled_df['world_MocapLimb_Ori_qw'] = world_MocapLimb_Ori_quat[:,3]
resampled_df['world_ThreePointsFrame_x'] = world_ThreePointsFrame_Pos[:,0]
resampled_df['world_ThreePointsFrame_y'] = world_ThreePointsFrame_Pos[:,1]
resampled_df['world_ThreePointsFrame_z'] = world_ThreePointsFrame_Pos[:,2]
resampled_df['world_ThreePointsFrame_qx'] = world_ThreePointsFrame_Ori_quat[:,0]
resampled_df['world_ThreePointsFrame_qy'] = world_ThreePointsFrame_Ori_quat[:,1]
resampled_df['world_ThreePointsFrame_qz'] = world_ThreePointsFrame_Ori_quat[:,2]
resampled_df['world_ThreePointsFrame_qw'] = world_ThreePointsFrame_Ori_quat[:,3]


print("Finished resampling.")

###############################  Plot of the resulting positions  ###############################



# # Plot of the resulting poses
# figPositions = go.Figure()

# rigidBody_ThreePointsFrame_Pos_plot = np.full((len(resampled_df), 3), rigidBody_ThreePointsFrame_Pos)

# figPositions.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["RigidBody001_tX"], mode='lines', name='world_RigidBody_Pos_x'))
# figPositions.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["RigidBody001_tY"], mode='lines', name='world_RigidBody_Pos_y'))
# figPositions.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["RigidBody001_tZ"], mode='lines', name='world_RigidBody_Pos_z'))

# figPositions.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=rigidBody_ThreePointsFrame_Pos_plot[:,0], mode='lines', name='rigidBody_ThreePointsFrame_Pos_x'))
# figPositions.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=rigidBody_ThreePointsFrame_Pos_plot[:,1], mode='lines', name='rigidBody_ThreePointsFrame_Pos_y'))
# figPositions.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=rigidBody_ThreePointsFrame_Pos_plot[:,2], mode='lines', name='rigidBody_ThreePointsFrame_Pos_z'))

# figPositions.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["world_ThreePointsFrame_x"], mode='lines', name='world_ThreePointsFrame_Pos_x'))
# figPositions.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["world_ThreePointsFrame_y"], mode='lines', name='world_ThreePointsFrame_Pos_y'))
# figPositions.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["world_ThreePointsFrame_z"], mode='lines', name='world_ThreePointsFrame_Pos_z'))

# figPositions.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["world_MocapLimb_Pos_x"], mode='lines', name='world_MocapLimb_Pos_x'))
# figPositions.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["world_MocapLimb_Pos_y"], mode='lines', name='world_MocapLimb_Pos_y'))
# figPositions.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=resampled_df["world_MocapLimb_Pos_z"], mode='lines', name='world_MocapLimb_Pos_z'))


# figPositions.update_layout(title=f"{scriptName}: Resulting positions in the world")

# figPositions.write_image(f'{path_to_project}/output_data/scriptResults/resampleAndExtractMocap/resultingLimbPos.png')

# if(displayLogs):
#     # Show the plotly figures
#     figPositions.show()






###############################  Plot of the resulting orientations  ###############################


# # Plot of the resulting poses
# figOrientations = go.Figure()

# world_RigidBody_Ori_euler = world_RigidBody_Ori_R.as_euler("xyz")
# rigidBody_ThreePointsFrame_Ori_euler = rigidBody_ThreePointsFrame_Ori_R.as_euler("xyz")
# threePointsFrame_MocapLimb_Ori_euler = threePointsFrame_MocapLimb_Ori_R.as_euler("xyz")
# world_ThreePointsFrame_Ori_euler = world_ThreePointsFrame_Ori_R.as_euler("xyz")
# world_MocapLimb_Ori_euler = world_MocapLimb_Ori_R.as_euler("xyz")

# rigidBody_MocapLimb_Ori_R = rigidBody_ThreePointsFrame_Ori_R * threePointsFrame_MocapLimb_Ori_R
# rigidBody_MocapLimb_Ori_euler = rigidBody_MocapLimb_Ori_R.as_euler("xyz")
# rigidBody_MocapLimb_Ori_euler_plot = np.full((len(world_RigidBody_Ori_euler), 3), rigidBody_MocapLimb_Ori_euler)
# threePointsFrame_MocapLimb_Ori_euler = np.full((len(world_RigidBody_Ori_euler), 3), threePointsFrame_MocapLimb_Ori_euler)

# world_RigidBody_Ori_euler_continuous = world_RigidBody_Ori_euler
# rigidBody_ThreePointsFrame_Ori_euler = np.full((len(world_RigidBody_Ori_euler_continuous), 3), rigidBody_ThreePointsFrame_Ori_euler)
# world_ThreePointsFrame_Ori_euler_continuous = world_ThreePointsFrame_Ori_euler
# world_MocapLimb_Ori_euler_continuous = world_MocapLimb_Ori_euler

# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,0], mode='lines', name='world_RigidBody_Ori_roll'))
# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,1], mode='lines', name='world_RigidBody_Ori_pitch'))
# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,2], mode='lines', name='world_RigidBody_Ori_yaw'))

# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=rigidBody_ThreePointsFrame_Ori_euler[:,0], mode='lines', name='rigidBody_ThreePointsFrame_Ori_roll'))
# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=rigidBody_ThreePointsFrame_Ori_euler[:,1], mode='lines', name='rigidBody_ThreePointsFrame_Ori_pitch'))
# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=rigidBody_ThreePointsFrame_Ori_euler[:,2], mode='lines', name='rigidBody_ThreePointsFrame_Ori_yaw'))

# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=threePointsFrame_MocapLimb_Ori_euler[:,0], mode='lines', name='threePointsFrame_MocapLimb_Ori_roll'))
# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=threePointsFrame_MocapLimb_Ori_euler[:,1], mode='lines', name='threePointsFrame_MocapLimb_Ori_pitch'))
# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=threePointsFrame_MocapLimb_Ori_euler[:,2], mode='lines', name='threePointsFrame_MocapLimb_Ori_yaw'))

# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=rigidBody_MocapLimb_Ori_euler_plot[:,0], mode='lines', name='rigidBody_MocapLimb_Ori_roll'))
# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=rigidBody_MocapLimb_Ori_euler_plot[:,1], mode='lines', name='rigidBody_MocapLimb_Ori_pitch'))
# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=rigidBody_MocapLimb_Ori_euler_plot[:,2], mode='lines', name='rigidBody_MocapLimb_Ori_yaw'))

# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_ThreePointsFrame_Ori_euler_continuous[:,0], mode='lines', name='world_ThreePointsFrame_Ori_roll'))
# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_ThreePointsFrame_Ori_euler_continuous[:,1], mode='lines', name='world_ThreePointsFrame_Ori_pitch'))
# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_ThreePointsFrame_Ori_euler_continuous[:,2], mode='lines', name='world_ThreePointsFrame_Ori_yaw'))

# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_MocapLimb_Ori_euler_continuous[:,0], mode='lines', name='world_MocapLimb_Ori_roll'))
# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_MocapLimb_Ori_euler_continuous[:,1], mode='lines', name='world_MocapLimb_Ori_pitch'))
# figOrientations.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_MocapLimb_Ori_euler_continuous[:,2], mode='lines', name='world_MocapLimb_Ori_yaw'))

# figOrientations.update_layout(title=f"{scriptName}: Resulting orientations in the world")


# figOrientations.write_image(f'{path_to_project}/output_data/scriptResults/resampleAndExtractMocap/resultingLimbOri.png')

# if(displayLogs):
#     # Show the plotly figures
#     figOrientations.show()


###############################  Local linear velocity of the mocapLimb in the world  ###############################

if(displayLogs):
    # We compute the velocity of the mocapLimb in the world
    world_MocapLimb_Vel_x = resampled_df['world_MocapLimb_Pos_x'].diff()/timeStep_s
    world_MocapLimb_Vel_y = resampled_df['world_MocapLimb_Pos_y'].diff()/timeStep_s
    world_MocapLimb_Vel_z = resampled_df['world_MocapLimb_Pos_z'].diff()/timeStep_s
    world_MocapLimb_Vel_x[0] = 0.0
    world_MocapLimb_Vel_y[0] = 0.0
    world_MocapLimb_Vel_z[0] = 0.0
    world_MocapLimb_Vel = np.stack((world_MocapLimb_Vel_x, world_MocapLimb_Vel_y, world_MocapLimb_Vel_z), axis = 1)
    # We compute the velocity of the mocap's rigid body in the world
    world_RigidBody_Vel_x = resampled_df['RigidBody001_tX'].diff()/timeStep_s
    world_RigidBody_Vel_y = resampled_df['RigidBody001_tY'].diff()/timeStep_s
    world_RigidBody_Vel_z = resampled_df['RigidBody001_tZ'].diff()/timeStep_s
    world_RigidBody_Vel_x[0] = 0.0
    world_RigidBody_Vel_y[0] = 0.0
    world_RigidBody_Vel_z[0] = 0.0

    world_RigidBody_Vel = np.stack((world_RigidBody_Vel_x, world_RigidBody_Vel_y, world_RigidBody_Vel_z), axis = 1)

    world_MocapLimb_LocVel = world_MocapLimb_Ori_R.apply(world_MocapLimb_Vel, inverse=True)
    world_RigidBody_LocVel = world_RigidBody_Ori_R.apply(world_RigidBody_Vel, inverse=True)

    # Plot of the resulting poses
    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_MocapLimb_LocVel[:,0], mode='lines', name='world_MocapLimb_LocalLinVel_x'))
    fig3.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_MocapLimb_LocVel[:,1], mode='lines', name='world_MocapLimb_LocalLinVel_y'))
    fig3.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_MocapLimb_LocVel[:,2], mode='lines', name='world_MocapLimb_LocalLinVel_z'))
    fig3.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_RigidBody_LocVel[:,0], mode='lines', name='world_RigidBody_LocalLinVel_x'))
    fig3.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_RigidBody_LocVel[:,1], mode='lines', name='world_RigidBody_LocalLinVel_y'))
    fig3.add_trace(go.Scatter(x=resampled_df["Time(Seconds)"], y=world_RigidBody_LocVel[:,2], mode='lines', name='world_RigidBody_LocalLinVel_z'))

    fig3.update_layout(title=f"{scriptName}: Local linear velocity")

    # Show the plotly figures
    fig3.show()





###############################  3d trajectory  ###############################


plot3dTrajectory = 'n'
if(displayLogs):
    plot3dTrajectory = input("Do you want to plot the resulting 3d trajectory? (y/n): ")

if plot3dTrajectory.lower() == 'y':


    # Create a 3D plot
    fig, ax = plt.subplots(1, 1)
    ax = fig.add_subplot(111, projection='3d')

    world_MocapLimb_Ori_mat = world_MocapLimb_Ori_R.as_matrix()
    world_ThreePointsFrame_Ori_mat = world_ThreePointsFrame_Ori_R.as_matrix()

    x_min = min(resampled_df["world_MocapLimb_Pos_x"].min(), resampled_df["RigidBody001_tX"].min())
    y_min = min(resampled_df["world_MocapLimb_Pos_y"].min(), resampled_df["RigidBody001_tY"].min())
    z_min = min(resampled_df["world_MocapLimb_Pos_z"].min(), resampled_df["RigidBody001_tZ"].min())
    x_min = x_min - np.abs(x_min*0.2)
    y_min = y_min - np.abs(y_min*0.2)
    z_min = z_min - np.abs(z_min*0.2)

    x_max = max(resampled_df["world_MocapLimb_Pos_x"].max(), resampled_df["RigidBody001_tX"].max())
    y_max = max(resampled_df["world_MocapLimb_Pos_y"].max(), resampled_df["RigidBody001_tY"].max())
    z_max = max(resampled_df["world_MocapLimb_Pos_z"].max(), resampled_df["RigidBody001_tZ"].max())
    x_max = x_max + np.abs(x_max*0.2)
    y_max = y_max + np.abs(y_max*0.2)
    z_max = z_max + np.abs(z_max*0.2)

    for t in range(0,len(resampled_df), 1000):
        quiverRigidBody = ax.quiver(resampled_df["RigidBody001_tX"][t], resampled_df["RigidBody001_tY"][t], resampled_df["RigidBody001_tZ"][t], *world_RigidBody_Ori_mat[t,:,0], color='red', linewidth=4, length=70)
        quiverRigidBody = ax.quiver(resampled_df["RigidBody001_tX"][t], resampled_df["RigidBody001_tY"][t], resampled_df["RigidBody001_tZ"][t], *world_RigidBody_Ori_mat[t,:,1], color='red', linewidth=1, length=50)
        quiverRigidBody = ax.quiver(resampled_df["RigidBody001_tX"][t], resampled_df["RigidBody001_tY"][t], resampled_df["RigidBody001_tZ"][t], *world_RigidBody_Ori_mat[t,:,2], color='red', linewidth=3, length=50)
        
        quiverRigidBody = ax.quiver(resampled_df["world_ThreePointsFrame_x"][t], resampled_df["world_ThreePointsFrame_y"][t], resampled_df["world_ThreePointsFrame_z"][t], *world_ThreePointsFrame_Ori_mat[t][:,0], color='orange', linewidth=3, length=70)
        quiverRigidBody = ax.quiver(resampled_df["world_ThreePointsFrame_x"][t], resampled_df["world_ThreePointsFrame_y"][t], resampled_df["world_ThreePointsFrame_z"][t], *world_ThreePointsFrame_Ori_mat[t][:,1], color='orange', linewidth=1, length=50)
        quiverRigidBody = ax.quiver(resampled_df["world_ThreePointsFrame_x"][t], resampled_df["world_ThreePointsFrame_y"][t], resampled_df["world_ThreePointsFrame_z"][t], *world_ThreePointsFrame_Ori_mat[t][:,2], color='orange', linewidth=1, length=50)

        quiverMocapLimb = ax.quiver(resampled_df["world_MocapLimb_Pos_x"][t], resampled_df["world_MocapLimb_Pos_y"][t], resampled_df["world_MocapLimb_Pos_z"][t], *world_MocapLimb_Ori_mat[t][:,0], color='blue', linewidth=4, length=70)
        quiverMocapLimb = ax.quiver(resampled_df["world_MocapLimb_Pos_x"][t], resampled_df["world_MocapLimb_Pos_y"][t], resampled_df["world_MocapLimb_Pos_z"][t], *world_MocapLimb_Ori_mat[t][:,1], color='blue', linewidth=1, length=50)
        quiverMocapLimb = ax.quiver(resampled_df["world_MocapLimb_Pos_x"][t], resampled_df["world_MocapLimb_Pos_y"][t], resampled_df["world_MocapLimb_Pos_z"][t], *world_MocapLimb_Ori_mat[t][:,2], color='blue', linewidth=3, length=50)

        quiverMocapLimb = ax.quiver(resampled_df["world_MocapLimb_Pos_x"][t], resampled_df["world_MocapLimb_Pos_y"][t], resampled_df["world_MocapLimb_Pos_z"][t], *mocapLimb_ThreePointsFrame_Ori[:,0], color='green', linewidth=8, length=70)
        quiverMocapLimb = ax.quiver(resampled_df["world_MocapLimb_Pos_x"][t], resampled_df["world_MocapLimb_Pos_y"][t], resampled_df["world_MocapLimb_Pos_z"][t], *mocapLimb_ThreePointsFrame_Ori[:,1], color='green', linewidth=1, length=50)
        quiverMocapLimb = ax.quiver(resampled_df["world_MocapLimb_Pos_x"][t], resampled_df["world_MocapLimb_Pos_y"][t], resampled_df["world_MocapLimb_Pos_z"][t], *mocapLimb_ThreePointsFrame_Ori[:,2], color='green', linewidth=3, length=50)

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
    ax.plot(resampled_df["world_ThreePointsFrame_x"], resampled_df["world_ThreePointsFrame_y"], resampled_df["world_ThreePointsFrame_z"], color='darkorange')
    ax.plot(resampled_df["world_MocapLimb_Pos_x"], resampled_df["world_MocapLimb_Pos_y"], resampled_df["world_MocapLimb_Pos_z"], color='darkblue')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])


    # Show the plot
    plt.show()


# Specify your patterns
patterns = ['Marker', 'threePoints']
# Get columns that contain any of the patterns
cols_to_drop = resampled_df.columns[resampled_df.columns.str.contains('|'.join(patterns))]
# Drop these columns
resampled_df = resampled_df.drop(columns=cols_to_drop)


# Save the DataFrame to a new CSV file
if(len(sys.argv) > 2):
    save_csv = sys.argv[2].lower()
else:
    save_csv = input("Do you want to save the data as a CSV file? (y/n): ")
    save_csv = save_csv.lower()

if save_csv == 'y':
    resampled_df.to_csv(output_csv_file_path_mocap, index=False, sep=';')
    print("Resampled mocap has been saved to ", output_csv_file_path_mocap)
else:
    print("Data not saved.")

