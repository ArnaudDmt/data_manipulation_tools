import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

timeStepInput = input("Please enter the timestep of the controller in milliseconds: ")

# Convert the input to a double
try:
    timeStep = int(timeStepInput)
    resample_str = f'{timeStep}ms'
    print(f"Resampling the MoCap data at {timeStep} ms")
except ValueError:
    print("That's not a valid int!")


# Define a list of patterns you want to match
pattern1 = ['Time(Seconds)','Marker1', 'Marker2', 'Marker3']  # Add more patterns as needed
pattern2 = r'RigidBody(?!.*Marker)'


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
        if 'Pos' in col:
            dataframe[col] = dataframe[col] / 1000
    return dataframe

def rename_columns(dataframe):
    for col in dataframe.columns:
        if 'Marker' in col:
            dataframe = dataframe.rename(columns={col: col.replace('RigidBody001:', '')})
    return dataframe

def time_to_seconds(time_value):
    return time_value.hour * 3600 + time_value.minute * 60 + time_value.second + time_value.microsecond / 1000000.

def rotation_matrix_to_quaternion(R):
    q0 = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
    q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
    q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
    return np.array([q0, q1, q2, q3])


# Load the CSV file into a DataFrame
csv_file_path = 'mocapData.csv'
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

# Plot the original and resampled data
ax = df_filtered.plot(x="Time(Seconds)", y=["Marker1_tX", "Marker1_tY", "Marker1_tZ"], label=['Original_1', 'Original_2', 'Original_3'])
resampled_df.plot(x="Time(Seconds)", y=["Marker1_tX", "Marker1_tY", "Marker1_tZ"], label=['Resampled_1', 'Resampled_2', 'Resampled_3'], ax=ax)

plt.legend()
plt.show()

# Computation of the frame transformation

head_P1_pos = np.array([-114.6, 1.4, 191.3])
head_P2_pos = np.array([95.2, -49.9, 202.6])
head_P3_pos = np.array([46.6, 71.1, 4.9])

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

# Convert quaternions to rotation matrices
rRigidBody = R.from_quat(resampled_df[["RigidBody001_qX", "RigidBody001_qY", "RigidBody001_qZ", "RigidBody001_qW"]].values)
rRigidBody_mat = rRigidBody.as_dcm()

worldHeadOri_mats = []
world_threePointsFrame_ori_mats = []
errorWithRigidBody_mats = []

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
# Iterate over each row
for index, row in resampled_df.iterrows():
    # Update world_P1_pos, world_P2_pos, and world_P3_pos from the current row
    world_P1_pos = np.array([row['Marker1_tX'], row['Marker1_tY'], row['Marker1_tZ']])
    world_P2_pos = np.array([row['Marker2_tX'], row['Marker2_tY'], row['Marker2_tZ']])
    world_P3_pos = np.array([row['Marker3_tX'], row['Marker3_tY'], row['Marker3_tZ']])
    
    worldThreePointsFramePos = (world_P1_pos + world_P2_pos + world_P3_pos) / 3
    world_threePointsFrame_x = (world_P2_pos - world_P1_pos)
    world_threePointsFrame_x = world_threePointsFrame_x / np.linalg.norm(world_threePointsFrame_x)
    world_threePointsFrame_y = np.cross(world_threePointsFrame_x, world_P3_pos - world_P1_pos)
    world_threePointsFrame_y = world_threePointsFrame_y / np.linalg.norm(world_threePointsFrame_y)
    world_threePointsFrame_z = np.cross(world_threePointsFrame_x, world_threePointsFrame_y)
    world_threePointsFrame_z = world_threePointsFrame_z / np.linalg.norm(world_threePointsFrame_z)
    world_threePointsFrame_ori = np.column_stack((world_threePointsFrame_x, world_threePointsFrame_y, world_threePointsFrame_z))
    
    worldHeadOri_mat = np.matmul(world_threePointsFrame_ori, threePointsFrame_head_ori)
    worldHeadOri = rotation_matrix_to_quaternion(worldHeadOri_mat)
    worldHeadPos = worldThreePointsFramePos + np.matmul(world_threePointsFrame_ori,threePointsFrame_head_pos)
    world_threePointsFrame_ori_quat = rotation_matrix_to_quaternion(world_threePointsFrame_ori)

    worldHeadOri_mats.append(worldHeadOri_mat)

    world_threePointsFrame_ori_mats.append(world_threePointsFrame_ori)

    errorWithRigidBody_mats.append(np.matmul(worldHeadOri_mat.T, rRigidBody_mat[index]))

    resampled_df.at[index, 'worldHeadPos_x'] = worldHeadPos[0]
    resampled_df.at[index, 'worldHeadPos_y'] = worldHeadPos[1]
    resampled_df.at[index, 'worldHeadPos_z'] = worldHeadPos[2]
    resampled_df.at[index, 'worldHeadOri_qx'] = worldHeadOri[1]
    resampled_df.at[index, 'worldHeadOri_qy'] = worldHeadOri[2]
    resampled_df.at[index, 'worldHeadOri_qz'] = worldHeadOri[3]
    resampled_df.at[index, 'worldHeadOri_qw'] = worldHeadOri[0]

    resampled_df.at[index, 'world_threePointsFrame_x'] = worldThreePointsFramePos[0]
    resampled_df.at[index, 'world_threePointsFrame_y'] = worldThreePointsFramePos[1]
    resampled_df.at[index, 'world_threePointsFrame_z'] = worldThreePointsFramePos[2]
    resampled_df.at[index, 'world_threePointsFrame_qx'] = world_threePointsFrame_ori_quat[1]
    resampled_df.at[index, 'world_threePointsFrame_qy'] = world_threePointsFrame_ori_quat[2]
    resampled_df.at[index, 'world_threePointsFrame_qz'] = world_threePointsFrame_ori_quat[3]
    resampled_df.at[index, 'world_threePointsFrame_qw'] = world_threePointsFrame_ori_quat[0]



errorR = R.from_dcm(errorWithRigidBody_mats)
errorQuat = errorR.as_quat()
xs = [x for x in range(len(errorQuat))]
errorQuat_x = [quat[0] for quat in errorQuat]
errorQuat_y = [quat[1] for quat in errorQuat]
errorQuat_z = [quat[2] for quat in errorQuat]
errorQuat_w = [quat[3] for quat in errorQuat]
fig1, ax1 = plt.subplots(1, 1)

ax1.plot(xs, errorQuat_x, label='errorQuat_x')
ax1.plot(xs, errorQuat_y, label='errorQuat_y')
ax1.plot(xs, errorQuat_z, label='errorQuat_z')
ax1.plot(xs, errorQuat_w, label='errorQuat_w')

ax1.legend()


# Create a 3D plot
fig, ax = plt.subplots(1, 1)
ax = fig.add_subplot(111, projection='3d')

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

for t in range(0,len(resampled_df), 200):
    quiverRigidBody = ax.quiver(resampled_df["RigidBody001_tX"][t], resampled_df["RigidBody001_tY"][t], resampled_df["RigidBody001_tZ"][t], *rRigidBody_mat[t,:,0], color='red', linewidth=4, length=70)
    quiverRigidBody = ax.quiver(resampled_df["RigidBody001_tX"][t], resampled_df["RigidBody001_tY"][t], resampled_df["RigidBody001_tZ"][t], *rRigidBody_mat[t,:,1], color='red', linewidth=1, length=50)
    quiverRigidBody = ax.quiver(resampled_df["RigidBody001_tX"][t], resampled_df["RigidBody001_tY"][t], resampled_df["RigidBody001_tZ"][t], *rRigidBody_mat[t,:,2], color='red', linewidth=3, length=50)
    
    quiverRigidBody = ax.quiver(resampled_df["world_threePointsFrame_x"][t], resampled_df["world_threePointsFrame_y"][t], resampled_df["world_threePointsFrame_z"][t], *world_threePointsFrame_ori_mats[t][:,0], color='orange', linewidth=3, length=70)
    quiverRigidBody = ax.quiver(resampled_df["world_threePointsFrame_x"][t], resampled_df["world_threePointsFrame_y"][t], resampled_df["world_threePointsFrame_z"][t], *world_threePointsFrame_ori_mats[t][:,1], color='orange', linewidth=1, length=50)
    quiverRigidBody = ax.quiver(resampled_df["world_threePointsFrame_x"][t], resampled_df["world_threePointsFrame_y"][t], resampled_df["world_threePointsFrame_z"][t], *world_threePointsFrame_ori_mats[t][:,2], color='orange', linewidth=1, length=50)

    quiverHead = ax.quiver(resampled_df["worldHeadPos_x"][t], resampled_df["worldHeadPos_y"][t], resampled_df["worldHeadPos_z"][t], *worldHeadOri_mats[t][:,0], color='blue', linewidth=4, length=70)
    quiverHead = ax.quiver(resampled_df["worldHeadPos_x"][t], resampled_df["worldHeadPos_y"][t], resampled_df["worldHeadPos_z"][t], *worldHeadOri_mats[t][:,1], color='blue', linewidth=1, length=50)
    quiverHead = ax.quiver(resampled_df["worldHeadPos_x"][t], resampled_df["worldHeadPos_y"][t], resampled_df["worldHeadPos_z"][t], *worldHeadOri_mats[t][:,2], color='blue', linewidth=3, length=50)

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
patterns = ['Marker', 'threePoints', 'RigidBody']
# Get columns that contain any of the patterns
cols_to_drop = resampled_df.columns[resampled_df.columns.str.contains('|'.join(patterns))]
# Drop these columns
resampled_df = resampled_df.drop(columns=cols_to_drop)
resampled_df = convert_mm_to_m(resampled_df)


# Save the DataFrame to a new CSV file
output_csv_file_path = 'output_file.csv'
resampled_df.to_csv(output_csv_file_path, index=False)
print("Output CSV file has been saved to", output_csv_file_path)