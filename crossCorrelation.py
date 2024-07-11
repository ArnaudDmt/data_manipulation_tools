import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.spatial.transform import Rotation as R




###############################  User input for the timestep  ###############################

timeStepInput = input("Please enter the timestep of the controller in milliseconds: ")

# Convert the input to a double
try:
    timeStepInt = int(timeStepInput)
    timeStepFloat = float(timeStepInput)/1000.0
    resample_str = f'{timeStepInt}ms'
except ValueError:
    print("That's not a valid int!")




###############################  Main variables initialization  ###############################

output_csv_file_path = 'realignedMocapLimbData.csv'



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

###############################  Poses retrieval  ###############################

# Extracting the poses related to the mocap
world_mocapRigidBody_Pos = np.array([mocapData['RigidBody001_tX'], mocapData['RigidBody001_tY'], mocapData['RigidBody001_tZ']]).T
world_RigidBody_Ori_R = R.from_quat(mocapData[["RigidBody001_qX", "RigidBody001_qY", "RigidBody001_qZ", "RigidBody001_qW"]].values)

world_mocapLimb_Pos = np.array([mocapData['world_MocapLimb_Pos_x'], mocapData['world_MocapLimb_Pos_y'], mocapData['world_MocapLimb_Pos_z']]).T
world_mocapLimb_Ori_R = R.from_quat(mocapData[["world_MocapLimb_Ori_qx", "world_MocapLimb_Ori_qy", "world_MocapLimb_Ori_qz", "world_MocapLimb_Ori_qw"]].values)


# Extracting the poses coming from mc_rtc
world_VanyteBody_Pos = np.array([observer_data['Observers_MainObserverPipeline_MCVanytEstimator_mocap_worldBodyKine_position_x'], observer_data['Observers_MainObserverPipeline_MCVanytEstimator_mocap_worldBodyKine_position_y'], observer_data['Observers_MainObserverPipeline_MCVanytEstimator_mocap_worldBodyKine_position_z']]).T
world_VanyteBody_Ori_R = R.from_quat(observer_data[["Observers_MainObserverPipeline_MCVanytEstimator_mocap_worldBodyKine_ori_x", "Observers_MainObserverPipeline_MCVanytEstimator_mocap_worldBodyKine_ori_y", "Observers_MainObserverPipeline_MCVanytEstimator_mocap_worldBodyKine_ori_z", "Observers_MainObserverPipeline_MCVanytEstimator_mocap_worldBodyKine_ori_w"]].values)
# We get the inverse of the orientation as the inverse quaternion was stored
world_VanyteBody_Ori_R = world_VanyteBody_Ori_R.inv()



###############################  Visualization of the extracted poses  ###############################


# Plot of the resulting positions
figPositions = go.Figure()

figPositions.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_Pos[:,0], mode='lines', name='world_Vanyte_Body_pos_x'))
figPositions.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_Pos[:,1], mode='lines', name='world_Vanyte_Body_pos_y'))
figPositions.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_Pos[:,2], mode='lines', name='world_Vanyte_Body_pos_z'))

figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_Pos[:,0], mode='lines', name='world_mocapLimb_Pos_x'))
figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_Pos[:,1], mode='lines', name='world_mocapLimb_Pos_y'))
figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_Pos[:,2], mode='lines', name='world_mocapLimb_Pos_z'))

figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=mocapData["RigidBody001_tX"], mode='lines', name='world_RigidBody_pos_x'))
figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=mocapData["RigidBody001_tY"], mode='lines', name='world_RigidBody_pos_y'))
figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=mocapData["RigidBody001_tZ"], mode='lines', name='world_RigidBody_pos_z'))

figPositions.update_layout(title="Resulting positions")

# Show the plotly figures
figPositions.show()


# Plot of the resulting orientations
figOrientations = go.Figure()

world_VanyteBody_Ori_euler = world_VanyteBody_Ori_R.as_euler("xyz")
world_mocapLimb_Ori_euler = world_mocapLimb_Ori_R.as_euler("xyz")
world_RigidBody_Ori_euler = world_RigidBody_Ori_R.as_euler("xyz")

world_VanyteBody_Ori_euler_continuous = continuous_euler(world_VanyteBody_Ori_euler)
world_mocapLimb_Ori_euler_continuous = continuous_euler(world_mocapLimb_Ori_euler)
world_RigidBody_Ori_euler_continuous = continuous_euler(world_RigidBody_Ori_euler)

figOrientations.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_Ori_euler_continuous[:,0], mode='lines', name='world_Vanyte_Body_ori_roll'))
figOrientations.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_Ori_euler_continuous[:,1], mode='lines', name='world_Vanyte_Body_ori_pitch'))
figOrientations.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_Ori_euler_continuous[:,2], mode='lines', name='world_Vanyte_Body_ori_yaw'))

figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_Ori_euler_continuous[:,0], mode='lines', name='world_mocapLimb_Ori_roll'))
figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_Ori_euler_continuous[:,1], mode='lines', name='world_mocapLimb_Ori_pitch'))
figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_Ori_euler_continuous[:,2], mode='lines', name='world_mocapLimb_Ori_yaw'))

figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,0], mode='lines', name='world_RigidBody_ori_roll'))
figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,1], mode='lines', name='world_RigidBody_ori_pitch'))
figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,2], mode='lines', name='world_RigidBody_ori_yaw'))


figOrientations.update_layout(title="Resulting orientations")

# Show the plotly figures
figOrientations.show()



###############################  Local linear velocity of the mocapLimb in the world  ###############################

# We compute the velocity of the mocapLimb in the world
world_mocapLimb_Vel_x = np.diff(world_mocapLimb_Pos[:,0])/timeStepFloat
world_mocapLimb_Vel_y = np.diff(world_mocapLimb_Pos[:,1])/timeStepFloat
world_mocapLimb_Vel_z = np.diff(world_mocapLimb_Pos[:,2])/timeStepFloat
world_mocapLimb_Vel_x = np.insert(world_mocapLimb_Vel_x, 0, 0.0, axis=0)
world_mocapLimb_Vel_y = np.insert(world_mocapLimb_Vel_y, 0, 0.0, axis=0)
world_mocapLimb_Vel_z = np.insert(world_mocapLimb_Vel_z, 0, 0.0, axis=0)
world_mocapLimb_Vel = np.stack((world_mocapLimb_Vel_x, world_mocapLimb_Vel_y, world_mocapLimb_Vel_z), axis = 1)

# We compute the velocity of the mocap's rigid body in the world
world_RigidBody_Vel_x = np.diff(world_mocapRigidBody_Pos[:,0])/timeStepFloat
world_RigidBody_Vel_y = np.diff(world_mocapRigidBody_Pos[:,1])/timeStepFloat
world_RigidBody_Vel_z = np.diff(world_mocapRigidBody_Pos[:,2])/timeStepFloat
world_RigidBody_Vel_x = np.insert(world_RigidBody_Vel_x, 0, 0.0, axis=0)
world_RigidBody_Vel_y = np.insert(world_RigidBody_Vel_y, 0, 0.0, axis=0)
world_RigidBody_Vel_z = np.insert(world_RigidBody_Vel_z, 0, 0.0, axis=0)
world_RigidBody_Vel = np.stack((world_RigidBody_Vel_x, world_RigidBody_Vel_y, world_RigidBody_Vel_z), axis = 1)

# We compute the velocity estimated by the Vanyte in the world
world_VanyteBody_Vel_x = np.diff(world_VanyteBody_Pos[:,0])/timeStepFloat
world_VanyteBody_Vel_y = np.diff(world_VanyteBody_Pos[:,1])/timeStepFloat
world_VanyteBody_Vel_z = np.diff(world_VanyteBody_Pos[:,2])/timeStepFloat
world_VanyteBody_Vel_x = np.insert(world_VanyteBody_Vel_x, 0, 0.0, axis=0)
world_VanyteBody_Vel_y = np.insert(world_VanyteBody_Vel_y, 0, 0.0, axis=0)
world_VanyteBody_Vel_z = np.insert(world_VanyteBody_Vel_z, 0, 0.0, axis=0)
world_VanyteBody_Vel = np.stack((world_VanyteBody_Vel_x, world_VanyteBody_Vel_y, world_VanyteBody_Vel_z), axis = 1)


# Now we get the local linear velocity
world_mocapLimb_LocVel = world_mocapLimb_Ori_R.apply(world_mocapLimb_Vel, inverse=True)
world_RigidBody_LocVel = world_RigidBody_Ori_R.apply(world_RigidBody_Vel, inverse=True)
world_VanyteBody_LocVel = world_VanyteBody_Ori_R.apply(world_VanyteBody_Vel, inverse=True)

# Plot of the resulting poses
fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_LocVel[:,0], mode='lines', name='world_mocapLimb_LocVel_x'))
fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_LocVel[:,1], mode='lines', name='world_mocapLimb_LocVel_y'))
fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_LocVel[:,2], mode='lines', name='world_mocapLimb_LocVel_z'))
fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_LocVel[:,0], mode='lines', name='world_RigidBody_LocalLinVel_x'))
fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_LocVel[:,1], mode='lines', name='world_RigidBody_LocalLinVel_y'))
fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_LocVel[:,2], mode='lines', name='world_RigidBody_LocalLinVel_z'))
fig2.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_LocVel[:,0], mode='lines', name='world_VanyteBody_LocVel_x'))
fig2.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_LocVel[:,1], mode='lines', name='world_VanyteBody_LocVel_y'))
fig2.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_LocVel[:,2], mode='lines', name='world_VanyteBody_LocVel_z'))

fig2.update_layout(title="Local linear velocity of the mocapLimb in the world / vs the one of the rigid body")
# Show the plotly figures
fig2.show()




###############################  Cross correlation  ###############################

def match_array_length(arr, desired_length):
    """
    Appends a row containing the final value of each column until the array length matches the desired length.
    
    Args:
        arr (np.ndarray): Input array.
        desired_length (int): Desired length of the array.
    
    Returns:
        np.ndarray: Array with the desired length.
    """
    # Get the final values of each column
    final_values = arr[-1]

    # Repeat the final values to match the desired length
    num_repeats = max(0, desired_length - len(arr))
    repeated_values = np.tile(final_values, (num_repeats, 1))

    # Append the repeated values to the original array
    result = np.vstack([arr, repeated_values])

    return result


def add_rows(df, num_rows, atEnd=True):
    """
    Adds multiple rows to a pandas DataFrame either at the beginning or at the end (efficient implementation).

    Args:
        df (pd.DataFrame): The input DataFrame.
        num_rows (int): Number of rows to add.
        atEnd (bool, optional): Position to add rows (True for "end", False for "beginning"). Defaults to True.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    if atEnd:
        # Create a DataFrame with the same columns as the input DataFrame
        new_rows = pd.DataFrame(np.tile(df.iloc[-1].values, (num_rows, 1)), columns=df.columns)
        # Set the last index of new_rows to be equal to the first index of df minus 1
        new_rows.index = range(df.index[-1] + 1, df.index[-1] + 1 + num_rows)
        # Concatenate the new rows with the original DataFrame
        df = pd.concat([df, new_rows], ignore_index=False)
    else:
        # Create a DataFrame with the same columns as the input DataFrame
        new_rows = pd.DataFrame(np.tile(df.iloc[0].values, (num_rows, 1)), columns=df.columns)
        # Set the last index of new_rows to be equal to the first index of df minus 1
        new_rows.index = range(df.index[0] - num_rows, df.index[0])
        # Concatenate the new rows with the original DataFrame
        df = pd.concat([new_rows, df], ignore_index=False)
    return df

def realignData(data1, data2, data1_name, data2_name):
    figInit = go.Figure()

    data2 = match_array_length(data2, len(data1))

    time1 = np.arange(0, len(data1), step=1)
    time2 = np.arange(0, len(data2), step=1)
    figInit.add_trace(go.Scatter(x=time1, y=data1[:,0], mode='lines', name=f'{data1_name}_1'))
    figInit.add_trace(go.Scatter(x=time1, y=data1[:,1], mode='lines', name=f'{data1_name}_2'))
    figInit.add_trace(go.Scatter(x=time1, y=data1[:,2], mode='lines', name=f'{data1_name}_3'))

    figInit.add_trace(go.Scatter(x=time2, y=data2[:,0], mode='lines', name=f'{data2_name}_1'))
    figInit.add_trace(go.Scatter(x=time2, y=data2[:,1], mode='lines', name=f'{data2_name}_2'))
    figInit.add_trace(go.Scatter(x=time2, y=data2[:,2], mode='lines', name=f'{data2_name}_3'))
    figInit.update_layout(title="Data before alignment.")
    figInit.show()

    
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

    if(shift > 0):
        print(f"The mocap data is {shift} iterations ahead of the controller.")
    if(shift < 0):
        print(f"The mocap data is {shift} iterations behind the controller.")

    data2_shifted = np.roll(data2, shift, axis=0)

    fig = go.Figure()
    time1 = np.arange(0, len(data1), step=1)
    time2 = np.arange(0, len(data2), step=1)

    fig.add_trace(go.Scatter(x=time1, y=data1[:,0], mode='lines', name=f'{data1_name}_1'))
    fig.add_trace(go.Scatter(x=time1, y=data1[:,1], mode='lines', name=f'{data1_name}_2'))
    fig.add_trace(go.Scatter(x=time1, y=data1[:,2], mode='lines', name=f'{data1_name}_3'))

    fig.add_trace(go.Scatter(x=time2, y=data2_shifted[:,0], mode='lines', name=f'{data2_name}_1'))
    fig.add_trace(go.Scatter(x=time2, y=data2_shifted[:,1], mode='lines', name=f'{data2_name}_2'))
    fig.add_trace(go.Scatter(x=time2, y=data2_shifted[:,2], mode='lines', name=f'{data2_name}_3'))
    fig.update_layout(title="Previsualization of data after alignment.")
    fig.show()

    return data2, shift


world_mocapLimb_LocVel, shift = realignData(world_VanyteBody_LocVel, world_mocapLimb_LocVel, "world_mocapLimb_LocVel", "world_VanyteBody_LocVel")


# Version which receives the shift to apply as an input
def realignData(data_to_shift,  shift):
    print(f"The observer_data will be shifted by {shift} indexes.")

    data_shifted = data_to_shift
    data_shifted.index += shift

    diffIndexInit = data_shifted.index[0] - observer_data.index[0]
    diffIndexFinal = data_shifted.index[-1] - observer_data.index[-1]

    if(diffIndexInit > 0):
        # Adding diffIndexInit rows at the beginning of the dataframe
        data_shifted = add_rows(data_shifted, diffIndexInit, False)
    if(diffIndexInit < 0):
        # Removing first diffIndexInit rows
        data_shifted = data_shifted.iloc[abs(diffIndexInit):]

    if(diffIndexFinal < 0):
        # Adding diffIndexInit rows at the end of the dataframe
        data_shifted = add_rows(data_shifted, abs(diffIndexFinal), True)
    if(diffIndexFinal > 0):
        # Removing last diffIndexFinal rows
        data_shifted = data_shifted.iloc[:diffIndexFinal]

    figAlignedIndexes = go.Figure()
    time = np.arange(0, len(data_shifted), step=1)
    figAlignedIndexes.add_trace(go.Scatter(x=data_shifted.index, y=data_shifted['world_MocapLimb_Pos_x'], mode='lines', name='world_MocapLimb_Pos_x'))
    figAlignedIndexes.add_trace(go.Scatter(x=data_shifted.index, y=data_shifted['world_MocapLimb_Pos_y'], mode='lines', name='world_MocapLimb_Pos_y'))
    figAlignedIndexes.add_trace(go.Scatter(x=data_shifted.index, y=data_shifted['world_MocapLimb_Pos_z'], mode='lines', name='world_MocapLimb_Pos_z'))
    figAlignedIndexes.add_trace(go.Scatter(x=observer_data.index, y=world_VanyteBody_Pos[:,0], mode='lines', name='world_VanyteBody_Pos_x'))
    figAlignedIndexes.add_trace(go.Scatter(x=observer_data.index, y=world_VanyteBody_Pos[:,1], mode='lines', name='world_VanyteBody_Pos_y'))
    figAlignedIndexes.add_trace(go.Scatter(x=observer_data.index, y=world_VanyteBody_Pos[:,2], mode='lines', name='world_VanyteBody_Pos_z'))
    figAlignedIndexes.update_layout(title="Data after index alignment.")
    figAlignedIndexes.show()

    data_shifted['Time(Seconds)'] = observer_data["t"]

    return data_shifted

realignedMocapData = realignData(mocapData, shift)

###############################  Shifted poses retrieval  ###############################

# Extracting the poses related to the realignedMocapData
world_mocapRigidBody_Pos = np.array([realignedMocapData['RigidBody001_tX'], realignedMocapData['RigidBody001_tY'], realignedMocapData['RigidBody001_tZ']]).T
world_RigidBody_Ori_R = R.from_quat(realignedMocapData[["RigidBody001_qX", "RigidBody001_qY", "RigidBody001_qZ", "RigidBody001_qW"]].values)

world_mocapLimb_Pos = np.array([realignedMocapData['world_MocapLimb_Pos_x'], realignedMocapData['world_MocapLimb_Pos_y'], realignedMocapData['world_MocapLimb_Pos_z']]).T
world_mocapLimb_Ori_R = R.from_quat(realignedMocapData[["world_MocapLimb_Ori_qx", "world_MocapLimb_Ori_qy", "world_MocapLimb_Ori_qz", "world_MocapLimb_Ori_qw"]].values)


###############################  Visualization of the extracted poses  ###############################


# Plot of the resulting positions
figPositions_realigned = go.Figure()

figPositions_realigned.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_Pos[:,0], mode='lines', name='world_Vanyte_Body_pos_x_realigned'))
figPositions_realigned.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_Pos[:,1], mode='lines', name='world_Vanyte_Body_pos_y_realigned'))
figPositions_realigned.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_Pos[:,2], mode='lines', name='world_Vanyte_Body_pos_z_realigned'))

figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Pos[:,0], mode='lines', name='world_mocapLimb_Pos_x_realigned'))
figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Pos[:,1], mode='lines', name='world_mocapLimb_Pos_y_realigned'))
figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Pos[:,2], mode='lines', name='world_mocapLimb_Pos_z_realigned'))

figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=realignedMocapData["RigidBody001_tX"], mode='lines', name='world_RigidBody_pos_x_realigned'))
figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=realignedMocapData["RigidBody001_tY"], mode='lines', name='world_RigidBody_pos_y_realigned'))
figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=realignedMocapData["RigidBody001_tZ"], mode='lines', name='world_RigidBody_pos_z_realigned'))

figPositions_realigned.update_layout(title="Realigned positions")

# Show the plotly figures
figPositions_realigned.show()


# Plot of the resulting orientations
figOrientations_realigned = go.Figure()

#world_VanyteBody_Ori_euler = world_VanyteBody_Ori_R.as_euler("xyz")
world_mocapLimb_Ori_euler = world_mocapLimb_Ori_R.as_euler("xyz")
world_RigidBody_Ori_euler = world_RigidBody_Ori_R.as_euler("xyz")

#world_VanyteBody_Ori_euler_continuous = continuous_euler(world_VanyteBody_Ori_euler)
world_mocapLimb_Ori_euler_continuous = continuous_euler(world_mocapLimb_Ori_euler)
world_RigidBody_Ori_euler_continuous = continuous_euler(world_RigidBody_Ori_euler)

figOrientations_realigned.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_Ori_euler_continuous[:,0], mode='lines', name='world_Vanyte_Body_ori_roll_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_Ori_euler_continuous[:,1], mode='lines', name='world_Vanyte_Body_ori_pitch_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_Ori_euler_continuous[:,2], mode='lines', name='world_Vanyte_Body_ori_yaw_realigned'))

figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Ori_euler_continuous[:,0], mode='lines', name='world_mocapLimb_Ori_roll_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Ori_euler_continuous[:,1], mode='lines', name='world_mocapLimb_Ori_pitch_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Ori_euler_continuous[:,2], mode='lines', name='world_mocapLimb_Ori_yaw_realigned'))

figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,0], mode='lines', name='world_RigidBody_ori_roll_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,1], mode='lines', name='world_RigidBody_ori_pitch_realigned'))
figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,2], mode='lines', name='world_RigidBody_ori_yaw_realigned'))

figOrientations_realigned.update_layout(title="Realigned orientations")

# Show the plotly figures
figOrientations_realigned.show()






###############################  Orientation difference wrt the initial orientation  ###############################

world_mocapLimb_Ori_R_transfo = world_mocapLimb_Ori_R * world_mocapLimb_Ori_R[0].inv()
world_RigidBody_Ori_R_transfo = world_RigidBody_Ori_R * world_RigidBody_Ori_R[0].inv()
world_VanyteBody_Ori_R_transfo = world_VanyteBody_Ori_R * world_VanyteBody_Ori_R[0].inv()

world_mocapLimb_Ori_transfo_euler = world_mocapLimb_Ori_R_transfo.as_euler("xyz")
world_RigidBody_Ori_transfo_euler = world_RigidBody_Ori_R_transfo.as_euler("xyz")
world_VanyteBody_Ori_transfo_euler = world_VanyteBody_Ori_R_transfo.as_euler("xyz")

world_mocapLimb_Ori_transfo_euler_continuous = continuous_euler(world_mocapLimb_Ori_transfo_euler)
world_RigidBody_Ori_transfo_euler_continuous = continuous_euler(world_RigidBody_Ori_transfo_euler)
world_VanyteBody_Ori_transfo_euler_continuous = continuous_euler(world_VanyteBody_Ori_transfo_euler)

fig3 = go.Figure()

fig3.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_mocapLimb_Ori_transfo_roll'))
fig3.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_mocapLimb_Ori_transfo_pitch'))
fig3.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_mocapLimb_Ori_transfo_yaw'))

fig3.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_RigidBody_Ori_transfo_roll'))
fig3.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_RigidBody_Ori_transfo_pitch'))
fig3.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_RigidBody_Ori_transfo_yaw'))

fig3.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_VanyteBody_Ori_transfo_roll'))
fig3.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_VanyteBody_Ori_transfo_pitch'))
fig3.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteBody_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_VanyteBody_Ori_transfo_yaw'))

fig3.update_layout(title="Orientation transformations")

# Show the plotly figures
fig3.show()



world_mocapLimb_Ori_quat = world_mocapLimb_Ori_R.as_quat()
output_df = pd.DataFrame({'t': observer_data['t'], 'realignedWorldMocapLimbPos_x': world_mocapLimb_Pos[:,0], 'realignedWorldMocapLimbPos_y': world_mocapLimb_Pos[:,1], 'realignedWorldMocapLimbPos_z': world_mocapLimb_Pos[:,2], 'realignedWorldMocapLimbOri_x': world_mocapLimb_Ori_quat[:,0], 'realignedWorldMocapLimbOri_y': world_mocapLimb_Ori_quat[:,1], 'realignedWorldMocapLimbOri_z': world_mocapLimb_Ori_quat[:,2], 'realignedWorldMocapLimbOri_w': world_mocapLimb_Ori_quat[:,3]})



# Save the DataFrame to a new CSV file
save_csv = input("Do you want to save the data as a CSV file? (y/n): ")

if save_csv.lower() == 'y':
    output_df.to_csv(output_csv_file_path, index=False)
    print("Output CSV file has been saved to ", output_csv_file_path)
else:
    print("Data not saved.")

