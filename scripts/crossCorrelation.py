import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.spatial.transform import Rotation as R





###############################  Main variables initialization  ###############################

displayLogs = True
path_to_project = ".."
scriptName = "crossCorrelation"


###############################  User inputs  ###############################


if(len(sys.argv) > 1):
    timeStepInput = sys.argv[1]
    if(len(sys.argv) > 2):
        displayLogs = sys.argv[2].lower() == 'true'
    if(len(sys.argv) > 4):
        path_to_project = sys.argv[4]
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


output_csv_file_path = f'{path_to_project}/output_data/realignedMocapLimbData.csv'
# Load the CSV files into pandas dataframes
observer_data = pd.read_csv(f'{path_to_project}/output_data/lightData.csv',  delimiter=';')
mocapData = pd.read_csv(f'{path_to_project}/output_data/resampledMocapData.csv', delimiter=';')



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




###############################  Poses retrieval  ###############################

# Extracting the poses related to the mocap
world_mocapRigidBody_Pos = np.array([mocapData['RigidBody001_tX'], mocapData['RigidBody001_tY'], mocapData['RigidBody001_tZ']]).T
world_RigidBody_Ori_R = R.from_quat(mocapData[["RigidBody001_qX", "RigidBody001_qY", "RigidBody001_qZ", "RigidBody001_qW"]].values)

world_mocapLimb_Pos = np.array([mocapData['world_MocapLimb_Pos_x'], mocapData['world_MocapLimb_Pos_y'], mocapData['world_MocapLimb_Pos_z']]).T
world_mocapLimb_Ori_R = R.from_quat(mocapData[["world_MocapLimb_Ori_qx", "world_MocapLimb_Ori_qy", "world_MocapLimb_Ori_qz", "world_MocapLimb_Ori_qw"]].values)


# Extracting the poses coming from mc_rtc
world_ObserverLimb_Pos = np.array([observer_data['MocapAligner_worldBodyKine_position_x'], observer_data['MocapAligner_worldBodyKine_position_y'], observer_data['MocapAligner_worldBodyKine_position_z']]).T
world_ObserverLimb_Ori_R = R.from_quat(observer_data[["MocapAligner_worldBodyKine_ori_x", "MocapAligner_worldBodyKine_ori_y", "MocapAligner_worldBodyKine_ori_z", "MocapAligner_worldBodyKine_ori_w"]].values)
# We get the inverse of the orientation as the inverse quaternion was stored
world_ObserverLimb_Ori_R = world_ObserverLimb_Ori_R.inv()



###############################  Visualization of the extracted poses  ###############################


if(displayLogs):
    # Plot of the resulting positions
    figPositions = go.Figure()

    figPositions.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Pos[:,0], mode='lines', name='world_Observer_Body_pos_x'))
    figPositions.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Pos[:,1], mode='lines', name='world_Observer_Body_pos_y'))
    figPositions.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Pos[:,2], mode='lines', name='world_Observer_Body_pos_z'))

    figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_Pos[:,0], mode='lines', name='world_mocapLimb_Pos_x'))
    figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_Pos[:,1], mode='lines', name='world_mocapLimb_Pos_y'))
    figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_Pos[:,2], mode='lines', name='world_mocapLimb_Pos_z'))

    figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=mocapData["RigidBody001_tX"], mode='lines', name='world_RigidBody_pos_x'))
    figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=mocapData["RigidBody001_tY"], mode='lines', name='world_RigidBody_pos_y'))
    figPositions.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=mocapData["RigidBody001_tZ"], mode='lines', name='world_RigidBody_pos_z'))

    figPositions.update_layout(title=f"{scriptName}: Positions before alignment")

    # Show the plotly figures
    figPositions.show()


    # Plot of the resulting orientations
    figOrientations = go.Figure()

    world_ObserverLimb_Ori_euler = world_ObserverLimb_Ori_R.as_euler("xyz")
    world_mocapLimb_Ori_euler = world_mocapLimb_Ori_R.as_euler("xyz")
    world_RigidBody_Ori_euler = world_RigidBody_Ori_R.as_euler("xyz")

    world_ObserverLimb_Ori_euler_continuous = continuous_euler(world_ObserverLimb_Ori_euler)
    world_mocapLimb_Ori_euler_continuous = continuous_euler(world_mocapLimb_Ori_euler)
    world_RigidBody_Ori_euler_continuous = continuous_euler(world_RigidBody_Ori_euler)

    figOrientations.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_euler_continuous[:,0], mode='lines', name='world_Observer_Body_ori_roll'))
    figOrientations.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_euler_continuous[:,1], mode='lines', name='world_Observer_Body_ori_pitch'))
    figOrientations.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_euler_continuous[:,2], mode='lines', name='world_Observer_Body_ori_yaw'))

    figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_Ori_euler_continuous[:,0], mode='lines', name='world_mocapLimb_Ori_roll'))
    figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_Ori_euler_continuous[:,1], mode='lines', name='world_mocapLimb_Ori_pitch'))
    figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_Ori_euler_continuous[:,2], mode='lines', name='world_mocapLimb_Ori_yaw'))

    figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,0], mode='lines', name='world_RigidBody_ori_roll'))
    figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,1], mode='lines', name='world_RigidBody_ori_pitch'))
    figOrientations.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,2], mode='lines', name='world_RigidBody_ori_yaw'))


    figOrientations.update_layout(title=f"{scriptName}: Orientations before alignment")

    # Show the plotly figures
    figOrientations.show()



###############################  Local linear velocity of the mocapLimb in the world  ###############################

# We compute the velocity of the mocapLimb in the world
world_mocapLimb_Vel_x = np.diff(world_mocapLimb_Pos[:,0])/timeStep_s
world_mocapLimb_Vel_y = np.diff(world_mocapLimb_Pos[:,1])/timeStep_s
world_mocapLimb_Vel_z = np.diff(world_mocapLimb_Pos[:,2])/timeStep_s
world_mocapLimb_Vel_x = np.insert(world_mocapLimb_Vel_x, 0, 0.0, axis=0)
world_mocapLimb_Vel_y = np.insert(world_mocapLimb_Vel_y, 0, 0.0, axis=0)
world_mocapLimb_Vel_z = np.insert(world_mocapLimb_Vel_z, 0, 0.0, axis=0)
world_mocapLimb_Vel = np.stack((world_mocapLimb_Vel_x, world_mocapLimb_Vel_y, world_mocapLimb_Vel_z), axis = 1)

# We compute the velocity of the mocap's rigid body in the world
world_RigidBody_Vel_x = np.diff(world_mocapRigidBody_Pos[:,0])/timeStep_s
world_RigidBody_Vel_y = np.diff(world_mocapRigidBody_Pos[:,1])/timeStep_s
world_RigidBody_Vel_z = np.diff(world_mocapRigidBody_Pos[:,2])/timeStep_s
world_RigidBody_Vel_x = np.insert(world_RigidBody_Vel_x, 0, 0.0, axis=0)
world_RigidBody_Vel_y = np.insert(world_RigidBody_Vel_y, 0, 0.0, axis=0)
world_RigidBody_Vel_z = np.insert(world_RigidBody_Vel_z, 0, 0.0, axis=0)
world_RigidBody_Vel = np.stack((world_RigidBody_Vel_x, world_RigidBody_Vel_y, world_RigidBody_Vel_z), axis = 1)

# We compute the velocity estimated by the Observer in the world
world_ObserverLimb_Vel_x = np.diff(world_ObserverLimb_Pos[:,0])/timeStep_s
world_ObserverLimb_Vel_y = np.diff(world_ObserverLimb_Pos[:,1])/timeStep_s
world_ObserverLimb_Vel_z = np.diff(world_ObserverLimb_Pos[:,2])/timeStep_s
world_ObserverLimb_Vel_x = np.insert(world_ObserverLimb_Vel_x, 0, 0.0, axis=0)
world_ObserverLimb_Vel_y = np.insert(world_ObserverLimb_Vel_y, 0, 0.0, axis=0)
world_ObserverLimb_Vel_z = np.insert(world_ObserverLimb_Vel_z, 0, 0.0, axis=0)
world_ObserverLimb_Vel = np.stack((world_ObserverLimb_Vel_x, world_ObserverLimb_Vel_y, world_ObserverLimb_Vel_z), axis = 1)


# Now we get the local linear velocity
world_mocapLimb_LocVel = world_mocapLimb_Ori_R.apply(world_mocapLimb_Vel, inverse=True)
world_RigidBody_LocVel = world_RigidBody_Ori_R.apply(world_RigidBody_Vel, inverse=True)
world_ObserverLimb_LocVel = world_ObserverLimb_Ori_R.apply(world_ObserverLimb_Vel, inverse=True)


if(displayLogs):
    # Plot of the resulting poses
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_LocVel[:,0], mode='lines', name='world_mocapLimb_LocVel_x'))
    fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_LocVel[:,1], mode='lines', name='world_mocapLimb_LocVel_y'))
    fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_mocapLimb_LocVel[:,2], mode='lines', name='world_mocapLimb_LocVel_z'))
    fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_LocVel[:,0], mode='lines', name='world_RigidBody_LocalLinVel_x'))
    fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_LocVel[:,1], mode='lines', name='world_RigidBody_LocalLinVel_y'))
    fig2.add_trace(go.Scatter(x=mocapData["Time(Seconds)"], y=world_RigidBody_LocVel[:,2], mode='lines', name='world_RigidBody_LocalLinVel_z'))
    fig2.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_LocVel[:,0], mode='lines', name='world_ObserverLimb_LocVel_x'))
    fig2.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_LocVel[:,1], mode='lines', name='world_ObserverLimb_LocVel_y'))
    fig2.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_LocVel[:,2], mode='lines', name='world_ObserverLimb_LocVel_z'))

    fig2.update_layout(title=f"{scriptName}: Local linear velocity before alignment")
    # Show the plotly figures
    fig2.show()




###############################  Cross correlation  ###############################

def match_array_length(arr, desired_length):
    len_diff = len(arr) - desired_length
    # If the array to resize is bigger, we delete the data at the end.
    if(len_diff > 0):
        result = arr[:desired_length, :]
    else:
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
    if(len(data1) > len(data2)):
        data2 = match_array_length(data2, len(data1))
    else:
        data1 = match_array_length(data1, len(data2))

    if(displayLogs):
        figInit = go.Figure()

        time1 = np.arange(0, len(data1), step=1)
        time2 = np.arange(0, len(data2), step=1)
        figInit.add_trace(go.Scatter(x=time1, y=data1[:,0], mode='lines', name=f'{data1_name}_1'))
        figInit.add_trace(go.Scatter(x=time1, y=data1[:,1], mode='lines', name=f'{data1_name}_2'))
        figInit.add_trace(go.Scatter(x=time1, y=data1[:,2], mode='lines', name=f'{data1_name}_3'))

        figInit.add_trace(go.Scatter(x=time2, y=data2[:,0], mode='lines', name=f'{data2_name}_1'))
        figInit.add_trace(go.Scatter(x=time2, y=data2[:,1], mode='lines', name=f'{data2_name}_2'))
        figInit.add_trace(go.Scatter(x=time2, y=data2[:,2], mode='lines', name=f'{data2_name}_3'))

        figInit.update_layout(title=f"{scriptName}: Data before alignment")
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
        print(f"The mocap data is {shift} iterations ahead of the observer.")
    if(shift < 0):
        print(f"The mocap data is {shift} iterations behind the observer.")

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

    fig.update_layout(title=f"{scriptName}: Previsualization of data after alignment")

    fig.write_image(f'{path_to_project}/output_data/scriptResults/crossCorrelation/temporally_aligned_loc_linVel.png')

    if(displayLogs):
        fig.show()

    return data2, shift


world_mocapLimb_LocVel, shift = realignData(world_ObserverLimb_LocVel, world_mocapLimb_LocVel, "world_mocapLimb_LocVel", "world_ObserverLimb_LocVel")


# Version which receives the shift to apply as an input
def realignData(data_to_shift,  shift):
    print(f"The mocap data will be shifted by {shift} indexes.")

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
        data_shifted = data_shifted.iloc[:len(data_shifted) - diffIndexFinal]

    overlapIndex = []
    for i in range(len(data_shifted)):
        if(i > diffIndexInit and i < len(data_shifted) + diffIndexFinal):
            overlapIndex.append(1)
        else:
            overlapIndex.append(0)

    if(displayLogs):
        figAlignedIndexes = go.Figure()

        figAlignedIndexes.add_trace(go.Scatter(x=data_shifted.index, y=data_shifted['world_MocapLimb_Pos_x'], mode='lines', name='world_MocapLimb_Pos_x'))
        figAlignedIndexes.add_trace(go.Scatter(x=data_shifted.index, y=data_shifted['world_MocapLimb_Pos_y'], mode='lines', name='world_MocapLimb_Pos_y'))
        figAlignedIndexes.add_trace(go.Scatter(x=data_shifted.index, y=data_shifted['world_MocapLimb_Pos_z'], mode='lines', name='world_MocapLimb_Pos_z'))
        figAlignedIndexes.add_trace(go.Scatter(x=observer_data.index, y=world_ObserverLimb_Pos[:,0], mode='lines', name='world_ObserverLimb_Pos_x'))
        figAlignedIndexes.add_trace(go.Scatter(x=observer_data.index, y=world_ObserverLimb_Pos[:,1], mode='lines', name='world_ObserverLimb_Pos_y'))
        figAlignedIndexes.add_trace(go.Scatter(x=observer_data.index, y=world_ObserverLimb_Pos[:,2], mode='lines', name='world_ObserverLimb_Pos_z'))

        figAlignedIndexes.update_layout(title=f"{scriptName}: Position after alignment")

        figAlignedIndexes.show()

    data_shifted['Time(Seconds)'] = observer_data["t"]
    data_shifted['overlapTime'] = overlapIndex

    return data_shifted

realignedMocapData = realignData(mocapData, shift)

###############################  Shifted poses retrieval  ###############################

# Extracting the poses related to the realignedMocapData
world_mocapRigidBody_Pos = np.array([realignedMocapData['RigidBody001_tX'], realignedMocapData['RigidBody001_tY'], realignedMocapData['RigidBody001_tZ']]).T
world_RigidBody_Ori_R = R.from_quat(realignedMocapData[["RigidBody001_qX", "RigidBody001_qY", "RigidBody001_qZ", "RigidBody001_qW"]].values)

world_mocapLimb_Pos = np.array([realignedMocapData['world_MocapLimb_Pos_x'], realignedMocapData['world_MocapLimb_Pos_y'], realignedMocapData['world_MocapLimb_Pos_z']]).T
world_mocapLimb_Ori_R = R.from_quat(realignedMocapData[["world_MocapLimb_Ori_qx", "world_MocapLimb_Ori_qy", "world_MocapLimb_Ori_qz", "world_MocapLimb_Ori_qw"]].values)


###############################  Visualization of the extracted poses  ###############################


if(displayLogs):
    # Plot of the resulting positions
    figPositions_realigned = go.Figure()

    figPositions_realigned.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Pos[:,0], mode='lines', name='world_Observer_Body_pos_x_realigned'))
    figPositions_realigned.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Pos[:,1], mode='lines', name='world_Observer_Body_pos_y_realigned'))
    figPositions_realigned.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Pos[:,2], mode='lines', name='world_Observer_Body_pos_z_realigned'))

    figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Pos[:,0], mode='lines', name='world_mocapLimb_Pos_x_realigned'))
    figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Pos[:,1], mode='lines', name='world_mocapLimb_Pos_y_realigned'))
    figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Pos[:,2], mode='lines', name='world_mocapLimb_Pos_z_realigned'))

    figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=realignedMocapData["RigidBody001_tX"], mode='lines', name='world_RigidBody_pos_x_realigned'))
    figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=realignedMocapData["RigidBody001_tY"], mode='lines', name='world_RigidBody_pos_y_realigned'))
    figPositions_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=realignedMocapData["RigidBody001_tZ"], mode='lines', name='world_RigidBody_pos_z_realigned'))

    figPositions_realigned.update_layout(title=f"{scriptName}: Realigned positions")

    # Show the plotly figures
    figPositions_realigned.show()


    # Plot of the resulting orientations
    figOrientations_realigned = go.Figure()

    #world_ObserverLimb_Ori_euler = world_ObserverLimb_Ori_R.as_euler("xyz")
    world_mocapLimb_Ori_euler = world_mocapLimb_Ori_R.as_euler("xyz")
    world_RigidBody_Ori_euler = world_RigidBody_Ori_R.as_euler("xyz")

    #world_ObserverLimb_Ori_euler_continuous = continuous_euler(world_ObserverLimb_Ori_euler)
    world_mocapLimb_Ori_euler_continuous = continuous_euler(world_mocapLimb_Ori_euler)
    world_RigidBody_Ori_euler_continuous = continuous_euler(world_RigidBody_Ori_euler)

    figOrientations_realigned.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_euler_continuous[:,0], mode='lines', name='world_Observer_Body_ori_roll_realigned'))
    figOrientations_realigned.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_euler_continuous[:,1], mode='lines', name='world_Observer_Body_ori_pitch_realigned'))
    figOrientations_realigned.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_euler_continuous[:,2], mode='lines', name='world_Observer_Body_ori_yaw_realigned'))

    figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Ori_euler_continuous[:,0], mode='lines', name='world_mocapLimb_Ori_roll_realigned'))
    figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Ori_euler_continuous[:,1], mode='lines', name='world_mocapLimb_Ori_pitch_realigned'))
    figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Ori_euler_continuous[:,2], mode='lines', name='world_mocapLimb_Ori_yaw_realigned'))

    figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,0], mode='lines', name='world_RigidBody_ori_roll_realigned'))
    figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,1], mode='lines', name='world_RigidBody_ori_pitch_realigned'))
    figOrientations_realigned.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_euler_continuous[:,2], mode='lines', name='world_RigidBody_ori_yaw_realigned'))

    figOrientations_realigned.update_layout(title=f"{scriptName}: Realigned orientations")

    # Show the plotly figures
    figOrientations_realigned.show()






#####################  Orientation and position difference wrt the initial frame  #####################



world_mocapLimb_pos_transfo = world_mocapLimb_Ori_R[0].apply(world_mocapLimb_Pos - world_mocapLimb_Pos[0], inverse=True)
world_RigidBody_pos_transfo = world_RigidBody_Ori_R[0].apply(world_mocapRigidBody_Pos - world_mocapRigidBody_Pos[0], inverse=True)
world_ObserverLimb_pos_transfo = world_ObserverLimb_Ori_R[0].apply(world_ObserverLimb_Pos - world_ObserverLimb_Pos[0], inverse=True)

world_mocapLimb_Ori_R_transfo = world_mocapLimb_Ori_R[0].inv() * world_mocapLimb_Ori_R
world_RigidBody_Ori_R_transfo = world_RigidBody_Ori_R[0].inv() * world_RigidBody_Ori_R
world_ObserverLimb_Ori_R_transfo = world_ObserverLimb_Ori_R[0].inv() * world_ObserverLimb_Ori_R

world_mocapLimb_Ori_transfo_euler = world_mocapLimb_Ori_R_transfo.as_euler("xyz")
world_RigidBody_Ori_transfo_euler = world_RigidBody_Ori_R_transfo.as_euler("xyz")
world_ObserverLimb_Ori_transfo_euler = world_ObserverLimb_Ori_R_transfo.as_euler("xyz")

world_mocapLimb_Ori_transfo_euler_continuous = continuous_euler(world_mocapLimb_Ori_transfo_euler)
world_RigidBody_Ori_transfo_euler_continuous = continuous_euler(world_RigidBody_Ori_transfo_euler)
world_ObserverLimb_Ori_transfo_euler_continuous = continuous_euler(world_ObserverLimb_Ori_transfo_euler)

figTransfo = go.Figure()

figTransfo.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_mocapLimb_Ori_transfo_roll'))
figTransfo.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_mocapLimb_Ori_transfo_pitch'))
figTransfo.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_mocapLimb_Ori_transfo_yaw'))

figTransfo.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_RigidBody_Ori_transfo_roll'))
figTransfo.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_RigidBody_Ori_transfo_pitch'))
figTransfo.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_RigidBody_Ori_transfo_yaw'))

figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_ObserverLimb_Ori_transfo_roll'))
figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_ObserverLimb_Ori_transfo_pitch'))
figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_ObserverLimb_Ori_transfo_yaw'))

figTransfo.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_pos_transfo[:,0], mode='lines', name='world_RigidBody_pos_transfo_x'))
figTransfo.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_pos_transfo[:,1], mode='lines', name='world_RigidBody_pos_transfo_y'))
figTransfo.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_RigidBody_pos_transfo[:,2], mode='lines', name='world_RigidBody_pos_transfo_z'))

figTransfo.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_pos_transfo[:,0], mode='lines', name='world_mocapLimb_pos_transfo_x'))
figTransfo.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_pos_transfo[:,1], mode='lines', name='world_mocapLimb_pos_transfo_y'))
figTransfo.add_trace(go.Scatter(x=realignedMocapData["Time(Seconds)"], y=world_mocapLimb_pos_transfo[:,2], mode='lines', name='world_mocapLimb_pos_transfo_z'))

figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_pos_transfo[:,0], mode='lines', name='world_ObserverLimb_pos_transfo_x'))
figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_pos_transfo[:,1], mode='lines', name='world_ObserverLimb_pos_transfo_y'))
figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_pos_transfo[:,2], mode='lines', name='world_ObserverLimb_pos_transfo_z'))

figTransfo.update_layout(title=f"{scriptName}: Realigned transformations")

figTransfo.write_image(f'{path_to_project}/output_data/scriptResults/crossCorrelation/temporally_aligned_ori_transfo.png')

if(displayLogs):
    # Show the plotly figures
    figTransfo.show()



world_mocapLimb_Ori_quat = world_mocapLimb_Ori_R.as_quat()
output_df = pd.DataFrame({'t': observer_data['t'], 'worldMocapLimbPos_x': world_mocapLimb_Pos[:,0], 'worldMocapLimbPos_y': world_mocapLimb_Pos[:,1], 'worldMocapLimbPos_z': world_mocapLimb_Pos[:,2], 'worldMocapLimbOri_qx': world_mocapLimb_Ori_quat[:,0], 'worldMocapLimbOri_qy': world_mocapLimb_Ori_quat[:,1], 'worldMocapLimbOri_qz': world_mocapLimb_Ori_quat[:,2], 'worldMocapLimbOri_qw': world_mocapLimb_Ori_quat[:,3], 'overlapTime': realignedMocapData['overlapTime']})


# Save the DataFrame to a new CSV file
if(len(sys.argv) > 3):
    save_csv = sys.argv[3].lower()
else:
    save_csv = input("Do you want to save the data as a CSV file? (y/n): ")
    save_csv = save_csv.lower()


if save_csv == 'y':
    output_df.to_csv(output_csv_file_path, index=False, sep=';')
    print("Output CSV file has been saved to ", output_csv_file_path)
else:
    print("Data not saved.")

