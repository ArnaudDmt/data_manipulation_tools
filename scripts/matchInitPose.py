import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go




###############################  Main variables initialization  ###############################


output_csv_file_path = '../output_data/resultMocapLimbData.csv'

# Load the CSV files into pandas dataframes
observer_data = pd.read_csv('../output_data/lightData.csv')
mocapData = pd.read_csv('../output_data/realignedMocapLimbData.csv', delimiter=',')
averageInterval = 10
displayLogs = True
matchTime = 0



###############################  User inputs  ###############################


if(len(sys.argv) > 1):
    matchTime = int(sys.argv[1])
    if(len(sys.argv) > 2):
        displayLogs = sys.argv[2].lower() == 'true'
else:
    matchTime = float(input("When do you want the mocap pose to match the observer's one? "))





###############################  Function definitions  ###############################


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

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

world_mocapLimb_Pos = np.array([mocapData['worldMocapLimbPos_x'], mocapData['worldMocapLimbPos_y'], mocapData['worldMocapLimbPos_z']]).T
world_mocapLimb_Ori_R = R.from_quat(mocapData[["worldMocapLimbOri_qx", "worldMocapLimbOri_qy", "worldMocapLimbOri_qz", "worldMocapLimbOri_qw"]].values)

# Extracting the poses coming from mc_rtc
world_VanyteLimb_Pos = np.array([observer_data['Observers_MainObserverPipeline_MCVanytEstimator_mocap_worldBodyKine_position_x'], observer_data['Observers_MainObserverPipeline_MCVanytEstimator_mocap_worldBodyKine_position_y'], observer_data['Observers_MainObserverPipeline_MCVanytEstimator_mocap_worldBodyKine_position_z']]).T
world_VanyteLimb_Ori_R = R.from_quat(observer_data[["Observers_MainObserverPipeline_MCVanytEstimator_mocap_worldBodyKine_ori_x", "Observers_MainObserverPipeline_MCVanytEstimator_mocap_worldBodyKine_ori_y", "Observers_MainObserverPipeline_MCVanytEstimator_mocap_worldBodyKine_ori_z", "Observers_MainObserverPipeline_MCVanytEstimator_mocap_worldBodyKine_ori_w"]].values)
# We get the inverse of the orientation as the inverse quaternion was stored
world_VanyteLimb_Ori_R = world_VanyteLimb_Ori_R.inv()



#####################  Orientation and position difference wrt the initial frame  #####################

world_mocapLimb_pos_transfo = world_mocapLimb_Ori_R[0].apply(world_mocapLimb_Pos - world_mocapLimb_Pos[0], inverse=True)
world_VanyteLimb_pos_transfo = world_VanyteLimb_Ori_R[0].apply(world_VanyteLimb_Pos - world_VanyteLimb_Pos[0], inverse=True)

world_mocapLimb_Ori_R_transfo = world_mocapLimb_Ori_R * world_mocapLimb_Ori_R[0].inv()
world_VanyteLimb_Ori_R_transfo = world_VanyteLimb_Ori_R * world_VanyteLimb_Ori_R[0].inv()

world_mocapLimb_Ori_transfo_euler = world_mocapLimb_Ori_R_transfo.as_euler("xyz")
world_VanyteLimb_Ori_transfo_euler = world_VanyteLimb_Ori_R_transfo.as_euler("xyz")

world_mocapLimb_Ori_transfo_euler_continuous = continuous_euler(world_mocapLimb_Ori_transfo_euler)
world_VanyteLimb_Ori_transfo_euler_continuous = continuous_euler(world_VanyteLimb_Ori_transfo_euler)


world_mocapLimb_Ori_euler = world_mocapLimb_Ori_R.as_euler("xyz")
world_VanyteLimb_Ori_euler = world_VanyteLimb_Ori_R.as_euler("xyz")

world_mocapLimb_Ori_euler_continuous = continuous_euler(world_mocapLimb_Ori_euler)
world_VanyteLimb_Ori_euler_continuous = continuous_euler(world_VanyteLimb_Ori_euler)


if(displayLogs):
    figInitPose = go.Figure()

    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_mocapLimb_Ori_euler_continuous[:,0], mode='lines', name='world_mocapLimb_Ori_roll'))
    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_mocapLimb_Ori_euler_continuous[:,1], mode='lines', name='world_mocapLimb_Ori_pitch'))
    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_mocapLimb_Ori_euler_continuous[:,2], mode='lines', name='world_mocapLimb_Ori_yaw'))

    figInitPose.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_Ori_euler_continuous[:,0], mode='lines', name='world_VanyteLimb_Ori_roll'))
    figInitPose.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_Ori_euler_continuous[:,1], mode='lines', name='world_VanyteLimb_Ori_pitch'))
    figInitPose.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_Ori_euler_continuous[:,2], mode='lines', name='world_VanyteLimb_Ori_yaw'))


    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_mocapLimb_Pos[:,0], mode='lines', name='world_mocapLimb_Pos_x'))
    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_mocapLimb_Pos[:,1], mode='lines', name='world_mocapLimb_Pos_y'))
    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_mocapLimb_Pos[:,2], mode='lines', name='world_mocapLimb_Pos_z'))

    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_VanyteLimb_Pos[:,0], mode='lines', name='world_VanyteLimb_Pos_x'))
    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_VanyteLimb_Pos[:,1], mode='lines', name='world_VanyteLimb_Pos_y'))
    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_VanyteLimb_Pos[:,2], mode='lines', name='world_VanyteLimb_Pos_z'))

    figInitPose.update_layout(title="Resulting pose")

    # Show the plotly figure
    figInitPose.show()


    figTransfoInit = go.Figure()

    figTransfoInit.add_trace(go.Scatter(x=mocapData["t"], y=world_mocapLimb_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_mocapLimb_Ori_transfo_roll'))
    figTransfoInit.add_trace(go.Scatter(x=mocapData["t"], y=world_mocapLimb_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_mocapLimb_Ori_transfo_pitch'))
    figTransfoInit.add_trace(go.Scatter(x=mocapData["t"], y=world_mocapLimb_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_mocapLimb_Ori_transfo_yaw'))

    figTransfoInit.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_VanyteLimb_Ori_transfo_roll'))
    figTransfoInit.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_VanyteLimb_Ori_transfo_pitch'))
    figTransfoInit.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_VanyteLimb_Ori_transfo_yaw'))


    figTransfoInit.add_trace(go.Scatter(x=mocapData["t"], y=world_mocapLimb_pos_transfo[:,0], mode='lines', name='world_mocapLimb_pos_transfo_x'))
    figTransfoInit.add_trace(go.Scatter(x=mocapData["t"], y=world_mocapLimb_pos_transfo[:,1], mode='lines', name='world_mocapLimb_pos_transfo_y'))
    figTransfoInit.add_trace(go.Scatter(x=mocapData["t"], y=world_mocapLimb_pos_transfo[:,2], mode='lines', name='world_mocapLimb_pos_transfo_z'))

    figTransfoInit.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_pos_transfo[:,0], mode='lines', name='world_VanyteLimb_pos_transfo_x'))
    figTransfoInit.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_pos_transfo[:,1], mode='lines', name='world_VanyteLimb_pos_transfo_y'))
    figTransfoInit.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_pos_transfo[:,2], mode='lines', name='world_VanyteLimb_pos_transfo_z'))

    figTransfoInit.update_layout(title="Initial transformations")

    # Show the plotly figures
    figTransfoInit.show()


###############################  Average around matching point  ###############################


# Find the index in the pandas dataframe that corresponds to the input time
matchIndex = mocapData[mocapData['t'] == matchTime].index[0]

world_mocapLimb_Pos_average_atMatch = np.mean(world_mocapLimb_Pos[matchIndex - averageInterval:matchIndex + averageInterval], axis = 0)
world_VanyteLimb_Pos_average_atMatch = np.mean(world_VanyteLimb_Pos[matchIndex - averageInterval:matchIndex + averageInterval], axis = 0)

world_mocapLimb_Ori_Quat = world_mocapLimb_Ori_R.as_quat()
world_VanyteLimb_Ori_quat = world_VanyteLimb_Ori_R.as_quat()

world_mocapLimb_Ori_Quat_average_atMatch = np.mean(world_mocapLimb_Ori_Quat[matchIndex - averageInterval:matchIndex + averageInterval], axis = 0)
world_VanyteLimb_Ori_Quat_average_atMatch = np.mean(world_VanyteLimb_Ori_quat[matchIndex - averageInterval:matchIndex + averageInterval], axis = 0)


world_mocapLimb_Ori_R_average_atMatch = R.from_quat(normalize(world_mocapLimb_Ori_Quat_average_atMatch))
world_VanyteLimb_Ori_R_average_atMatch = R.from_quat(normalize(world_VanyteLimb_Ori_Quat_average_atMatch))


mocapLimb_world_Ori_R_average_atMatch = world_mocapLimb_Ori_R_average_atMatch.inv()

mocapVanyte_Ori_R = mocapLimb_world_Ori_R_average_atMatch * world_VanyteLimb_Ori_R_average_atMatch
new_world_mocapLimb_Ori_R = world_mocapLimb_Ori_R * mocapVanyte_Ori_R


# Allows the position of the mocap to match with the one of the vanyte at the desired time, while preserving the transformation between the current frame and the inital one for each iteration
new_world_mocapLimb_Pos = (new_world_mocapLimb_Ori_R[0] * world_mocapLimb_Ori_R[0].inv()).apply(world_mocapLimb_Pos - world_mocapLimb_Pos_average_atMatch) + world_VanyteLimb_Pos_average_atMatch


if(displayLogs):
    new_world_mocapLimb_Ori_euler = new_world_mocapLimb_Ori_R.as_euler("xyz")
    world_VanyteLimb_Ori_euler = world_VanyteLimb_Ori_R.as_euler("xyz")

    new_world_mocapLimb_Ori_euler_continuous = continuous_euler(new_world_mocapLimb_Ori_euler)
    world_VanyteLimb_Ori_euler_continuous = continuous_euler(world_VanyteLimb_Ori_euler)

    figNewPose = go.Figure()

    figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=new_world_mocapLimb_Ori_euler_continuous[:,0], mode='lines', name='world_mocapLimb_Ori_roll'))
    figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=new_world_mocapLimb_Ori_euler_continuous[:,1], mode='lines', name='world_mocapLimb_Ori_pitch'))
    figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=new_world_mocapLimb_Ori_euler_continuous[:,2], mode='lines', name='world_mocapLimb_Ori_yaw'))

    figNewPose.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_Ori_euler_continuous[:,0], mode='lines', name='world_VanyteLimb_Ori_roll'))
    figNewPose.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_Ori_euler_continuous[:,1], mode='lines', name='world_VanyteLimb_Ori_pitch'))
    figNewPose.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_Ori_euler_continuous[:,2], mode='lines', name='world_VanyteLimb_Ori_yaw'))


    figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=new_world_mocapLimb_Pos[:,0], mode='lines', name='world_mocapLimb_Pos_x'))
    figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=new_world_mocapLimb_Pos[:,1], mode='lines', name='world_mocapLimb_Pos_y'))
    figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=new_world_mocapLimb_Pos[:,2], mode='lines', name='world_mocapLimb_Pos_z'))

    figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=world_VanyteLimb_Pos[:,0], mode='lines', name='world_VanyteLimb_Pos_x'))
    figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=world_VanyteLimb_Pos[:,1], mode='lines', name='world_VanyteLimb_Pos_y'))
    figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=world_VanyteLimb_Pos[:,2], mode='lines', name='world_VanyteLimb_Pos_z'))

    figNewPose.update_layout(title="Resulting pose")

    # Show the plotly figure
    figNewPose.show()



#####################  Orientation and position difference wrt the initial frame  #####################

if(displayLogs):
    new_world_mocapLimb_pos_transfo = new_world_mocapLimb_Ori_R[0].apply(new_world_mocapLimb_Pos - new_world_mocapLimb_Pos[0], inverse=True)
    world_VanyteLimb_pos_transfo = world_VanyteLimb_Ori_R[0].apply(world_VanyteLimb_Pos - world_VanyteLimb_Pos[0], inverse=True)

    new_world_mocapLimb_Ori_R_transfo = new_world_mocapLimb_Ori_R * new_world_mocapLimb_Ori_R[0].inv()
    world_VanyteLimb_Ori_R_transfo = world_VanyteLimb_Ori_R * world_VanyteLimb_Ori_R[0].inv()

    new_world_mocapLimb_Ori_transfo_euler = new_world_mocapLimb_Ori_R_transfo.as_euler("xyz")
    world_VanyteLimb_Ori_transfo_euler = world_VanyteLimb_Ori_R_transfo.as_euler("xyz")

    new_world_mocapLimb_Ori_transfo_euler_continuous = continuous_euler(new_world_mocapLimb_Ori_transfo_euler)
    world_VanyteLimb_Ori_transfo_euler_continuous = continuous_euler(world_VanyteLimb_Ori_transfo_euler)

    figTransfo = go.Figure()

    figTransfo.add_trace(go.Scatter(x=mocapData["t"], y=new_world_mocapLimb_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_mocapLimb_Ori_transfo_roll'))
    figTransfo.add_trace(go.Scatter(x=mocapData["t"], y=new_world_mocapLimb_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_mocapLimb_Ori_transfo_pitch'))
    figTransfo.add_trace(go.Scatter(x=mocapData["t"], y=new_world_mocapLimb_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_mocapLimb_Ori_transfo_yaw'))

    figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_VanyteLimb_Ori_transfo_roll'))
    figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_VanyteLimb_Ori_transfo_pitch'))
    figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_VanyteLimb_Ori_transfo_yaw'))


    figTransfo.add_trace(go.Scatter(x=mocapData["t"], y=new_world_mocapLimb_pos_transfo[:,0], mode='lines', name='world_mocapLimb_pos_transfo_x'))
    figTransfo.add_trace(go.Scatter(x=mocapData["t"], y=new_world_mocapLimb_pos_transfo[:,1], mode='lines', name='world_mocapLimb_pos_transfo_y'))
    figTransfo.add_trace(go.Scatter(x=mocapData["t"], y=new_world_mocapLimb_pos_transfo[:,2], mode='lines', name='world_mocapLimb_pos_transfo_z'))

    figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_pos_transfo[:,0], mode='lines', name='world_VanyteLimb_pos_transfo_x'))
    figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_pos_transfo[:,1], mode='lines', name='world_VanyteLimb_pos_transfo_y'))
    figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_VanyteLimb_pos_transfo[:,2], mode='lines', name='world_VanyteLimb_pos_transfo_z'))

    figTransfo.update_layout(title="Transformations")

    # Show the plotly figures
    figTransfo.show()


new_world_mocapLimb_Ori_quat = new_world_mocapLimb_Ori_R.as_quat()

mocapData['worldMocapLimbPos_x'] = new_world_mocapLimb_Pos[:,0]
mocapData['worldMocapLimbPos_y'] = new_world_mocapLimb_Pos[:,1]
mocapData['worldMocapLimbPos_z'] = new_world_mocapLimb_Pos[:,2]
mocapData['worldMocapLimbOri_qx'] = new_world_mocapLimb_Ori_quat[:,0]
mocapData['worldMocapLimbOri_qy'] = new_world_mocapLimb_Ori_quat[:,1]
mocapData['worldMocapLimbOri_qz'] = new_world_mocapLimb_Ori_quat[:,2]
mocapData['worldMocapLimbOri_qw'] = new_world_mocapLimb_Ori_quat[:,3]


# Save the DataFrame to a new CSV file
if(len(sys.argv) > 3):
    save_csv = sys.argv[3].lower()
else:
    save_csv = input("Do you want to save the data as a CSV file? (y/n): ")
    save_csv = save_csv.lower()


if save_csv == 'y':
    mocapData.to_csv(output_csv_file_path, index=False)
    print("Output CSV file has been saved to ", output_csv_file_path)
else:
    print("Data not saved.")


