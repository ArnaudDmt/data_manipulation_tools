import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go




###############################  Main variables initialization  ###############################

path_to_project = ".."

averageInterval = 10
displayLogs = True
matchTime = 0
scriptName = "matchInitPose"


###############################  User inputs  ###############################


if(len(sys.argv) > 1):
    matchTime = int(sys.argv[1])
    if(len(sys.argv) > 2):
        displayLogs = sys.argv[2].lower() == 'true'
    if(len(sys.argv) > 4):
        path_to_project = sys.argv[4]
else:
    matchTime = float(input("When do you want the mocap pose to match the observer's one? "))



output_csv_file_path = f'{path_to_project}/output_data/resultMocapLimbData.csv'
# Load the CSV files into pandas dataframes
observer_data = pd.read_csv(f'{path_to_project}/output_data/lightData.csv', delimiter=';')
mocapData = pd.read_csv(f'{path_to_project}/output_data/realignedMocapLimbData.csv', delimiter=';')



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

def get_invariant_orthogonal_vector(Rhat, Rtez):
    epsilon = 2.2204460492503131e-16
    Rhat_Rtez = np.dot(Rhat, Rtez)
    if np.all(np.abs(Rhat_Rtez[:2]) < epsilon):
        return np.array([1, 0, 0])
    else:
        return np.array([Rhat_Rtez[1], -Rhat_Rtez[0], 0])

def merge_tilt_with_yaw_axis_agnostic(Rtez, R2):
    ez = np.array([0, 0, 1])
    v1 = Rtez

    m = get_invariant_orthogonal_vector(R2, Rtez)
    m = m / np.linalg.norm(m)

    ml = np.dot(R2.T, m)

    R_temp1 = np.column_stack((np.cross(m, ez), m, ez))

    R_temp2 = np.vstack((np.cross(ml, v1).T, ml.T, v1.T))

    return np.dot(R_temp1, R_temp2)

###############################  Poses retrieval  ###############################

# Extracting the poses related to the mocap

world_MocapLimb_Pos = np.array([mocapData['worldMocapLimbPos_x'], mocapData['worldMocapLimbPos_y'], mocapData['worldMocapLimbPos_z']]).T
world_MocapLimb_Ori_R = R.from_quat(mocapData[["worldMocapLimbOri_qx", "worldMocapLimbOri_qy", "worldMocapLimbOri_qz", "worldMocapLimbOri_qw"]].values)

# Extracting the poses coming from mc_rtc
world_ObserverLimb_Pos = np.array([observer_data['MocapAligner_worldBodyKine_position_x'], observer_data['MocapAligner_worldBodyKine_position_y'], observer_data['MocapAligner_worldBodyKine_position_z']]).T
world_ObserverLimb_Ori_R = R.from_quat(observer_data[["MocapAligner_worldBodyKine_ori_x", "MocapAligner_worldBodyKine_ori_y", "MocapAligner_worldBodyKine_ori_z", "MocapAligner_worldBodyKine_ori_w"]].values)
# We get the inverse of the orientation as the inverse quaternion was stored
world_ObserverLimb_Ori_R = world_ObserverLimb_Ori_R.inv()


overlapIndex = mocapData['overlapTime']


#####################  Orientation and position difference wrt the initial frame  #####################

world_MocapLimb_pos_transfo = world_MocapLimb_Ori_R[0].apply(world_MocapLimb_Pos - world_MocapLimb_Pos[0], inverse=True)
world_ObserverLimb_pos_transfo = world_ObserverLimb_Ori_R[0].apply(world_ObserverLimb_Pos - world_ObserverLimb_Pos[0], inverse=True)

world_MocapLimb_Ori_R_transfo = world_MocapLimb_Ori_R[0].inv() * world_MocapLimb_Ori_R 
world_ObserverLimb_Ori_R_transfo = world_ObserverLimb_Ori_R[0].inv() * world_ObserverLimb_Ori_R 

world_MocapLimb_Ori_transfo_euler = world_MocapLimb_Ori_R_transfo.as_euler("xyz")
world_ObserverLimb_Ori_transfo_euler = world_ObserverLimb_Ori_R_transfo.as_euler("xyz")

world_MocapLimb_Ori_transfo_euler_continuous = continuous_euler(world_MocapLimb_Ori_transfo_euler)
world_ObserverLimb_Ori_transfo_euler_continuous = continuous_euler(world_ObserverLimb_Ori_transfo_euler)


world_MocapLimb_Ori_euler = world_MocapLimb_Ori_R.as_euler("xyz")
world_ObserverLimb_Ori_euler = world_ObserverLimb_Ori_R.as_euler("xyz")

world_MocapLimb_Ori_euler_continuous = continuous_euler(world_MocapLimb_Ori_euler)
world_ObserverLimb_Ori_euler_continuous = continuous_euler(world_ObserverLimb_Ori_euler)


if(displayLogs):
    figInitPose = go.Figure()

    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_MocapLimb_Ori_euler_continuous[:,0], mode='lines', name='world_MocapLimb_Ori_roll'))
    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_MocapLimb_Ori_euler_continuous[:,1], mode='lines', name='world_MocapLimb_Ori_pitch'))
    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_MocapLimb_Ori_euler_continuous[:,2], mode='lines', name='world_MocapLimb_Ori_yaw'))

    figInitPose.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_euler_continuous[:,0], mode='lines', name='world_ObserverLimb_Ori_roll'))
    figInitPose.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_euler_continuous[:,1], mode='lines', name='world_ObserverLimb_Ori_pitch'))
    figInitPose.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_euler_continuous[:,2], mode='lines', name='world_ObserverLimb_Ori_yaw'))


    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_MocapLimb_Pos[:,0], mode='lines', name='world_MocapLimb_Pos_x'))
    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_MocapLimb_Pos[:,1], mode='lines', name='world_MocapLimb_Pos_y'))
    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_MocapLimb_Pos[:,2], mode='lines', name='world_MocapLimb_Pos_z'))

    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_ObserverLimb_Pos[:,0], mode='lines', name='world_ObserverLimb_Pos_x'))
    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_ObserverLimb_Pos[:,1], mode='lines', name='world_ObserverLimb_Pos_y'))
    figInitPose.add_trace(go.Scatter(x=mocapData["t"], y=world_ObserverLimb_Pos[:,2], mode='lines', name='world_ObserverLimb_Pos_z'))

    figInitPose.update_layout(title=f"{scriptName}: Poses before matching")


    # Show the plotly figure
    figInitPose.show()


    figTransfoInit = go.Figure()

    figTransfoInit.add_trace(go.Scatter(x=mocapData["t"], y=world_MocapLimb_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_MocapLimb_Ori_transfo_roll'))
    figTransfoInit.add_trace(go.Scatter(x=mocapData["t"], y=world_MocapLimb_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_MocapLimb_Ori_transfo_pitch'))
    figTransfoInit.add_trace(go.Scatter(x=mocapData["t"], y=world_MocapLimb_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_MocapLimb_Ori_transfo_yaw'))

    figTransfoInit.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_ObserverLimb_Ori_transfo_roll'))
    figTransfoInit.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_ObserverLimb_Ori_transfo_pitch'))
    figTransfoInit.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_ObserverLimb_Ori_transfo_yaw'))


    figTransfoInit.add_trace(go.Scatter(x=mocapData["t"], y=world_MocapLimb_pos_transfo[:,0], mode='lines', name='world_MocapLimb_pos_transfo_x'))
    figTransfoInit.add_trace(go.Scatter(x=mocapData["t"], y=world_MocapLimb_pos_transfo[:,1], mode='lines', name='world_MocapLimb_pos_transfo_y'))
    figTransfoInit.add_trace(go.Scatter(x=mocapData["t"], y=world_MocapLimb_pos_transfo[:,2], mode='lines', name='world_MocapLimb_pos_transfo_z'))

    figTransfoInit.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_pos_transfo[:,0], mode='lines', name='world_ObserverLimb_pos_transfo_x'))
    figTransfoInit.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_pos_transfo[:,1], mode='lines', name='world_ObserverLimb_pos_transfo_y'))
    figTransfoInit.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_pos_transfo[:,2], mode='lines', name='world_ObserverLimb_pos_transfo_z'))

    figTransfoInit.update_layout(title=f"{scriptName}: Transformations before matching")


    # Show the plotly figures
    figTransfoInit.show()


###############################  Average around matching point  ###############################


# Find the index in the pandas dataframe that corresponds to the input time
matchIndex = mocapData[mocapData['t'] == matchTime].index[0]

lowerIndex = matchIndex - averageInterval
if(lowerIndex < 0):
    lowerIndex = 0

overlapIndex[:matchIndex] = 0

world_MocapLimb_Pos_average_atMatch = np.mean(world_MocapLimb_Pos[lowerIndex:matchIndex + averageInterval], axis = 0)
world_ObserverLimb_Pos_average_atMatch = np.mean(world_ObserverLimb_Pos[lowerIndex:matchIndex + averageInterval], axis = 0)

world_MocapLimb_Ori_Quat = world_MocapLimb_Ori_R.as_quat()
world_ObserverLimb_Ori_quat = world_ObserverLimb_Ori_R.as_quat()

world_MocapLimb_Ori_Quat_average_atMatch = np.mean(world_MocapLimb_Ori_Quat[lowerIndex:matchIndex + averageInterval], axis = 0)
world_ObserverLimb_Ori_Quat_average_atMatch = np.mean(world_ObserverLimb_Ori_quat[lowerIndex:matchIndex + averageInterval], axis = 0)


world_MocapLimb_Ori_R_average_atMatch = R.from_quat(normalize(world_MocapLimb_Ori_Quat_average_atMatch))
world_ObserverLimb_Ori_R_average_atMatch = R.from_quat(normalize(world_ObserverLimb_Ori_Quat_average_atMatch))



###############################  Computation of the aligned mocap's pose  ###############################

MocapLimb_world_Ori_R_average_atMatch = world_MocapLimb_Ori_R_average_atMatch.inv()

# mocapObserver_Ori_R = MocapLimb_world_Ori_R_average_atMatch * world_ObserverLimb_Ori_R_average_atMatch
# new_world_MocapLimb_Ori_R = world_MocapLimb_Ori_R * mocapObserver_Ori_R
# new_world_MocapLimb_Ori_R = world_ObserverLimb_Ori_R_average_atMatch * MocapLimb_world_Ori_R_average_atMatch * world_MocapLimb_Ori_R



# Allows the yaw of the mocap to match with the one of the Observer at the desired time
world_ObserverLimb_Ori_R_transfoWithRespectToMatch = world_MocapLimb_Ori_R_average_atMatch.inv() * world_MocapLimb_Ori_R
mergedOriAtMatch = merge_tilt_with_yaw_axis_agnostic(world_MocapLimb_Ori_R_average_atMatch.apply(np.array([0, 0, 1]), inverse=True), world_ObserverLimb_Ori_R_average_atMatch.as_matrix())
mergedOriAtMatch_R = R.from_matrix(mergedOriAtMatch)
new_world_MocapLimb_Ori_R = mergedOriAtMatch_R * world_ObserverLimb_Ori_R_transfoWithRespectToMatch

# We do the same for the position
new_world_MocapLimb_Pos = world_ObserverLimb_Pos_average_atMatch + (mergedOriAtMatch_R * world_MocapLimb_Ori_R_average_atMatch.inv()).apply(world_MocapLimb_Pos - world_MocapLimb_Pos_average_atMatch)


###############################  Plot of the matched poses  ###############################


new_world_MocapLimb_Ori_euler = new_world_MocapLimb_Ori_R.as_euler("xyz")
world_ObserverLimb_Ori_euler = world_ObserverLimb_Ori_R.as_euler("xyz")

new_world_MocapLimb_Ori_euler_continuous = continuous_euler(new_world_MocapLimb_Ori_euler)
world_ObserverLimb_Ori_euler_continuous = continuous_euler(world_ObserverLimb_Ori_euler)

figNewPose = go.Figure()

figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=new_world_MocapLimb_Ori_euler_continuous[:,0], mode='lines', name='world_MocapLimb_Ori_roll'))
figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=new_world_MocapLimb_Ori_euler_continuous[:,1], mode='lines', name='world_MocapLimb_Ori_pitch'))
figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=new_world_MocapLimb_Ori_euler_continuous[:,2], mode='lines', name='world_MocapLimb_Ori_yaw'))

figNewPose.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_euler_continuous[:,0], mode='lines', name='world_ObserverLimb_Ori_roll'))
figNewPose.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_euler_continuous[:,1], mode='lines', name='world_ObserverLimb_Ori_pitch'))
figNewPose.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_euler_continuous[:,2], mode='lines', name='world_ObserverLimb_Ori_yaw'))


figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=new_world_MocapLimb_Pos[:,0], mode='lines', name='world_MocapLimb_Pos_x'))
figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=new_world_MocapLimb_Pos[:,1], mode='lines', name='world_MocapLimb_Pos_y'))
figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=new_world_MocapLimb_Pos[:,2], mode='lines', name='world_MocapLimb_Pos_z'))

figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=world_ObserverLimb_Pos[:,0], mode='lines', name='world_ObserverLimb_Pos_x'))
figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=world_ObserverLimb_Pos[:,1], mode='lines', name='world_ObserverLimb_Pos_y'))
figNewPose.add_trace(go.Scatter(x=mocapData["t"], y=world_ObserverLimb_Pos[:,2], mode='lines', name='world_ObserverLimb_Pos_z'))

figNewPose.update_layout(title=f"{scriptName}: Pose after matching")

figNewPose.write_image(f'{path_to_project}/output_data/scriptResults/matchInitPose/spatially_aligned_pos_transfo.png')

if(displayLogs):
    # Show the plotly figure
    figNewPose.show()



#####################  Orientation and position difference wrt the initial frame  #####################


new_world_MocapLimb_pos_transfo = new_world_MocapLimb_Ori_R[0].apply(new_world_MocapLimb_Pos - new_world_MocapLimb_Pos[0], inverse=True)
world_ObserverLimb_pos_transfo = world_ObserverLimb_Ori_R[0].apply(world_ObserverLimb_Pos - world_ObserverLimb_Pos[0], inverse=True)

new_world_MocapLimb_Ori_R_transfo = new_world_MocapLimb_Ori_R[0].inv() * new_world_MocapLimb_Ori_R
world_ObserverLimb_Ori_R_transfo = world_ObserverLimb_Ori_R[0].inv() * world_ObserverLimb_Ori_R

new_world_MocapLimb_Ori_transfo_euler = new_world_MocapLimb_Ori_R_transfo.as_euler("xyz")
world_ObserverLimb_Ori_transfo_euler = world_ObserverLimb_Ori_R_transfo.as_euler("xyz")

new_world_MocapLimb_Ori_transfo_euler_continuous = continuous_euler(new_world_MocapLimb_Ori_transfo_euler)
world_ObserverLimb_Ori_transfo_euler_continuous = continuous_euler(world_ObserverLimb_Ori_transfo_euler)

figTransfo = go.Figure()

figTransfo.add_trace(go.Scatter(x=mocapData["t"], y=new_world_MocapLimb_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_MocapLimb_Ori_transfo_roll'))
figTransfo.add_trace(go.Scatter(x=mocapData["t"], y=new_world_MocapLimb_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_MocapLimb_Ori_transfo_pitch'))
figTransfo.add_trace(go.Scatter(x=mocapData["t"], y=new_world_MocapLimb_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_MocapLimb_Ori_transfo_yaw'))

figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_transfo_euler_continuous[:,0], mode='lines', name='world_ObserverLimb_Ori_transfo_roll'))
figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_transfo_euler_continuous[:,1], mode='lines', name='world_ObserverLimb_Ori_transfo_pitch'))
figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_Ori_transfo_euler_continuous[:,2], mode='lines', name='world_ObserverLimb_Ori_transfo_yaw'))


figTransfo.add_trace(go.Scatter(x=mocapData["t"], y=new_world_MocapLimb_pos_transfo[:,0], mode='lines', name='world_MocapLimb_pos_transfo_x'))
figTransfo.add_trace(go.Scatter(x=mocapData["t"], y=new_world_MocapLimb_pos_transfo[:,1], mode='lines', name='world_MocapLimb_pos_transfo_y'))
figTransfo.add_trace(go.Scatter(x=mocapData["t"], y=new_world_MocapLimb_pos_transfo[:,2], mode='lines', name='world_MocapLimb_pos_transfo_z'))

figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_pos_transfo[:,0], mode='lines', name='world_ObserverLimb_pos_transfo_x'))
figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_pos_transfo[:,1], mode='lines', name='world_ObserverLimb_pos_transfo_y'))
figTransfo.add_trace(go.Scatter(x=observer_data["t"], y=world_ObserverLimb_pos_transfo[:,2], mode='lines', name='world_ObserverLimb_pos_transfo_z'))

figTransfo.update_layout(title=f"{scriptName}: Transformations after matching")

figTransfo.write_image(f'{path_to_project}/output_data/scriptResults/matchInitPose/spatially_aligned_ori_transfo.png')

if(displayLogs):
    # Show the plotly figures
    figTransfo.show()


new_world_MocapLimb_Ori_quat = new_world_MocapLimb_Ori_R.as_quat()

mocapData['worldMocapLimbPos_x'] = new_world_MocapLimb_Pos[:,0]
mocapData['worldMocapLimbPos_y'] = new_world_MocapLimb_Pos[:,1]
mocapData['worldMocapLimbPos_z'] = new_world_MocapLimb_Pos[:,2]
mocapData['worldMocapLimbOri_qx'] = new_world_MocapLimb_Ori_quat[:,0]
mocapData['worldMocapLimbOri_qy'] = new_world_MocapLimb_Ori_quat[:,1]
mocapData['worldMocapLimbOri_qz'] = new_world_MocapLimb_Ori_quat[:,2]
mocapData['worldMocapLimbOri_qw'] = new_world_MocapLimb_Ori_quat[:,3]

mocapData['overlapTime'] = overlapIndex


observer_data['Mocap_pos_x'] = new_world_MocapLimb_Pos[:,0]
observer_data['Mocap_pos_y'] = new_world_MocapLimb_Pos[:,1]
observer_data['Mocap_pos_z'] = new_world_MocapLimb_Pos[:,2]
observer_data['Mocap_ori_x'] = new_world_MocapLimb_Ori_quat[:,0]
observer_data['Mocap_ori_y'] = new_world_MocapLimb_Ori_quat[:,1]
observer_data['Mocap_ori_z'] = new_world_MocapLimb_Ori_quat[:,2]
observer_data['Mocap_ori_w'] = new_world_MocapLimb_Ori_quat[:,3]
observer_data['Mocap_datasOverlapping'] = mocapData['overlapTime'].apply(lambda x: 'Datas overlap' if x == 1 else 'Datas not overlapping')


#####################  3D plot of the pose  #####################

if(displayLogs):
    x_min = min((world_MocapLimb_Pos[:,0]).min(), (new_world_MocapLimb_Pos[:,0]).min(), (world_ObserverLimb_Pos[:,0]).min())
    y_min = min((world_MocapLimb_Pos[:,1]).min(), (new_world_MocapLimb_Pos[:,1]).min(), (world_ObserverLimb_Pos[:,1]).min())
    z_min = min((world_MocapLimb_Pos[:,2]).min(), (new_world_MocapLimb_Pos[:,2]).min(), (world_ObserverLimb_Pos[:,2]).min())
    x_min = x_min - np.abs(x_min*0.2)
    y_min = y_min - np.abs(y_min*0.2)
    z_min = z_min - np.abs(z_min*0.2)

    x_max = max((world_MocapLimb_Pos[:,0]).max(), (new_world_MocapLimb_Pos[:,0]).max(), (world_ObserverLimb_Pos[:,0]).max())
    y_max = max((world_MocapLimb_Pos[:,1]).max(), (new_world_MocapLimb_Pos[:,1]).max(), (world_ObserverLimb_Pos[:,1]).max())
    z_max = max((world_MocapLimb_Pos[:,2]).max(), (new_world_MocapLimb_Pos[:,2]).max(), (world_ObserverLimb_Pos[:,2]).max())
    x_max = x_max + np.abs(x_max*0.2)
    y_max = y_max + np.abs(y_max*0.2)
    z_max = z_max + np.abs(z_max*0.2)


    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter3d(
        x=world_MocapLimb_Pos[:,0], 
        y=world_MocapLimb_Pos[:,1], 
        z=world_MocapLimb_Pos[:,2],
        mode='lines',
        line=dict(color='darkblue'),
        name='world_MocapLimb_Pos'
    ))

    fig.add_trace(go.Scatter3d(
        x=new_world_MocapLimb_Pos[:,0], 
        y=new_world_MocapLimb_Pos[:,1], 
        z=new_world_MocapLimb_Pos[:,2],
        mode='lines',
        line=dict(color='darkred'),
        name='new_world_MocapLimb_Pos'
    ))

    fig.add_trace(go.Scatter3d(
        x=world_ObserverLimb_Pos[:,0], 
        y=world_ObserverLimb_Pos[:,1], 
        z=world_ObserverLimb_Pos[:,2],
        mode='lines',
        line=dict(color='darkgreen'),
        name='world_ObserverLimb_Pos'
    ))

    # Add big points at the initial positions
    fig.add_trace(go.Scatter3d(
        x=[world_MocapLimb_Pos[0,0]], 
        y=[world_MocapLimb_Pos[0,1]], 
        z=[world_MocapLimb_Pos[0,2]],
        mode='markers',
        marker=dict(size=5, color='darkblue'),
        name='Start world_MocapLimb_Pos'
    ))

    fig.add_trace(go.Scatter3d(
        x=[new_world_MocapLimb_Pos[0,0]], 
        y=[new_world_MocapLimb_Pos[0,1]], 
        z=[new_world_MocapLimb_Pos[0,2]],
        mode='markers',
        marker=dict(size=5, color='darkred'),
        name='Start new_world_MocapLimb_Pos'
    ))

    fig.add_trace(go.Scatter3d(
        x=[world_ObserverLimb_Pos[0,0]], 
        y=[world_ObserverLimb_Pos[0,1]], 
        z=[world_ObserverLimb_Pos[0,2]],
        mode='markers',
        marker=dict(size=5, color='darkgreen'),
        name='Start world_ObserverLimb_Pos'
    ))

    # Add a big point at the matching time
    fig.add_trace(go.Scatter3d(
        x=[new_world_MocapLimb_Pos[matchIndex,0]], 
        y=[new_world_MocapLimb_Pos[matchIndex,1]], 
        z=[new_world_MocapLimb_Pos[matchIndex,2]],
        mode='markers',
        marker=dict(size=5, color='darkorange'),
        name='Matching pose'
    ))

    # Update layout
    fig.update_layout(
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
        , title=f"{scriptName}: 3D trajectory after matching"
    )

    # Show the plot
    fig.show()





# Save the DataFrame to a new CSV file
if(len(sys.argv) > 3):
    save_csv = sys.argv[3].lower()
else:
    save_csv = input("Do you want to save the data as a CSV file? (y/n): ")
    save_csv = save_csv.lower()


if save_csv == 'y':
    mocapData.to_csv(output_csv_file_path, index=False, sep=';')
    observer_data.to_csv(f'{path_to_project}/output_data/observerResultsCSV.csv', index=False, sep=';')
    print("Output CSV file has been saved to ", output_csv_file_path)
else:
    print("Data not saved.")


