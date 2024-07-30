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
observer_data = pd.read_csv(f'{path_to_project}/output_data/lightData.csv')
mocapData = pd.read_csv(f'{path_to_project}/output_data/realignedMocapLimbData.csv', delimiter=',')



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

world_MocapLimb_Pos = np.array([mocapData['worldMocapLimbPos_x'], mocapData['worldMocapLimbPos_y'], mocapData['worldMocapLimbPos_z']]).T
world_MocapLimb_Ori_R = R.from_quat(mocapData[["worldMocapLimbOri_qx", "worldMocapLimbOri_qy", "worldMocapLimbOri_qz", "worldMocapLimbOri_qw"]].values)

# Extracting the poses coming from mc_rtc
world_ObserverLimb_Pos = np.array([observer_data['MocapAligner_worldBodyKine_position_x'], observer_data['MocapAligner_worldBodyKine_position_y'], observer_data['MocapAligner_worldBodyKine_position_z']]).T
world_ObserverLimb_Ori_R = R.from_quat(observer_data[["MocapAligner_worldBodyKine_ori_x", "MocapAligner_worldBodyKine_ori_y", "MocapAligner_worldBodyKine_ori_z", "MocapAligner_worldBodyKine_ori_w"]].values)
# We get the inverse of the orientation as the inverse quaternion was stored
world_ObserverLimb_Ori_R = world_ObserverLimb_Ori_R.inv()



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

    figInitPose.update_layout(title="Resulting pose")

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

    figTransfoInit.update_layout(title="Initial transformations")

    # Show the plotly figures
    figTransfoInit.show()


###############################  Average around matching point  ###############################


# Find the index in the pandas dataframe that corresponds to the input time
matchIndex = mocapData[mocapData['t'] == matchTime].index[0]

world_MocapLimb_Pos_average_atMatch = np.mean(world_MocapLimb_Pos[matchIndex - averageInterval:matchIndex + averageInterval], axis = 0)
world_ObserverLimb_Pos_average_atMatch = np.mean(world_ObserverLimb_Pos[matchIndex - averageInterval:matchIndex + averageInterval], axis = 0)

world_MocapLimb_Ori_Quat = world_MocapLimb_Ori_R.as_quat()
world_ObserverLimb_Ori_quat = world_ObserverLimb_Ori_R.as_quat()

world_MocapLimb_Ori_Quat_average_atMatch = np.mean(world_MocapLimb_Ori_Quat[matchIndex - averageInterval:matchIndex + averageInterval], axis = 0)
world_ObserverLimb_Ori_Quat_average_atMatch = np.mean(world_ObserverLimb_Ori_quat[matchIndex - averageInterval:matchIndex + averageInterval], axis = 0)


world_MocapLimb_Ori_R_average_atMatch = R.from_quat(normalize(world_MocapLimb_Ori_Quat_average_atMatch))
world_ObserverLimb_Ori_R_average_atMatch = R.from_quat(normalize(world_ObserverLimb_Ori_Quat_average_atMatch))



###############################  Computation of the aligned mocap's pose  ###############################

MocapLimb_world_Ori_R_average_atMatch = world_MocapLimb_Ori_R_average_atMatch.inv()

mocapObserver_Ori_R = MocapLimb_world_Ori_R_average_atMatch * world_ObserverLimb_Ori_R_average_atMatch
new_world_MocapLimb_Ori_R = world_MocapLimb_Ori_R * mocapObserver_Ori_R

# Allows the position of the mocap to match with the one of the Observer at the desired time, while preserving the transformation between the current frame and the inital one for each iteration
new_world_MocapLimb_Pos = (world_ObserverLimb_Ori_R_average_atMatch * world_ObserverLimb_Ori_R_average_atMatch.inv()).apply(world_MocapLimb_Pos - world_MocapLimb_Pos_average_atMatch) + world_ObserverLimb_Pos_average_atMatch



###############################  Plot of the matched poses  ###############################

if(displayLogs):
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

    figNewPose.update_layout(title="Resulting pose")

    # Show the plotly figure
    figNewPose.show()



#####################  Orientation and position difference wrt the initial frame  #####################

if(displayLogs):
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

    figTransfo.update_layout(title="Transformations")

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
    mocapData.to_csv(output_csv_file_path, index=False)
    print("Output CSV file has been saved to ", output_csv_file_path)
else:
    print("Data not saved.")


