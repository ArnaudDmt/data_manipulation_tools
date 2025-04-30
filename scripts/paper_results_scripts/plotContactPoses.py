import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  # For colorKinetics palette generation
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import plotly.io as pio   
pio.kaleido.scope.mathjax = None


# Load the CSV files into pandas dataframes


default_path = '.../Projects/HRP5_MultiContact_1'

contactNames = ["RightFootForceSensor"] #, "LeftFootForceSensor", "LeftHandForceSensor"] # ["RightFootForceSensor", "LeftFootForceSensor", "LeftHandForceSensor"]
contactNames_areas_to_fill = ["LeftFootForceSensor", "LeftHandForceSensor"]

contactNames_restPoses = ["RightFootForceSensor"] # ["RightFootForceSensor", "LeftFootForceSensor", "LeftHandForceSensor"]
contactNames_areas_to_fill2 = [] #["RightFootForceSensor"]


contactNameToPlot = {"RightFootForceSensor": "Right foot", "LeftFootForceSensor": "Left foot", "LeftHandForceSensor": "Left hand"}

estimator_plot_args = {
    'Controller': {'name': 'Control', 'lineWidth': 5},
    'Vanyte': {'name': 'Vanyt-e', 'lineWidth': 5},
    'Hartley': {'name': 'RI-EKF', 'lineWidth': 5},
    'KineticsObserver': {'name': 'Kinetics Observer', 'lineWidth': 7},
    'KO_APC': {'name': 'KO_APC', 'lineWidth': 5},
    'KO_ASC': {'name': 'KO_ASC', 'lineWidth': 5},
    'KO_ZPC': {'name': 'KO-ZPC', 'lineWidth': 5},
    'KOWithoutWrenchSensors': {'name': 'KOWithoutWrenchSensors', 'lineWidth': 5},
    'Mocap': {'name': 'Ground truth', 'lineWidth': 5}
}




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


def plotContactPoses(estimators_to_plot = None, colors = None, path = default_path):

    encoders_data = pd.read_csv(f'{path}/output_data/logReplay.csv',  delimiter=';')

    observer_data = pd.read_csv(f'{path}/output_data/observerResultsCSV.csv',  delimiter=';')
    observer_data = observer_data[observer_data["Mocap_datasOverlapping"] == "Datas overlap"]

    estimatorsPoses = { 'Mocap': {'pos': observer_data[['Mocap_pos_x', 'Mocap_pos_y', 'Mocap_pos_z']].to_numpy(), \
                                    'ori': R.from_quat(observer_data[['Mocap_ori_x', 'Mocap_ori_y', 'Mocap_ori_z', 'Mocap_ori_w']].to_numpy())}, \
                        # 'Controller': {'pos': observer_data[['Controller_tx', 'Controller_ty', 'Controller_tz']].to_numpy(), \
                        #             'ori': R.from_quat(observer_data[['Controller_qx', 'Controller_qy', 'Controller_qz', 'Controller_qw']].to_numpy())}, \
                        'KineticsObserver': {  'pos': observer_data[['KO_posW_tx', 'KO_posW_ty', 'KO_posW_tz']].to_numpy(), \
                                                'ori': R.from_quat(observer_data[['KO_posW_qx', 'KO_posW_qy', 'KO_posW_qz', 'KO_posW_qw']].to_numpy())}, \
                        'KO_ZPC': {'pos': observer_data[['KO_ZPC_posW_tx', 'KO_ZPC_posW_ty', 'KO_ZPC_posW_tz']].to_numpy(), \
                                    'ori': R.from_quat(observer_data[['KO_ZPC_posW_qx', 'KO_ZPC_posW_qy', 'KO_ZPC_posW_qz', 'KO_ZPC_posW_qw']].to_numpy())}, \
                        'Hartley': {'pos': observer_data[['Hartley_Position_x', 'Hartley_Position_y', 'Hartley_Position_z']].to_numpy(), \
                                    'ori': R.from_quat(observer_data[['Hartley_Orientation_x', 'Hartley_Orientation_y', 'Hartley_Orientation_z', 'Hartley_Orientation_w']].to_numpy())}
                        }
    
    if(estimators_to_plot == None):
        estimators_to_plot = estimatorsPoses.keys()
    else:
        #estimators_to_plot = list(set(estimators_to_plot).intersection(estimatorsPoses.keys()))
        estimators_to_plot = [x for x in estimators_to_plot if x in estimatorsPoses]

    if(colors == None):
        # Generate colorKinetics palette for the estimators
        colors_t = px.colors.qualitative.Plotly  # Use Plotly's colorKinetics palette
        colors_t = [px.colors.hex_to_rgb(colors_t[i]) for i in range(len(estimators_to_plot))]
        colors = dict.fromkeys(estimators_to_plot)
        
        for i,estimator in enumerate(colors.keys()):
            colors[estimator] = colors_t[i]

    
    fig = go.Figure()

    fbContactPoses = dict.fromkeys(contactNames)

    index_range = [0,2830]
    y_mins = []
    y_maxs = []

    iterations = range(index_range[0], index_range[1])
    shapes = []

    for estimatorName in estimators_to_plot:
        colorEst = colors[estimatorName]
        colorEst = f'rgba({colorEst[0]}, {colorEst[1]}, {colorEst[2]}, 1)'

        for contactName in fbContactPoses.keys():
            # Create a boolean mask based on whether the contact state is "set"
            is_set_mask = encoders_data[f"Observers_MainObserverPipeline_MCKineticsObserver_debug_contactState_isSet_{contactName}"] == "Set"

            fbContactPoses[contactName] = {"position": None, "orientation": None}
            
            fbContactPoses[contactName]["position"] = encoders_data[[f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_position_x', f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_position_y', f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_position_z']].to_numpy()

            # Replace invalid quaternion rows with a default identity quaternion [0, 0, 0, 1]
            default_quat = [0, 0, 0, 1]

            ori_quat = encoders_data[
                [f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_orientation_x', 
                f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_orientation_y', 
                f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_orientation_z', 
                f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_orientation_w']
            ].to_numpy()

            # Replace invalid quaternions where isSet is "unset"
            ori_quat[~is_set_mask] = default_quat

            # Use `from_quat` on the full array
            orientations = R.from_quat(ori_quat)

            fbContactPoses[contactName]["orientation"] = R.from_quat(ori_quat).inv()

            # worldFbPos =  observer_data[[f'{estimatorsPath["Kinetics Observer"][0]}x', f'{estimatorsPath["Kinetics Observer"][0]}y', f'{estimatorsPath["Kinetics Observer"][0]}z']].to_numpy()
            # worldFbOri_quat = observer_data[[f'{estimatorsPath["Kinetics Observer"][1]}x', f'{estimatorsPath["Kinetics Observer"][1]}y', f'{estimatorsPath["Kinetics Observer"][1]}z', f'{estimatorsPath["Kinetics Observer"][1]}w']].to_numpy()
            # worldFbOri_R = R.from_quat(worldFbOri_quat).inv()

            worldFbPos = estimatorsPoses[estimatorName]['pos']
            worldFbOri_R = estimatorsPoses[estimatorName]['ori']

            worldContactPoses = dict.fromkeys(contactNames)

            worldContactPoses[contactName] = {"position": None, "orientation": None}
            worldContactPoses[contactName]["position"] = worldFbPos + worldFbOri_R.apply(fbContactPoses[contactName]["position"])
            worldContactPoses[contactName]["orientation"] = worldFbOri_R * fbContactPoses[contactName]["orientation"]

            worldContactOri_euler = worldContactPoses[contactName]["orientation"].as_euler('xyz', degrees=True)

            worldContactPoses[contactName]["position"][~is_set_mask] = None
            worldContactOri_euler[~is_set_mask] = None
            fbContactPoses[contactName]["position"][~is_set_mask] = None


            worldContactOri_euler_continuous = continuous_euler(worldContactOri_euler)

            # fig.add_trace(go.Scatter(x=observer_data["t"], y=worldContactPoses[contactName]["position"][:,0], mode='lines', name=f'{estimatorName} | {contactName}: position x'))
            # fig.add_trace(go.Scatter(x=observer_data["t"], y=worldContactPoses[contactName]["position"][:,1], mode='lines', name=f'{estimatorName} | {contactName}: position y'))
            # fig.add_trace(go.Scatter(x=observer_data["t"], y=worldContactPoses[contactName]["position"][:,2], mode='lines', name=f'{estimatorName} | {contactName}: position z'))

            # fig.add_trace(go.Scatter(x=observer_data["t"], y=worldContactOri_euler[:,0], mode='lines', name=f'{estimatorName} | {contactName}: roll'))
            # fig.add_trace(go.Scatter(x=observer_data["t"], y=worldContactOri_euler[:,1], mode='lines', name=f'{estimatorName} | {contactName}: pitch'))
            # fig.add_trace(go.Scatter(x=observer_data["t"], y=worldContactOri_euler[:,2], mode='lines', name=f'{estimatorName} | {contactName}: yaw'))

            fig.add_trace(go.Scatter(x=observer_data["t"], y=worldContactOri_euler[:,2], mode='lines', line=dict(color=colorEst, width = estimator_plot_args[estimatorName]['lineWidth']), name=f'{estimator_plot_args[estimatorName]["name"]}'))        

            list_without_nan = [x for x in worldContactOri_euler[index_range[0]:index_range[1],2] if str(x) != 'nan']
            y_mins.append(np.min(list_without_nan))
            y_maxs.append(np.max(list_without_nan))

    

    def generate_turbo_subset_colors(colormapName, contactsList):
        cmap = plt.get_cmap(colormapName)
        listCoeffs = np.linspace(0.2, 0.8, len(contactsList))
        colors={}
        # Generate colors and reduce intensity
        for idx, estimator in enumerate(contactsList):
            r, g, b, t = cmap(listCoeffs[idx])
            colors[estimator] = (r, g, b, 1) 
    
        return colors

    colors2 = generate_turbo_subset_colors('rainbow', contactNames_areas_to_fill)

    for contactName2 in contactNames_areas_to_fill:
        is_set_mask = encoders_data[f"Observers_MainObserverPipeline_MCKineticsObserver_debug_contactState_isSet_{contactName2}"][index_range[0]:index_range[1]] == "Set"
        #contact_state = [encoders_data[f"Observers_MainObserverPipeline_MCKineticsObserver_debug_contactState_isSet_{contactName2}"]  for _ in iterations]
        # Identify "Set" regions
        set_regions = []
        start = None
        for i, state in enumerate(is_set_mask):
            if state and start is None:
                # Begin a new region
                start = observer_data["t"][iterations[i]]  # Start of "Set" region
            elif not state and start is not None:
                # End the current region
                set_regions.append((start, observer_data["t"][iterations[i - 1]]))
                start = None

        # Handle the case where the last region ends at the final iteration
        if start is not None:
            set_regions.append((start, observer_data["t"][iterations[-1]]))

        # Assign colorKinetics for the current contact
        fillcolor2 = colors2[contactName2]
        fillcolor2 = f'rgba({fillcolor2[0]}, {fillcolor2[1]}, {fillcolor2[2]}, 0.3)'

        # Add an scatter trace as a legend proxy
        fig.add_trace(go.Scatter(
            x=[None],  # Dummy x value
            y=[None],  # Dummy y value
            mode="markers",
            marker=dict(size=10, color=fillcolor2),
            name=f"{contactNameToPlot[contactName2]}",
            legend="legend2"
        ))
        
        # Create shapes for shaded regions
        y_min, y_max = -10, 10  # Set bounds for y-axis
        for start, end in set_regions:
            shapes.append(dict(
                type="rect",
                xref="x",
                yref="y",
                x0=start,
                y0=y_min,
                x1=end,
                y1=y_max,
                opacity=0.4,
                line_width=0,
                layer="below",
                fillcolor=fillcolor2,
            ))

    y_min = min(y_mins)
    y_max = max(y_maxs)

    max_abs = max(np.abs(y_min), np.abs(y_max))

    y_min = y_min - max_abs * 0.1
    y_max = y_max + max_abs * 0.1

    #listData = list(fig.data)

    #print(listData[0].name)
    #print(listData[-len(contactNames_areas_to_fill) - 1].name)

    #listData = listData.reverse()
    #listData[0], listData[-len(contactNames_areas_to_fill) - 1] = listData[-len(contactNames_areas_to_fill) - 1], listData[0]

    #print(listData[0].name)
    #print(listData[-len(contactNames_areas_to_fill) - 1].name)

    #fig.data = tuple(listData)

    fig.update_xaxes(
                range = [observer_data["t"][index_range[0]], observer_data["t"][index_range[1]]]
            )

    fig.update_yaxes(
                range=[y_min, y_max]
            )
    fig.update_layout(
                shapes=shapes,
                xaxis_title="Time (s)",
                yaxis_title="Yaw (°)",
                template="plotly_white",
                legend=dict(
                    yanchor="bottom",
                    y=1.03,
                    xanchor="left",
                    x=0.01,
                    orientation='h',
                    bgcolor = 'rgba(0,0,0,0)',
                    traceorder='reversed',
                    font = dict(family = 'Times New Roman', size=20, color="black")
                    ),
                    legend2=dict(
                        yanchor="top",
                        y=0.98,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(0,0,0,0)',
                        font = dict(family = 'Times New Roman', size=20, color="black")
                    ),
                margin=dict(l=0,r=0,b=0,t=0),
                font = dict(family = 'Times New Roman', size=20, color="black")
            )

    # Show the plotly figure
    fig.show()

    fig.write_image(f'/tmp/rightFoot_yaw.pdf')

    

def plotContactRestPoses(colors = None, path = default_path):
    import numpy as np
    import plotly.graph_objects as go

    # Function to extract yaw from quaternion
    def quaternion_to_yaw(quat):
        x, y, z, w = quat
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # Function to create a surface given position and yaw
    def create_surface(position, yaw, size=0.05):
        half_size = size / 2
        # Define square vertices in local frame
        vertices = np.array([
            [-half_size, -half_size, 0],
            [half_size, -half_size, 0],
            [half_size, half_size, 0],
            [-half_size, half_size, 0]
        ])
        
        # Rotation matrix for yaw
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Rotate and translate vertices
        rotated_vertices = vertices @ rotation_matrix.T
        rotated_vertices += position
        
        # Define faces of the surface
        x, y, z = rotated_vertices.T
        return go.Mesh3d(
            x=x, y=y, z=z,
            opacity=0.1,
            color='blue',
            i=[0, 1, 2],
            j=[1, 2, 3],
            k=[2, 3, 0],
        )

    
    encoders_data = pd.read_csv(f'{path}/output_data/logReplay.csv',  delimiter=';')

    observer_data = pd.read_csv(f'{path}/output_data/observerResultsCSV.csv',  delimiter=';')
    observer_data = observer_data[observer_data["Mocap_datasOverlapping"] == "Datas overlap"]

    if(colors == None):
        # Generate colorKinetics palette for the estimators
        colors_t = px.colors.qualitative.Plotly  # Use Plotly's colorKinetics palette
        colors_t = [px.colors.hex_to_rgb(colors_t[0])]
        colors = dict.fromkeys("KineticsObserver")
        
        for i,estimator in enumerate(colors.keys()):
            colors[estimator] = colors_t[i]

    # Initialize plot
    fig3d = go.Figure()
    fig = go.Figure()

    fbContactPoses = dict.fromkeys(contactNames_restPoses)
    restContactPoses = dict.fromkeys(contactNames_restPoses)

    index_range = [0, len(observer_data) - 1]
    y_mins = []
    y_maxs = []

    iterations = range(index_range[0], index_range[1])
    shapes = []
    
    colorKinetics = colors["KineticsObserver"]
    colorKinetics = f'rgba({colorKinetics[0]}, {colorKinetics[1]}, {colorKinetics[2]}, 1)'

    colorMocap = colors["Mocap"]
    colorMocap = f'rgba({colorMocap[0]}, {colorMocap[1]}, {colorMocap[2]}, 1)'

    for contactName in restContactPoses.keys():
        # Create a boolean mask based on whether the contact state is "set"
        is_set_mask = encoders_data[f"Observers_MainObserverPipeline_MCKineticsObserver_debug_contactState_isSet_{contactName}"] == "Set"

        fbContactPoses[contactName] = {"position": None, "orientation": None}
        
        fbContactPoses[contactName]["position"] = encoders_data[[f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_position_x', f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_position_y', f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_position_z']].to_numpy()

        # Replace invalid quaternion rows with a default identity quaternion [0, 0, 0, 1]
        default_quat = [0, 0, 0, 1]

        ori_quat_fbContact = encoders_data[
            [f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_orientation_x', 
            f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_orientation_y', 
            f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_orientation_z', 
            f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_orientation_w']
        ].to_numpy()

        # Replace invalid quaternions where isSet is "unset"
        ori_quat_fbContact[~is_set_mask] = default_quat

        # Use `from_quat` on the full array
        orientations = R.from_quat(ori_quat_fbContact)

        

        mocapPose = {'pos': observer_data[['Mocap_pos_x', 'Mocap_pos_y', 'Mocap_pos_z']].to_numpy(), \
                                    'ori': R.from_quat(observer_data[['Mocap_ori_x', 'Mocap_ori_y', 'Mocap_ori_z', 'Mocap_ori_w']].to_numpy())
                        }
    

        worldFbPos_mocap = mocapPose['pos']
        worldFbOri_R_mocap = mocapPose['ori']

        worldContactPoses_mocap = dict.fromkeys(contactNames_restPoses)

        fbContactPoses[contactName]["orientation"] = R.from_quat(ori_quat_fbContact).inv()
        worldContactPoses_mocap[contactName] = {"position": None, "orientation": None}
        worldContactPoses_mocap[contactName]["position"] = worldFbPos_mocap + worldFbOri_R_mocap.apply(fbContactPoses[contactName]["position"])
        worldContactPoses_mocap[contactName]["orientation"] = worldFbOri_R_mocap * fbContactPoses[contactName]["orientation"]

        worldContactOri_mocap_euler = worldContactPoses_mocap[contactName]["orientation"].as_euler('xyz', degrees=True)

        worldContactPoses_mocap[contactName]["position"][~is_set_mask] = None
        worldContactOri_mocap_euler[~is_set_mask] = None
        fbContactPoses[contactName]["position"][~is_set_mask] = None

        restContactPoses[contactName] = {"position": None, "orientation": None}
        
        restContactPoses[contactName]["position"] = encoders_data[[f'Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_contact_{contactName}_position_x', f'Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_contact_{contactName}_position_y', f'Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_contact_{contactName}_position_z']].to_numpy()

        
        
        # Replace invalid quaternion rows with a default identity quaternion [0, 0, 0, 1]
        default_quat = [0, 0, 0, 1]

        ori_quat_KO_rest_init = encoders_data[
            [f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_initKine_ori_x', 
            f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_initKine_ori_y', 
            f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_initKine_ori_z', 
            f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_initKine_ori_w']
        ].to_numpy()

        ori_quat_KO_rest = encoders_data[
            [f'Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_contact_{contactName}_orientation_x', 
            f'Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_contact_{contactName}_orientation_y', 
            f'Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_contact_{contactName}_orientation_z', 
            f'Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_contact_{contactName}_orientation_w']
        ].to_numpy()

        # Replace invalid quaternions where isSet is "unset"
        ori_quat_KO_rest[~is_set_mask] = default_quat

        restContactPoses[contactName]["orientation"] = R.from_quat(ori_quat_KO_rest).inv()
        restContactOri_euler = restContactPoses[contactName]["orientation"].as_euler('xyz', degrees=True)

        zeroRoll = np.zeros(len(restContactOri_euler))

        restContactPoses[contactName]["position"][~is_set_mask] = None
        restContactOri_euler[~is_set_mask] = None
        zeroRoll[~is_set_mask] = None

        # Track previous state
        for i in range(1, len(is_set_mask)):
            if is_set_mask[i]:
                # Detect transition from unset to set
                if not is_set_mask[i - 1]:
                    # Update the previous value with ori_quat_KO_rest_init
                    ori = R.from_quat(ori_quat_KO_rest_init[i]).inv()
                    restContactOri_euler[i - 1] = ori.as_euler('xyz', degrees=True)
            else:
                restContactOri_euler[i] = None  # Keep None for unset states
        

        # for i in range(len(restContactPoses[contactName]["position"])):
        #     if i % 100 == 0:
        #         pos = restContactPoses[contactName]["position"][i]
                
        #         yaw = restContactOri_euler[i,2]
        #         surface = create_surface(pos, yaw)
        #         fig3d.add_trace(surface)


        # Find indices within the time range
        #time_range_indices = [3200,4400]

        # Find the index of the closest yaw to zero in the time range
        #closest_to_zero_index = time_range_indices[0] + np.argmin(np.abs(restContactOri_euler[time_range_indices[0]:time_range_indices[1], 2]))

        # fig.add_vline(
        #     x=observer_data["t"][closest_to_zero_index], 
        #     line=dict(color="red", dash="dash"), 
        #     name="Closest Yaw = 0"
        # )


        # fig.add_trace(go.Scatter(x=observer_data["t"], y=restContactPoses[contactName]["position"][:,0], mode='lines', name=f'Kinetics Observer | {contactName}: position x'))
        # fig.add_trace(go.Scatter(x=observer_data["t"], y=restContactPoses[contactName]["position"][:,1], mode='lines', name=f'Kinetics Observer | {contactName}: position y'))
        #fig.add_trace(go.Scatter(x=observer_data["t"], y=restContactPoses[contactName]["position"][:,2], mode='lines', name=f'Kinetics Observer | {contactName}: position z'))
        # fig.add_trace(go.Scatter(x=observer_data["t"], y=worldContactPoses_mocap[contactName]["position"][:,0], mode='lines', line=dict(color="black"), name=f'MoCap | {contactName}: position x'))
        # fig.add_trace(go.Scatter(x=observer_data["t"], y=worldContactPoses_mocap[contactName]["position"][:,1], mode='lines', line=dict(color="black"), name=f'MoCap | {contactName}: position y'))
        #fig.add_trace(go.Scatter(x=observer_data["t"], y=worldContactPoses_mocap[contactName]["position"][:,2], mode='lines', line=dict(color="black"), name=f'MoCap | {contactName}: position z'))
        
        #fig.add_trace(go.Scatter(x=observer_data["t"], y=restContactOri_euler[:,0], mode='lines', name=f'Kinetics Observer | {contactName}: roll'))
        #fig.add_trace(go.Scatter(x=observer_data["t"], y=restContactOri_euler[:,1], mode='lines', name=f'Kinetics Observer | {contactName}: pitch'))
        #fig.add_trace(go.Scatter(x=observer_data["t"], y=restContactOri_euler[:,2], mode='lines', name=f'Kinetics Observer | {contactName}: yaw'))
        #fig.add_trace(go.Scatter(x=observer_data["t"], y=worldContactOri_mocap_euler[:,0], mode='lines', line=dict(color="black"), name=f'MoCap | {contactName}: roll'))
        #fig.add_trace(go.Scatter(x=observer_data["t"], y=worldContactOri_mocap_euler[:,1], mode='lines', line=dict(color="black"), name=f'MoCap | {contactName}: pitch'))
        #fig.add_trace(go.Scatter(x=observer_data["t"], y=worldContactOri_mocap_euler[:,2], mode='lines', line=dict(color="black"), name=f'MoCap | {contactName}: yaw'))

        #fig.add_trace(go.Scatter(x=observer_data["t"], y=restContactOri_euler[:,2], mode='lines', line=dict(color=colorKinetics, width = estimator_plot_args[estimatorName]['lineWidth']), name=f'{estimator_plot_args[estimatorName]["name"]}'))        

        iter_start = 3000
        iter_end = 4600

        print(np.mean([x for x in worldContactOri_mocap_euler[iter_start:iter_end,0] if str(x) != 'nan']))

        figMain = go.Figure()

        figMain.add_trace(go.Scatter(x=observer_data["t"], y=restContactOri_euler[:,0], mode='lines', line=dict(color=colorKinetics, width = estimator_plot_args["KineticsObserver"]['lineWidth']), name=f'Kinetics Observer'))
        figMain.add_trace(go.Scatter(x=observer_data["t"], y=worldContactOri_mocap_euler[:,0], mode='lines', line=dict(color=colorMocap, width = estimator_plot_args["Mocap"]['lineWidth']), name=f'Motion Capture'))
        x_interval = [t for t, is_set in zip(observer_data["t"][iter_start:iter_end], is_set_mask[iter_start:iter_end]) if is_set]
        figMain.add_trace(go.Scatter(
                        x = x_interval,
                        y= - 19.7 * np.ones(len(x_interval)), 
                        line=dict(color="green", width = estimator_plot_args["Mocap"]['lineWidth']),
                        name='Actual Roll'
                    ))
        figMain.add_trace(go.Scatter(
                        x = observer_data["t"],
                        y= zeroRoll, 
                        line=dict(color="green", width = estimator_plot_args["Mocap"]['lineWidth'])
                    ))
        figInset = go.Figure()
        figInset.add_trace(go.Scatter(
                        x=observer_data["t"][iter_start:iter_end], 
                        y=restContactOri_euler[iter_start:iter_end,0], 
                        line=dict(color=colorKinetics, width = estimator_plot_args["KineticsObserver"]['lineWidth']), 
                        mode='lines',
                        showlegend= False
                    ))
        figInset.add_trace(go.Scatter(
                        x=observer_data["t"][iter_start:iter_end], 
                        y=worldContactOri_mocap_euler[iter_start:iter_end,0], 
                        line=dict(color=colorMocap, width = estimator_plot_args["Mocap"]['lineWidth']),
                        showlegend= False,
                    ))
        
        figInset.add_trace(go.Scatter(
                        x= x_interval,
                        y= - 19.7 * np.ones(len(x_interval)), 
                        line=dict(color="green", width = 2),
                        showlegend= False
                    ))
        
        minY2 = np.min(list(filter(lambda v: v==v, restContactOri_euler[iter_start:iter_end:,0] -1)))
        maxY2 = np.max(list(filter(lambda v: v==v, worldContactOri_mocap_euler[iter_start:iter_end:,0] +1)))

        rect_x_start_main = observer_data["t"][iter_start]
        rect_x_end_main = observer_data["t"][iter_end]
        rect_y_start_main = minY2
        rect_y_end_main = maxY2

        rect_x_start_inset = 0.5  # Domain start for xaxis2
        rect_x_end_inset = 0.8    # Domain end for xaxis2
        rect_y_start_inset = 0.1  # Domain start for yaxis2
        rect_y_end_inset = 0.40    # Domain end for yaxis2
        
        figMain.update_layout(
            plot_bgcolor= "rgba(0,0,0,0)", 
            paper_bgcolor= "rgba(0,0,0,0)",
            legend=dict(
                    yanchor="bottom",
                    y=1.00,
                    xanchor="left",
                    x=0.01,
                    orientation="h",
                    bgcolor="rgba(0,0,0,0)",
                    font = dict(family = 'Times New Roman', size=22, color="black"),
            ),
            margin=dict(l=0,r=0,b=0,t=0),
            font = dict(family = 'Times New Roman', size=22, color="black"),
            xaxis=dict(  # Primary x-axis configuration
                    title="Time (s)",
                    gridcolor= 'lightgrey', 
                    gridwidth= 3,
                ),
            yaxis=dict(  # Primary y-axis configuration
                    title="Roll (°)",
                    gridcolor= 'lightgrey',
                    gridwidth= 3,
                    zerolinecolor= 'lightgrey',
                ),
            shapes=[
                    # Rectangle for the inset area
                    dict(
                        type="rect",
                        xref="x",  # Reference to the primary x-axis
                        yref="y",  # Reference to the primary y-axis
                        x0=rect_x_start_main,
                        x1=rect_x_end_main,
                        y0=rect_y_start_main,
                        y1=rect_y_end_main,
                        line=dict(
                            color="black",  # Border color of the rectangle
                            width=2,
                        ),
                )]
        )

        figInset.update_layout(
            plot_bgcolor= "rgba(0,0,0,0)", 
            paper_bgcolor= "rgba(0,0,0,0)",
            margin=dict(l=0,r=0,b=0,t=0),
            font=dict(family="Times New Roman", size=60, color="black"),

            xaxis=dict(  # Inset x-axis configuration
                        #domain=[rect_x_start_inset, rect_x_end_inset],
                        range=[rect_x_start_main, rect_x_end_main],
                        gridcolor= 'lightgrey',
                        dtick=2
                    ),
            yaxis=dict(  # Inset y-axis configuration
                        #domain=[rect_y_start_inset, rect_y_end_inset],
                        range=[rect_y_start_main, rect_y_end_main],
                        gridcolor= 'lightgrey',
                        dtick=5
                    ),
            # shapes=[
            #         # Rectangle for the inset plot itself
            #         dict(
            #             type="rect",
            #             #fillcolor="white",
            #             xref="paper",  # Reference to the figure's paper coordinates
            #             yref="paper",  # Reference to the figure's paper coordinates
            #             x0=rect_x_start_inset - 0.04,
            #             x1=rect_x_end_inset + 0.03,
            #             y0=rect_y_start_inset - 0.05,
            #             y1=rect_y_end_inset + 0.04,
            #             line=dict(
            #                 color="black",  # Border color for the inset plot
            #                 width=2,
            #                 ),
            #             )
            #         ]

        )        

        # fig.update_xaxes(
        #             mirror=True,
        #             ticks='outside',
        #             showline=True,
        #             linecolor='black',
        #             gridcolor='lightgrey'
        #         )
        # fig.update_yaxes(
        #             mirror=True,
        #             ticks='outside',
        #             showline=True,
        #             linecolor='black',
        #             gridcolor='lightgrey'
        #         )
        
        # fig.update_layout({'plot_bgcolor': "rgba(0,0,0,0)", 'paper_bgcolor': "rgba(0,0,0,0)"})
        # fig.update_xaxes({'gridcolor': 'lightgrey', 'zerolinecolor': 'lightgrey', 'linecolor': 'white', 'linewidth': 5})
        # fig.update_yaxes({'gridcolor': 'lightgrey', 'zerolinecolor': 'lightgrey', 'linecolor': 'white'})


        figMain.show()
        figInset.show()
        figMain.write_image(f'/tmp/rightFoot_rest_roll_main.svg')
        figInset.write_image(f'/tmp/rightFoot_rest_roll_inset.svg')
        exit(0)
        fig.write_image(f'/tmp/rightFoot_rest_roll.pdf')

        y_mins.append(np.min(restContactOri_euler[:,0]))
        y_maxs.append(np.max(worldContactOri_mocap_euler[:,0]))

        y_mins.append(np.min(restContactPoses[contactName]["position"][:,2]))
        y_mins.append(np.min(worldContactPoses_mocap[contactName]["position"][:,2]))



    def generate_turbo_subset_colors(colormapName, contactsList):
        cmap = plt.get_cmap(colormapName)
        listCoeffs = np.linspace(0.2, 0.8, len(contactsList))
        colors={}
        # Generate colors and reduce intensity
        for idx, estimator in enumerate(contactsList):
            r, g, b, t = cmap(listCoeffs[idx])
            colors[estimator] = (r, g, b, 1) 
    
        return colors

    colors2 = generate_turbo_subset_colors('rainbow', contactNames_areas_to_fill2)

    for contactName2 in contactNames_areas_to_fill2:
        is_set_mask = encoders_data[f"Observers_MainObserverPipeline_MCKineticsObserver_debug_contactState_isSet_{contactName2}"][index_range[0]:index_range[1]] == "Set"
        #contact_state = [encoders_data[f"Observers_MainObserverPipeline_MCKineticsObserver_debug_contactState_isSet_{contactName2}"]  for _ in iterations]
        # Identify "Set" regions
        set_regions = []
        start = None
        for i, state in enumerate(is_set_mask):
            if state and start is None:
                # Begin a new region
                start = observer_data["t"][iterations[i]]  # Start of "Set" region
            elif not state and start is not None:
                # End the current region
                set_regions.append((start, observer_data["t"][iterations[i - 1]]))
                start = None

        # Handle the case where the last region ends at the final iteration
        if start is not None:
            set_regions.append((start, observer_data["t"][iterations[-1]]))

        # Assign colorKinetics for the current contact
        fillcolor2 = colors2[contactName2]
        fillcolor2 = f'rgba({fillcolor2[0]}, {fillcolor2[1]}, {fillcolor2[2]}, 0.3)'

        # Add an scatter trace as a legend proxy
        fig.add_trace(go.Scatter(
            x=[None],  # Dummy x value
            y=[None],  # Dummy y value
            mode="markers",
            marker=dict(size=10, color=fillcolor2),
            name=f"{contactNameToPlot[contactName2]}",
            legend="legend2"
        ))
        
        # Create shapes for shaded regions
        y_min, y_max = -20, 20  # Set bounds for y-axis
        for start, end in set_regions:
            shapes.append(dict(
                type="rect",
                xref="x",
                yref="y",
                x0=start,
                y0=y_min,
                x1=end,
                y1=y_max,
                opacity=0.4,
                line_width=0,
                layer="below",
                fillcolor=fillcolor2,
            ))

    y_min = min(y_mins)
    y_max = max(y_maxs)

    max_abs = max(np.abs(y_min), np.abs(y_max))

    y_min = y_min - max_abs * 0.1
    y_max = y_max + max_abs * 0.1

    fig.update_xaxes(
                range = [observer_data["t"][index_range[0]], observer_data["t"][index_range[1]]]
            )

    fig.update_yaxes(
                range=[y_min, y_max]
            )
    
    # Update plot layout
    fig3d.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    # Show plot
    #fig3d.show()
    
    fig.update_layout(
                shapes=shapes,
                xaxis_title="Time (s)",
                yaxis_title="Yaw (°)",
                template="plotly_white",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    orientation='h',
                    bgcolor = 'rgba(0,0,0,0)',
                    font = dict(family = 'Times New Roman')
                    ),
                    legend2=dict(
                        yanchor="top",
                        y=0.92,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(0,0,0,0)',
                    ),
                margin=dict(l=0,r=0,b=0,t=0),
                font = dict(family = 'Times New Roman')
            )

    # Show the plotly figure
    fig.show()

    fig.write_image(f'/tmp/rightFoot_rest_roll_2.pdf')




if __name__ == '__main__':
    plotContactPoses()
