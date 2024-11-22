import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px  # For color palette generation
from scipy.spatial.transform import Rotation as R

import plotly.io as pio   
pio.kaleido.scope.mathjax = None


# Load the CSV files into pandas dataframes


default_path = '.../Projects/HRP5_MultiContact_1'

contactNames = ["RightFootForceSensor"] #, "LeftFootForceSensor", "LeftHandForceSensor"] # ["RightFootForceSensor", "LeftFootForceSensor", "LeftHandForceSensor"]

estimator_plot_args = {
    'Controller': {'name': 'Control', 'lineWidth': 2},
    'Vanyte': {'name': 'Vanyt-e', 'lineWidth': 2},
    'Hartley': {'name': 'RI-EKF', 'lineWidth': 2},
    'KineticsObserver': {'name': 'Kinetics Observer', 'lineWidth': 4},
    'KO_APC': {'name': 'KO_APC', 'lineWidth': 2},
    'KO_ASC': {'name': 'KO_ASC', 'lineWidth': 2},
    'KO_ZPC': {'name': 'KO-ZPC', 'lineWidth': 2},
    'KODisabled_WithProcess': {'name': 'KODisabled_WithProcess', 'lineWidth': 2},
    'Mocap': {'name': 'Ground truth', 'lineWidth': 2}
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
        estimators_to_plot = list(set(estimators_to_plot).intersection(estimatorsPoses.keys()))

    if(colors == None):
        # Generate color palette for the estimators
        colors_t = px.colors.qualitative.Plotly  # Use Plotly's color palette
        colors_t = [px.colors.hex_to_rgb(colors_t[i]) for i in range(len(estimators_to_plot))]
        colors = dict.fromkeys(estimators_to_plot)
        
        for i,estimator in enumerate(colors.keys()):
            colors[estimator] = colors_t[i]

    
    fig = go.Figure()

    fbContactPoses = dict.fromkeys(contactNames)

    index_range = [0,3000]
    y_mins = []
    y_maxs = []

    for estimatorName in estimators_to_plot:
        color = colors[estimatorName]
        color = f'rgba({color[0]}, {color[1]}, {color[2]}, 1)'

        for contactName in fbContactPoses.keys():
            # Create a boolean mask based on whether the contact state is "set"
            is_set_mask = encoders_data[f"Observers_MainObserverPipeline_MCKineticsObserver_debug_contactState_isSet_{contactName}"] == "Set"

            fbContactPoses[contactName] = {"position": None, "orientation": None}
            
            fbContactPoses[contactName]["position"] = encoders_data[[f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_position_x', f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_position_y', f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_position_z']].to_numpy()
            ori_quat = encoders_data[[f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_orientation_x', f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_orientation_y', f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_orientation_z', f'Observers_MainObserverPipeline_MCKineticsObserver_debug_contactKine_{contactName}_inputUserContactKine_orientation_w']].to_numpy()
            
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

            fig.add_trace(go.Scatter(x=observer_data["t"], y=worldContactOri_euler[:,2], mode='lines', line=dict(color=color, width = estimator_plot_args[estimatorName]['lineWidth']), name=f'{estimator_plot_args[estimatorName]["name"]}'))

            list_without_nan = [x for x in worldContactOri_euler[index_range[0]:index_range[1],2] if str(x) != 'nan']
            y_mins.append(np.min(list_without_nan))
            y_maxs.append(np.max(list_without_nan))

    

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

    fig.update_layout(
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
                margin=dict(l=0,r=0,b=0,t=0),
                font = dict(family = 'Times New Roman')
            )

    # Show the plotly figure
    fig.show()

    fig.write_image(f'/tmp/rightFoot_yaw.pdf')
    



if __name__ == '__main__':
    plotContactPoses()
