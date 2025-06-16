import sys
import yaml
import pandas as pd
import numpy as np

import numpy as np
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go

if(len(sys.argv) > 1):
    timeStepInput = sys.argv[1]
    if(len(sys.argv) > 2):
        path_to_project = sys.argv[2]
        if(len(sys.argv) > 3):
            heavy_log = sys.argv[3]


df_Observers = pd.read_csv(f'{path_to_project}/output_data/repairedSkipped_mc_rtc_iters.csv', delimiter=';')
dfHartley = pd.read_csv(f'{path_to_project}/output_data/HartleyOutputCSV.csv', delimiter=';')

try:
    # Check if the timestep was given in milliseconds
    if(timeStepInput.isdigit()):
        timeStep_ms = int(timeStepInput)
        timeStep_s = float(timeStep_ms)/1000.0
    else:
        timeStep_s = float(timeStepInput)
        timeStep_ms = int(timeStep_s*1000.0)

    actual_dt = df_Observers['t'].iloc[1] - df_Observers['t'].iloc[0]
    if not (np.isclose(actual_dt, timeStep_s, rtol=1e-5, atol=1e-9) or np.isclose(actual_dt, timeStep_s * 2, rtol=1e-5, atol=1e-9)):
        print(f"The timestep used in the pipeline ({timeStep_s} s) doesn't match that in the log, please check.")
        sys.exit(1)
    if heavy_log and timeStep_ms <= 3:
        # we check if the log has already been downsampled
        if np.isclose(actual_dt, timeStep_s * 2, rtol=1e-5, atol=1e-9):
            timeStep_s = timeStep_s * 2
        else:
            print(f"The log is heavy and was sampled at high frequency. Downsampling it. Initial sampling time: {timeStep_ms} ms. Downsampling at {timeStep_ms * 2} ms.") 
            timeStep_ms *= 2
            timeStep_s = float(timeStep_ms/1000)    
            df_Observers = df_Observers.iloc[::2].reset_index(drop=True)
            # dfHartley = dfHartley.iloc[::2].reset_index(drop=True)
            df_Observers.to_csv(f'{path_to_project}/output_data/repairedSkipped_mc_rtc_iters.csv', index=False, sep=';')
            # dfHartley.to_csv(f'{path_to_project}/output_data/HartleyOutputCSV.csv', index=False, sep=';')
except ValueError:
    print(f"The input timestep is not valid: {timeStepInput}")


# Extracting the poses coming from mc_rtc
world_ObserverLimb_Pos = np.array([df_Observers['MocapAligner_worldBodyKine_position_x'], df_Observers['MocapAligner_worldBodyKine_position_y'], df_Observers['MocapAligner_worldBodyKine_position_z']]).T


world_ObserverLimb_Vel_x = np.diff(world_ObserverLimb_Pos[:,0])/timeStep_s
world_ObserverLimb_Vel_y = np.diff(world_ObserverLimb_Pos[:,1])/timeStep_s
world_ObserverLimb_Vel_z = np.diff(world_ObserverLimb_Pos[:,2])/timeStep_s
world_ObserverLimb_Vel_x = np.insert(world_ObserverLimb_Vel_x, 0, 0.0, axis=0)
world_ObserverLimb_Vel_y = np.insert(world_ObserverLimb_Vel_y, 0, 0.0, axis=0)
world_ObserverLimb_Vel_z = np.insert(world_ObserverLimb_Vel_z, 0, 0.0, axis=0)
world_ObserverLimb_Vel = np.stack((world_ObserverLimb_Vel_x, world_ObserverLimb_Vel_y, world_ObserverLimb_Vel_z), axis = 1)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_Observers["t"], y=world_ObserverLimb_Vel[:,0], mode='lines', name='world_ObserverLimb_Vel_x'))
fig2.add_trace(go.Scatter(x=df_Observers["t"], y=world_ObserverLimb_Vel[:,1], mode='lines', name='world_ObserverLimb_Vel_y'))
fig2.add_trace(go.Scatter(x=df_Observers["t"], y=world_ObserverLimb_Vel[:,2], mode='lines', name='world_ObserverLimb_Vel_z'))

fig2.update_layout(title=f" Local linear velocity before alignment 2")
# Show the plotly figures
fig2.show()



# fetching the name of the body the mocap is attached to
with open(f'{path_to_project}/projectConfig.yaml', 'r') as file:
    try:
        projConf_yaml_str = file.read()
        projConf_yamlData = yaml.safe_load(projConf_yaml_str)
        enabled_body = projConf_yamlData.get('EnabledBody')
        robotName = projConf_yamlData.get('EnabledRobot')
    except yaml.YAMLError as exc:
        print(exc)

# fetching the standardized name of the body
with open('../markersPlacements.yaml', 'r') as file:
    try:
        markersPlacements_str = file.read()
        markersPlacements_yamlData = yaml.safe_load(markersPlacements_str)
        for robot in markersPlacements_yamlData['robots']:
            # If the robot name matches
            if robot['name'] == robotName:
                # Iterate over the bodies of the robot
                for body in robot['bodies']:
                    # If the body name matches
                    if body['name'] == enabled_body:
                        mocapBody = body['standardized_name']

    except yaml.YAMLError as exc:
        print(exc)

observersList = []
with open('../observersInfos.yaml', 'r') as file:
    try:
        observersInfos_str = file.read()
        observersInfos_yamlData = yaml.safe_load(observersInfos_str)
        for observer in observersInfos_yamlData['observers']:
            if observer["abbreviation"] != 'Mocap':
                if mocapBody in observer['kinematics']:
                    if observer['abbreviation'] + '_position_x' in df_Observers.columns:
                        observersList.append(observer["abbreviation"])
    except yaml.YAMLError as exc:
        print(exc)


if 'RI-EKF_IMU_position_x' in df_Observers.columns:
    observersList.append('RI-EKF')


observersList.append("Mocap")
class InlineListDumper(yaml.SafeDumper):
    def represent_sequence(self, tag, sequence, flow_style=None):
        # Use flow style (inline) only for top-level lists
        return super().represent_sequence(tag, sequence, flow_style=True)
    
data = dict(
    observers = observersList,
    robot = robotName,
    mocapBody = mocapBody,
    timeStep_s = timeStep_s
)

with open(f'{path_to_project}/output_data/observers_infos.yaml', 'w') as outfile:
    yaml.dump(data, outfile, Dumper=InlineListDumper, sort_keys=False)