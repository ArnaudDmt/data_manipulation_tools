import os
import sys
import pandas as pd
import yaml



path_to_project = ".."

if(len(sys.argv) > 1):
    path_to_project = sys.argv[1]


with open('../observersInfos.yaml', 'r') as file:
    try:
        observersInfos_str = file.read()
        observersInfos_yamlData = yaml.safe_load(observersInfos_str)
    except yaml.YAMLError as exc:
        print(exc)

def add_observers_columns():
    # Iterate over the observers
    for observer in observersInfos_yamlData['observers']:
        for body in observer['kinematics']:
            for kine in observer['kinematics'][body]:
                if type(observer['kinematics'][body][kine]) is list:
                    for axis in observer['kinematics'][body][kine]:
                        exact_patterns.append(axis)
                else:
                    exact_patterns.append(observer['kinematics'][body][kine])
  
# Define a list of patterns you want to match
partial_pattern = ['MocapAligner', 'HartleyIEKF', 'Accelerometer_linearAcceleration', 'Accelerometer_angularVelocity']  # Add more patterns as needed
exact_patterns = ['t']  # Add more column names as needed
input_csv_file_path = f'{path_to_project}/output_data/logReplay.csv'
output_csv_file_path = f'{path_to_project}/output_data/lightData.csv'

add_observers_columns()

# Filter columns based on the predefined patterns
def filterColumns(dataframe, partial_pattern, exact_patterns):
    filtered_columns = []
    for col in dataframe.columns:
        if col in exact_patterns or any(pattern in col for pattern in partial_pattern):
            filtered_columns.append(col)

    return filtered_columns

# Load the CSV files into pandas dataframes
replayData = pd.read_csv(input_csv_file_path, delimiter=';')

light_columns = filterColumns(replayData, partial_pattern, exact_patterns)
replayData_light = replayData[light_columns].copy()

if os.path.isfile(f'{path_to_project}/output_data/HartleyOutputCSV.csv') and 'HartleyIEKF_imuFbKine_position_x' in replayData:
    dfHartley = pd.read_csv(f'{path_to_project}/output_data/HartleyOutputCSV.csv', delimiter=';')
    dfHartley=dfHartley.set_index(['t']).add_prefix('Hartley_').reset_index()

    replayData_light = pd.merge(replayData_light, dfHartley, on ='t')


def rename_observers_columns():
    # fetching the name of the body the mocap is attached to
    with open(f'{path_to_project}/projectConfig.yaml', 'r') as file:
        try:
            projConf_yaml_str = file.read()
            projConf_yamlData = yaml.safe_load(projConf_yaml_str)
            enabled_body = projConf_yamlData.get('EnabledBody')
            robotName = projConf_yamlData.get('EnabledRobot')
        except yaml.YAMLError as exc:
            print(exc)
            
    with open('../markersPlacements.yaml', 'r') as file:
        try:
            markersPlacements_str = file.read()
            markers_yamlData = yaml.safe_load(markersPlacements_str)
            for robot in markers_yamlData['robots']:
                # If the robot name matches
                if robot['name'] == robotName:
                    # Iterate over the bodies of the robot
                    for body in robot['bodies']:
                        # If the body name matches
                        if body['name'] == enabled_body:
                            mocapBody = body['standardized_name']

        except yaml.YAMLError as exc:
            print(exc)

    for observer in observersInfos_yamlData['observers']:
        for body in observer['kinematics']:
            prefix = observer['abbreviation']
            if body != mocapBody:
                prefix += '_' + body
            for kine in observer['kinematics'][body]:
                if kine == "position":
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][0], prefix + '_position_x'), inplace=True)
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][1], prefix + '_position_y'), inplace=True)
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][2], prefix + '_position_z'), inplace=True)
                if kine == "orientation":
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][0], prefix + '_orientation_x'), inplace=True)
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][1], prefix + '_orientation_y'), inplace=True)
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][2], prefix + '_orientation_z'), inplace=True)
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][3], prefix + '_orientation_w'), inplace=True)
                if kine == "linVel":
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][0], prefix + '_linVel_x'), inplace=True)
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][1], prefix + '_linVel_y'), inplace=True)
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][2], prefix + '_linVel_z'), inplace=True)
                if kine == "angVel":
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][0], prefix + '_angVel_x'), inplace=True)
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][1], prefix + '_angVel_y'), inplace=True)
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][2], prefix + '_angVel_z'), inplace=True)
                if kine == "locLinVel":
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][0], prefix + '_locLinVel_x'), inplace=True)
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][1], prefix + '_locLinVel_y'), inplace=True)
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][2], prefix + '_locLinVel_z'), inplace=True)
                if kine == "gyroBias":
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][0], prefix + '_gyroBias_x'), inplace=True)
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][1], prefix + '_gyroBias_y'), inplace=True)
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine][2], prefix + '_gyroBias_z'), inplace=True)
                if kine == "contact_isSet":
                    replayData_light.rename(columns=lambda x: x.replace(observer['kinematics'][body][kine], prefix + '_isSet'), inplace=True)
                    
rename_observers_columns()

cols = replayData_light.columns.tolist()
cols.insert(0, cols.pop(cols.index('t')))
df_Observers = replayData_light[cols]

replayData_light.insert(0, 't', replayData_light.pop('t'))

replayData_light.to_csv(output_csv_file_path, index=False,  sep=';')

print("Output CSV file has been saved to", output_csv_file_path)
