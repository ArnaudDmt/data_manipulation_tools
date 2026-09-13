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

# Read the header alone to decide which columns are wanted, then load only those. Loading the
# whole file first and subsetting afterwards costs the full width in memory: HRP5P_LongWalk is
# 1.47M rows, and once the lightening kept a few more channel families its CSV reached 19 GB,
# which exhausted 62 GB of RAM and took the machine down.
header = pd.read_csv(input_csv_file_path, delimiter=';', nrows=0)
light_columns = filterColumns(header, partial_pattern, exact_patterns)
replayData_light = pd.read_csv(input_csv_file_path, delimiter=';', usecols=light_columns)
# usecols does not preserve the order asked for; restore it so downstream positional use holds.
replayData_light = replayData_light[light_columns]

if os.path.isfile(f'{path_to_project}/output_data/HartleyOutputCSV.csv') and 'HartleyIEKF_imuFbKine_position_x' in header.columns:
    dfHartley = pd.read_csv(f'{path_to_project}/output_data/HartleyOutputCSV.csv', delimiter=';')
    dfHartley=dfHartley.set_index(['t']).add_prefix('Hartley_').reset_index()

    # The parser emits exactly one row per log row, in order, and stamps `t` from its own counter.
    # So the two columns name the same instants but not the same floating-point values: the log's
    # time carries accumulated representation error (2699.0419999) where the parser's is written
    # rounded (2699.042). Joining on the raw float matched only while the two happened to agree
    # bit for bit and silently dropped everything after the first mismatch -- 13% of
    # HRP5P_LongWalk, whose evaluation window was cut by 249 s without any warning. Row order is
    # the relation that actually holds, so use it, and check the clocks agree to within half a
    # period rather than trusting them as a key.
    if len(dfHartley) == len(replayData_light):
        times = replayData_light['t'].to_numpy()
        step = pd.Series(times).diff().median()
        drift = abs(dfHartley['t'].to_numpy() - times).max()
        if drift > step / 2:
            raise RuntimeError(
                f"HartleyOutputCSV.csv and the log have the same length but their clocks differ by "
                f"up to {drift:.6g} s, more than half a period ({step / 2:.6g} s); they are not the "
                "same run")
        for column in (name for name in dfHartley.columns if name != 't'):
            replayData_light[column] = dfHartley[column].to_numpy()
    else:
        # Different lengths mean the two are not row-aligned; fall back to a tolerant time join so
        # the mismatch shows up as missing rows rather than as a silent truncation.
        print(f"WARNING HartleyOutputCSV.csv has {len(dfHartley)} rows for {len(replayData_light)} "
              "log rows; joining on time with a half-period tolerance")
        step = replayData_light['t'].diff().median()
        replayData_light = pd.merge_asof(replayData_light.sort_values('t'),
                                         dfHartley.sort_values('t'), on='t',
                                         direction='nearest', tolerance=step / 2)


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
