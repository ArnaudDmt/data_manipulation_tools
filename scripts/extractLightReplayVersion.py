import os
import sys
import pandas as pd



path_to_project = ".."

if(len(sys.argv) > 1):
    path_to_project = sys.argv[1]



    
# Define a list of patterns you want to match
partial_pattern = ['MocapAligner', 'Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world', 'Observers_MainObserverPipeline_MCKineticsObserver_mcko_fb', 'Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState', 'Observers_MainObserverPipeline_KOAPC_mcko_fb', 'Observers_MainObserverPipeline_KOASC_mcko_fb', 'Observers_MainObserverPipeline_KOZPC_mcko_fb', 'Observers_MainObserverPipeline_KOWithoutWrenchSensors_mcko_fb', 'Observers_MainObserverPipeline_Tilt_FloatingBase_world', 'HartleyIEKF', 'Accelerometer', 'ff_']  # Add more patterns as needed
exact_patterns = ['t']  # Add more column names as needed
input_csv_file_path = f'{path_to_project}/output_data/logReplay.csv'
output_csv_file_path = f'{path_to_project}/output_data/lightData.csv'

# Filter columns based on the predefined patterns
def filterColumns(dataframe, partial_pattern, exact_patterns):
    filtered_columns = []
    for col in dataframe.columns:
        if col in exact_patterns or any(pattern in col for pattern in partial_pattern):
            filtered_columns.append(col)

    return filtered_columns

# Load the CSV files into pandas dataframes
replayData = pd.read_csv(f'{path_to_project}/output_data/logReplay.csv', delimiter=';')
light_columns = filterColumns(replayData, partial_pattern, exact_patterns)
replayData_light = replayData[light_columns].copy()

if os.path.isfile(f'{path_to_project}/output_data/HartleyOutputCSV.csv') and 'HartleyIEKF_imuFbKine_position_x' in replayData:
    dfHartley = pd.read_csv(f'{path_to_project}/output_data/HartleyOutputCSV.csv', delimiter=';')
    dfHartley=dfHartley.set_index(['t']).add_prefix('Hartley_').reset_index()

    replayData_light = pd.merge(replayData_light, dfHartley, on ='t')

replayData_light.rename(columns=lambda x: x.replace('Observers_MainObserverPipeline_MCKineticsObserver_mcko_fb', 'KO'), inplace=True)
replayData_light.rename(columns=lambda x: x.replace('Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState', 'KoState'), inplace=True)
replayData_light.rename(columns=lambda x: x.replace('Observers_MainObserverPipeline_KOAPC_mcko_fb', 'KO_APC'), inplace=True)
replayData_light.rename(columns=lambda x: x.replace('Observers_MainObserverPipeline_KOASC_mcko_fb', 'KO_ASC'), inplace=True)
replayData_light.rename(columns=lambda x: x.replace('Observers_MainObserverPipeline_KOZPC_mcko_fb', 'KO_ZPC'), inplace=True)
replayData_light.rename(columns=lambda x: x.replace('Observers_MainObserverPipeline_KOWithoutWrenchSensors_mcko_fb', 'KOWithoutWrenchSensors'), inplace=True)
replayData_light.rename(columns=lambda x: x.replace('Observers_MainObserverPipeline_MCVanytEstimator_FloatingBase_world', 'Vanyte'), inplace=True)    
replayData_light.rename(columns=lambda x: x.replace('Observers_MainObserverPipeline_Tilt_FloatingBase_world', 'Tilt'), inplace=True)
replayData_light.rename(columns=lambda x: x.replace('ff', 'Controller'), inplace=True)

replayData_light.to_csv(output_csv_file_path, index=False,  sep=';')
print("Output CSV file has been saved to", output_csv_file_path)
