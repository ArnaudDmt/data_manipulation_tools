import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

###############################  User inputs  ###############################


if(len(sys.argv) > 1):
    timeStepInput = sys.argv[1]
    if(len(sys.argv) > 2):
        path_to_project = sys.argv[2]
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

output_csv_file_path_observers = f'{path_to_project}/output_data/repairedSkipped_mc_rtc_iters.csv'


###############################  Remove skipped indexes  ###############################
# In this part we check on what iterations the computation time exceeded the timestep, so we add a timestep to the time on these iterations and the following.

observer_data = pd.read_csv(f'{path_to_project}/output_data/lightData.csv',  delimiter=';')
perf_GlobalRun_log = pd.read_csv(f'{path_to_project}/output_data/perf_GlobalRun_log.csv',  delimiter=';')

def removeSkippedIndexes(observer_data, perf_GlobalRun_log):
    observer_data = observer_data.copy()

    # Make a copy of the 't' column to modify
    corrected_t = observer_data['t'].copy()

    # List to store new virtual rows
    virtual_rows = []

    # Iterate over the rows to fix the `t` column
    for i in range(1, len(observer_data)):
        delay = perf_GlobalRun_log.loc[i, 'perf_GlobalRun']
        if delay > timeStep_ms:
            nb_skipped = int(delay / timeStep_ms) - 1  # minus one because this step already exists
            print(f'Found {nb_skipped + 1} skipped iterations at index {i}.')

            for j in range(1, nb_skipped + 1):
                skipped_t = corrected_t.iloc[i - 1] + timeStep_s * j
                new_row = observer_data.iloc[i - 1].copy()
                new_row['t'] = skipped_t
                new_row['is_virtual'] = True
                virtual_rows.append(new_row)

            corrected_t.iloc[i:] += timeStep_s * nb_skipped

    # Update the corrected 't' column in the DataFrame
    observer_data['t'] = corrected_t
    observer_data['is_virtual'] = False  # Mark original rows as not virtual

    # Convert the list of virtual rows into a DataFrame
    virtual_data = pd.DataFrame(virtual_rows)

    # Combine the original data with the virtual rows
    combined_data = pd.concat([observer_data, virtual_data], ignore_index=True)

    
    # Sort by time to ensure proper order
    combined_data = combined_data.sort_values('t', kind='mergesort').reset_index(drop=True)

    return combined_data


combined_data = removeSkippedIndexes(observer_data, perf_GlobalRun_log)


combined_data.to_csv(output_csv_file_path_observers, index=False,  sep=';')

sys.exit(1)
print("Corrected observer data has been saved to ", output_csv_file_path_observers)
