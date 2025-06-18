import yaml
import sys

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
                        keys_set.add(axis.rsplit('_', 1)[0])
                        break
                else:
                    keys_set.add(observer['kinematics'][body][kine])

# Define a list of patterns you want to match
partial_pattern = ['MocapAligner*', 'HartleyIEKF*', 'Accelerometer_linearAcceleration*', 'Accelerometer_angularVelocity*']  # Add more patterns as needed
exact_patterns = ['t', 'perf_GlobalRun']  # Add more column names as needed

keys_set = set(exact_patterns)

keys_set = keys_set.union(partial_pattern)
add_observers_columns()

import shlex

keys = []

for key in keys_set:
    keys.append(f'{key}')

# Create a string with each item quoted and shell-safe
print(shlex.join(keys))