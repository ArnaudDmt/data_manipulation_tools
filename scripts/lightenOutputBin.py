import yaml
import sys
import subprocess

path_to_project = ".."
log_path = None

if(len(sys.argv) > 1):
    path_to_project = sys.argv[1]
if len(sys.argv) > 2:
    log_path = sys.argv[2]

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
# Paper contact-orientation and held-out-wrench plots consume these channels.
partial_pattern += [f'Observers_MainObserverPipeline_MCKineticsObserver_{family}*' for family in (
    'debug_contactKine_', 'debug_contactState_', 'debug_wrenchesInCentroid_',
    'MEKF_estimatedState_contact_', 'MEKF_estimatedState_extForceCentr', 'MEKF_estimatedState_extTorqueCentr')]
# The disturbance-wrench figure needs what the hand sensor measured. Keep that one sensor, not
# every sensor: on HRP5P_LongWalk (1.47M rows) each extra family costs gigabytes, and
# extractLightReplayVersion reads the whole CSV into memory in one go.
partial_pattern += ['LeftHandForceSensor*']
# The Passthrough pipeline also runs VALINOR and the KO-ZPC instance, whose channels were being
# discarded here. Keep only the poses and velocities the figures and metrics consume -- taking
# the whole KOZPC_* family pulls in its debug contact channels and multiplies the log size.
partial_pattern += ['Observers_MainObserverPipeline_MCValinor_FloatingBase*',
                    'Observers_MainObserverPipeline_KOZPC_mcko_fb_posW*',
                    'Observers_MainObserverPipeline_KOZPC_mcko_fb_velW*']
exact_patterns = ['t', 'perf_GlobalRun']  # Add more column names as needed

keys_set = set(exact_patterns)

keys_set = keys_set.union(partial_pattern)
add_observers_columns()

if log_path:
    show = subprocess.run(["mc_bin_utils", "show", log_path], check=True, capture_output=True, text=True)
    available = {
        line[2:].split(" (", 1)[0]
        for line in show.stdout.splitlines()
        if line.startswith("- ")
    }
    keys_set = {
        key for key in keys_set
        if (key.endswith("*") and any(entry.startswith(key[:-1]) for entry in available))
        or (not key.endswith("*") and key in available)
    }

import shlex

keys = []

for key in sorted(keys_set):
    keys.append(f'{key}')

# Create a string with each item quoted and shell-safe
print(shlex.join(keys))
