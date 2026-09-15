This repository regroups tools that help alignigning the data obtained from a mocap with the data of mc_rtc. 

## Fast Kinetics evaluation

Create the repository Python environment once (PEP 668-safe):

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
```

Build the ROS 2 wrapper once, then prepare the four canonical HRP5P MultiContact and three RHPS1 Slippage datasets once:

```bash
cd /home/arnaud/devel/src/catkin_ws
colcon build --merge-install --packages-select kinetics_observer_ros2 test_state_obs_ros2

cd /home/arnaud/devel/src/data_manipulation_tools
.venv/bin/python scripts/kinetics_eval.py prepare --regenerate-missing
```

Preparation writes `kinetics_eval.yaml`. Edit that file, then run:

```bash
.venv/bin/python scripts/kinetics_eval.py run --label my-tuning
```

Each run is stored under `results/<label>-<config-hash>/`; `results/latest`
points to the newest run. The folder contains the exact YAML, Kinetics
trajectories, RPG relative-error outputs, per-project XY/position/roll-pitch-yaw plots, `summary.csv`, and `report.html`.
RI-EKF and Kinetics are both evaluated against the same synchronized mocap
trajectory. The recurring run is synchronous and does not replay ROS topics.

Run the resumable 128-trial covariance search with:

```bash
.venv/bin/python scripts/kinetics_tune.py
```

The trial ledger, winning overlay, and research summary are written to
`results/kinetics-tuning-constrained-20260822/`. The search constrains new-contact uncertainty to at least 1 cm/1 degree, scores slippage Z and velocity against RI-EKF, and can be extended with `--trials`. Use `--covariance-overlay PATH` before the `run`
subcommand to evaluate an overlay while preserving each robot's resolved settings.

To this end, the mocap's data must be acquired the following way:
    - place the markers at the locations indicated on the pictures of MarkerPlacements.pdf. In the mocap's software, re-label the markers 1, 2 and 3 respectively: Marker1, Marker2 and Marker3.
    - at the end of the experiment, export the data to a csv file.

# How to run
run the script "./routine.sh" that will propose you to work either on an existing project or a new project. Upon creation of a new project, the script will create a dedicated folder containing the necessary folders. Once created, please paste the mocap's csv and mc_rtc's log inside the folder "raw_data" and fill in the file "projectConfig.yaml" with the names of the robot used and the limb the mocap markers are placed on. These names must match the ones used in mc_rtc.
Re-run the script "./routine.sh", which will ask you the timestep used by mc_rtc during the experiment.
To generate the necessary data, the mc_rtc's log will be replayed, the pipeline reads each projectConfig.yaml and creates a per-project MainRobot replay override (HRP5P for MultiContact, RHPS1 for Slippage).
After changes, the script can be run with three options:
* --compute-metrics: compute essentially the trajectory evaluation metrics of the observers for a desired projet / set of projects. Takes time if the logs are long.
* --debug: run the routine with the possibility to re-run the steps one by one with more plots to debug the different steps.
* --plot-results: plot the results of the computed metrics for the desired project/set of projects, without recomputing these metrics. Calls generate_metrics_plots.py, which calls the individual codes for the plots. The latter can be commented in or out.


# Description of the main scripts
* resampleMocapAndCorrectObserversTime.py : converts the point trajectories of the three specifically placed mocap markers to the floating base trajectory, after resampling the mocap's signal to the desired frequency. Also corrects the time increments for the observers when an iteration took too much time.
* extractLightReplayVersion.py : extracts the necessary data from mc_rtc logs so its handling is faster
* crossCorrelation.py : performs cross-correlation between the mocap's data and mc_rtc's data (using the estimated local linear velocity which does not depend on non-observable variables and is thus less impacted by estimation inaccuracies) to find the time at which they temporally match. The mocap's data is then realigned and its length is then matched with the one of mc_rtc's log.
* matchInitPose.py : when the mocap and the controller are not started at the same time while working with a body different from the floating base for the data matching, one can get a discrepancy between the pose of the floating base obtained from the observer and from the mocap. This is due to the fact that the data of the mocap is considered constant when missing at the beginning and at the end (to get the same length that the observer's data), the obtained transformation between the first frame and each consecutive frame might thus be incorrect (ex: the head's orientation might change when startng the controller). The script matchInitPose.py allows to choose a time (in seconds) at which both data are supposed to match to solve this issue. It can also be used to compare the estimated poses from different starting points. Please note:
    * The variable 'overlapTime' is set to 1 (otherwise 0) at the times the mocap's data has not been filled with constant values and should match the one of the observer. When comparing the obtained trajectories, one should consider only this part.
    * The matching is made using the estimation contained in the realRobot of mc_rtc, if you want to match the mocap with a specific estimator, please make sure the update is made with it.


# Troubleshoot
If you encounter issues either with the final result or during the run, please check these possible reasons. To help you debugging, the resulting plot of each script is stored in the folder output_data/scriptResults/<scriptName>. You can also run the script with the argument "debug" to display dynamic plots and visualize each step better.
In addition:
- Please make sure that the timestep you give at the beginning matches the one used during the run and with the one defined in ".config/mc_rtc/controllers/Passthrough.yaml".
- Please check that the body's name given in "~/.config/mc_rtc/plugins/MocapAligner.yaml" is correct.
- Please read the part about pose matching in the case the obtained displacement seems to be correct but the initial pose is not.
