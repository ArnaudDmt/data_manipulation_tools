This repository regroups tools that help alignigning the data obtained from a mocap with the data of mc_rtc. 
To this end, the mocap's data must be acquired the following way:
    - place the markers at the locations indicated on the pictures of MarkerPlacements.pdf. In the mocap's software, re-label the markers 1, 2 and 3 respectively: Marker1, Marker2 and Marker3.
    - at the end of the experiment, export the data to a csv file.

# How to run
run the script "./routine.sh" that will propose you to work either on an existing project or a new project. Upon creation of a new project, the script will create a dedicated folder containing the necessary folders. Once created, please paste the mocap's csv and mc_rtc's log inside the folder "raw_data" and fill in the file "projectConfig.yaml" with the names of the robot used and the limb the mocap markers are placed on. These names must match the ones used in mc_rtc.
Re-run the script "./routine.sh", which will ask you the timestep used by mc_rtc during the experiment.
To generate the necessary data, the mc_rtc's log will be replayed, please make sure that the file mc_rtc.yaml is correctly set up.


# Description of the main scripts
* resampleAndExtract_fromMocap.py : converts the point trajectories of the three specifically placed mocap markers to the floating base trajectory, after resampling the mocap's signal to the desired frequency.
* extractLightReplayVersion.py : extracts the necessary data from mc_rtc logs so its handling is faster
* crossCorrelation.py : performs cross-correlation between the mocap's data and mc_rtc's data (using the estimated local linear velocity which does not depend on non-observable variables and is thus less impacted by estimation inaccuracies) to find the time at which they temporally match. The mocap's data is then realigned and its length is then matched with the one of mc_rtc's log.
* matchInitPose.py : when the mocap and the controller are not started at the same time while working with a body different from the floating base for the data matching, one can get a discrepancy between the pose of the floating base obtained from the observer and from the mocap. This is due to the fact that the data of the mocap is considered constant when missing at the beginning and at the end (to get the same length that the observer's data), the obtained transformation between the first frame and each consecutive frame might thus be incorrect (ex: the head's orientation might change when startng the controller). The script matchInitPose.py allows to choose a time (in seconds) at which both data are supposed to match to solve this issue. It can also be used to compare the estimated poses from different starting points. Please note:
    * The variable 'overlapTime' is set to 1 (otherwise 0) at the times the mocap's data has not been filled with constant values and should match the one of the observer. When comparing the obtained trajectories, one should consider only this part.
    * The matching is made using the estimation contained in the realRobot of mc_rtc, if you want to match the mocap with a specific estimator, please make sure the update is made with it.


# Troubleshoot
If you encounter issues either with the final result or during the run, please check these possible reasons:
- Please make sure that the timestep you give at the beginning matches the one used during the run and with the one defined in ".config/mc_rtc/controllers/Passthrough.yaml".
- Please check that the body's name given in "~/.config/mc_rtc/plugins/MocapAligner.yaml" is correct.
- Please read the part about pose matching in the case the obtained displacement seems to be correct but the initial pose is not.
- If a specific script seems to be the issue, please copy and paste the folder "scripts" and the file "markersPlacements.yaml" in the root of the project and run the script using the command "python <script.py>". Some debugging logs will help identifying the issue.