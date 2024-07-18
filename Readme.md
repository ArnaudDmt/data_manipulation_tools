This repository regroups tools that help alignigning the data obtained from a mocap with the data of mc_rtc. 
To this end, the mocap's data must be acquired the following way:
    - place the markers at the locations indicated on the pictures of MarkerPlacements.pdf. In the mocap's software, re-label the markers 1, 2 and 3 respectively: Marker1, Marker2 and Marker3.
    - at the end of the experiment, export the data as a csv file and store it in the folder "raw_data".

The log obtained from mc_rtc must also be stored in the folder "raw_data".
To run the entire pipeline, please run "./routine.sh". If you encounter permission issues, please run "chmod +x routine.sh".


# Troubleshoot
If you encounter issues either with the final result or during the run, please check these possible reasons:
- Please make sure that the timestep you give at the beginning matches the one used during the run and with the one defined in ".config/mc_rtc/controllers/Passthrough.yaml".
- Please check that the body's name given in "~/.config/mc_rtc/plugins/MocapAligner.yaml" is correct.
- Please read the part about pose matching in the case the obtained displacement seems to be correct but the initial pose is not.

# Pose matching
When the mocap and the controller are not started at the same time while working with a body different from the floating base for the data matching, one can get a discrepancy between the pose of the floating base obtained from the observer and from the mocap. This is due to the fact that the data of the mocap is considered constant when missing at the beginning and at the end, the obtained transformation between the first frame and each consecutive frame might thus be incorrect (ex: the head's orientation might change when startng the controller). The script matchInitPose.py allows to choose a time (in seconds) at which both data are supposed to match to solve this issue. It can also be used to compare the estimated poses from different starting points. 