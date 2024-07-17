#!/bin/zsh

cwd=$(pwd)


mocapLog="mocapData.csv"
if [ -f "$mocapLog" ]; then
    echo "The log file of the mocap was found."
else
    echo "The log file of the mocap does not exist or is not named as expected. Expected: $mocapLog."
    exit
fi

logReplayCSV="logReplay.csv"
if [ -f "$logReplayCSV" ]; then
    if [[ $(ag "mocap_worldBodyKine" $logReplayCSV) ]]; then
        echo "The csv file of the replay with the observers has been found."
    else
        echo "The csv file of the replay with the observers has been found but does not contain the transformation from the body to the floating base. Please delete this file ($logReplayCSV) and re-run the script."
        exit
fi
else
    logReplayBin="logReplay.bin"
    if [ -f "$logReplayBin" ]; then
        echo "The bin file of the replay with the observers has been found. Converting to csv."
        mc_bin_to_log $logReplayBin
    else
        mcrtcLog="controllerLog.bin"
        if [ -f "$mcrtcLog" ]; then
            echo "The log file of the controller was found."
            yamlFile="$HOME/.config/mc_rtc/controllers/Passthrough.yaml"
            if [ -f "$yamlFile" ]; then
                echo "The configuration file for the replay with the observers was found."
                if ! ag -i "MocapVisualizer" $yamlFile || ! ag -i "firstRun:" $yamlFile; then
                    # Execute your action here if both patterns are found
                    echo "Please add the MocapVisualizer observer to the list of the observers and check that the configuration firstRun exists, and that the name of the body is passed to the MCVanytEstimator under the category mocap/bodyName"
                    exit
                fi
            else
                echo "The scripts excepts to find a configuration file named $yamlFile if the complete log of the observers is not given."
                exit
            fi
            sed -i "/^\([[:space:]]*firstRun: \).*/s//\1"true"/" $yamlFile
            mc_rtc_ticker --replay-outputs -e -l $mcrtcLog
            cd /tmp/
            LOG=$(find -iname "mc-control*" | grep "Passthrough" | grep ".bin" | tail -1)
            mv $LOG $cwd/logReplay.bin
            mc_bin_to_log logReplay.bin
        else
            echo "The log file of the controller does not exist or is not named as expected. Expected: $mcrtcLog."
            exit
        fi
    fi
fi

logReplayCSV="logReplay.csv"

# Prompt the user for input
echo "Please enter the timestep of the controller in milliseconds: "
read timeStep

echo "Please enter the time at which you want the pose of the mocap and the one of the observer must match: "
read matchTime

resampledMocapData="resampledMocapData.csv"
if [ -f "$resampledMocapData" ]; then
    echo "The mocap's data has already been resampled. Using the existing data."
else
    echo "Starting the resampling of the mocap's signal."
    python resampleAndExtract_fromMocap.py "$timeStep" "False" "y"
    echo "Resampling of the mocap's signal completed."
fi

lightData="lightData.csv"
if [ -f "$lightData" ]; then
    echo "The light version of the observer's data has already been extracted. Using the existing data."
else
    echo "Starting the extraction of the light version of the observer's data."
    python extractLightReplayVersion.py
    echo "Extraction of the light version of the observer's data completed."
fi

realignedMocapLimbData="realignedMocapLimbData.csv"
if [ -f "$realignedMocapLimbData" ]; then
    echo "The temporally aligned version of the mocap's data already exists. Using the existing data."
else
    echo "Starting the cross correlation for temporal data alignement."
    python crossCorrelation.py "$timeStep" "False" "y"
    echo "Temporal alignement of the mocap's data with the observer's data completed."
fi

resultMocapLimbData="resultMocapLimbData.csv"
if [ -f "$resultMocapLimbData" ]; then
    echo "The mocap's data has already been completely treated. Using the existing data."
else
    echo "Matching the pose of the mocap with the pose of the observer at $matchTime seconds."
    python matchInitPose.py "$matchTime" "False" "y"
    echo "Matching of the pose of the mocap with the pose of the observer completed."
fi



echo "Do you want to replay the log with the obtained mocap's data?"
select replayWithMocap in "Yes" "No"; do
    case $replayWithMocap in
        Yes ) mcrtcLog="controllerLog.bin"; sed -i "/^\([[:space:]]*firstRun: \).*/s//\1"false"/" $yamlFile; mc_rtc_ticker --replay-outputs -e -l $mcrtcLog; break;;
        No ) exit;;
    esac
done