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
python resampleAndExtract_fromMocap.py "$timeStep"
