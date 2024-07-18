#!/bin/zsh

set -e



cwd=$(pwd)


rawDataPath="raw_data"
outputDataPath="output_data"
scriptsPath="scripts"

# files of the resulting data after each step
resampledMocapData="$outputDataPath/resampledMocapData.csv"
lightData="$outputDataPath/lightData.csv"
realignedMocapLimbData="$outputDataPath/realignedMocapLimbData.csv"
resultMocapLimbData="$outputDataPath/resultMocapLimbData.csv"




if [ ! -f "$realignedMocapLimbData" ]; then
    # Prompt the user for input
    echo "Please enter the timestep of the controller in milliseconds. This must match the timestep set in ".config/mc_rtc/controllers/Passthrough.yaml": "
    read timeStep
fi

yamlFile="$HOME/.config/mc_rtc/controllers/Passthrough.yaml"
if [ ! -f "$yamlFile" ]; then
    echo "The scripts excepts to find a configuration file named $yamlFile."
    exit
fi


mocapLog="$rawDataPath/mocapData.csv"
if [ -f "$mocapLog" ]; then
    echo "The log file of the mocap was found."
else
    echo "The log file of the mocap does not exist or is not named as expected. Expected: $mocapLog."
    exit
fi

logReplayCSV="$outputDataPath/logReplay.csv"
if [ -f "$logReplayCSV" ]; then
    echo "The csv file of the replay with the observers has been found."
else
    logReplayBin="$outputDataPath/logReplay.bin"
    if [ -f "$logReplayBin" ]; then
        echo "The bin file of the replay with the observers has been found. Converting to csv."
        cd $outputDataPath
        mc_bin_to_log "../$logReplayBin"
    else
        mcrtcLog="$rawDataPath/controllerLog.bin"
        if [ -f "$mcrtcLog" ]; then
            echo "The log file of the controller was found. Replaying the log with the observer."
            
            pluginConfPath="$HOME/.config/mc_rtc/plugins/MocapAligner.yaml"
            if [ ! -f "$pluginConfPath" ]; then
                mkdir -p $HOME/.config/mc_rtc/plugins 
                touch $pluginConfPath
            fi
            if ! grep -v '^#' $pluginConfPath | grep -q "bodyName"; then
                echo "Please give the name of the body the mocap markers are set on: "
                read bodyName
                if [ -s $pluginConfPath ]; then
                    awk -i inplace -v name="$bodyName" 'FNR==1 {print "bodyName:", name}1' $pluginConfPath
                else
                    echo "bodyName: $bodyName" > $pluginConfPath
                fi
            fi
            pluginActivated=true
            if ! grep -v '^#' $HOME/.config/mc_rtc/mc_rtc.yaml | grep -q "MocapAligner"; then
                pluginActivated=false
                echo "The plugin MocapAligner was not activated. Activating it for the replay."
                if [ -s $HOME/.config/mc_rtc/mc_rtc.yaml ]; then
                    awk -i inplace 'FNR==1 {print "Plugins: [MocapAligner] \n"}1' $HOME/.config/mc_rtc/mc_rtc.yaml
                else
                    echo "Plugins: [MocapAligner]" > $HOME/.config/mc_rtc/mc_rtc.yaml
                fi
            fi

            if ! ag -i "MocapVisualizer" $yamlFile || ! ag -i "firstRun:" $yamlFile; then
                # Execute your action here if both patterns are found
                echo "Please add the MocapVisualizer observer to the list of the observers and check that the configuration firstRun exists"
                exit
            fi
            sed -i "/^\([[:space:]]*firstRun: \).*/s//\1"true"/" $yamlFile
            mc_rtc_ticker --no-sync --replay-outputs -e -l $mcrtcLog
            cd /tmp/
            LOG=$(find -iname "mc-control*" | grep "Passthrough" | grep -v "latest" | grep ".bin" | sort | tail -1)
            echo "Copying the replay's bin file ($LOG) to the output_data folder"
            mv $LOG $cwd/$logReplayBin
            cd $cwd/$outputDataPath
            mc_bin_to_log logReplay.bin
            cd $cwd
        else
            echo "The log file of the controller does not exist or is not named as expected. Expected: $mcrtcLog."
            exit
        fi
    fi
fi


cd $cwd





if [ -f "$resampledMocapData" ]; then
    echo "The mocap's data has already been resampled. Using the existing data."
else
    echo "Starting the resampling of the mocap's signal."
    cd $cwd/$scriptsPath
    python resampleAndExtract_fromMocap.py "$timeStep" "False" "y"
    echo "Resampling of the mocap's signal completed."
fi
echo 

cd $cwd


if [ -f "$lightData" ]; then
    echo "The light version of the observer's data has already been extracted. Using the existing data."
else
    echo "Starting the extraction of the light version of the observer's data."
    cd $cwd/$scriptsPath
    python extractLightReplayVersion.py
    echo "Extraction of the light version of the observer's data completed."
fi
echo 

cd $cwd


if [ -f "$realignedMocapLimbData" ]; then
    echo "The temporally aligned version of the mocap's data already exists. Using the existing data."
else
    echo "Starting the cross correlation for temporal data alignement."
    cd $cwd/$scriptsPath
    python crossCorrelation.py "$timeStep" "False" "y"
    echo "Temporal alignement of the mocap's data with the observer's data completed."
fi
echo 

cd $cwd


if [ -f "$resultMocapLimbData" ]; then
    echo "The mocap's data has already been completely treated. Using the existing data."
else
    # Prompt the user for input
    echo "Please enter the time at which you want the pose of the mocap and the one of the observer must match: "
    read matchTime
    cd $cwd/$scriptsPath
    python matchInitPose.py "$matchTime" "False" "y"
    echo "Matching of the pose of the mocap with the pose of the observer completed."
fi
echo 

echo "Do you want to match the pose of the mocap and of the observer at a different timing?"
select changeMatchTime in "Yes" "No"; do
    case $changeMatchTime in
        Yes ) echo "Please enter the time at which you want the pose of the mocap and the one of the observer must match: " ; read matchTime; cd $cwd/$scriptsPath; python matchInitPose.py "$matchTime" "False" "y"; echo "Matching of the pose of the mocap with the pose of the observer completed."; break;;
        No ) exit;;
    esac
done

cd $cwd


echo "Do you want to replay the log with the obtained mocap's data?"
select replayWithMocap in "Yes" "No"; do
    case $replayWithMocap in
        Yes ) mcrtcLog="$rawDataPath/controllerLog.bin"; pwd; echo $mcrtcLog ; sed -i "/^\([[:space:]]*firstRun: \).*/s//\1"false"/" $yamlFile; mc_rtc_ticker --no-sync --replay-outputs -e -l $mcrtcLog; break;;
        No ) exit;;
    esac
done

echo "The pipeline finished without any issue. If you are not satisfied with the result, please re-run the scripts one by one and help yourself with the logs for the debug. Please also make sure that the time you set for the matching of the mocap and the observer is correct."