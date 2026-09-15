############################ Configuration files test ############################


if [ ! -f "$replay_yaml" ]; then
    echo "The scripts excepts to find a configuration file named $replay_yaml."
    exit
fi


############################ Checking if a robot was given to select the mocap markers ############################

if grep -v '^#' $projectConfig | grep -q "Use_HartleyIEKF"; then
    if [[ ! $(grep 'Use_HartleyIEKF:' $projectConfig | grep -v '^#' | sed 's/Use_HartleyIEKF://' | sed 's: ::g') ]]; then
        echo "Plugin for Hartley's IEKF detected, do you want to add it to the comparison ?"
        select hartley in "Yes" "No"; do
            case $hartley in
                Yes ) 
                    sed -i "s/Use_HartleyIEKF:/& true/" $projectConfig
                    break;;
                No ) 
                    sed -i "s/Use_HartleyIEKF:/& false/" $projectConfig
                    break;;
            esac
        done
    fi
fi

if [[ $(grep 'Use_HartleyIEKF:' $projectConfig | grep -v '^#' | sed 's/Use_HartleyIEKF: //' | sed 's: ::g') == "true" ]]; then
    useHartley=true
else
    useHartley=false
fi

############################ Checking if the sensors must be made noisy ############################

if [[ $(grep 'Use_NoisySensors:' $projectConfig | grep -v '^#' | sed 's/Use_NoisySensors: //' | sed 's: ::g') == "true" ]]; then
    useNoisySensors=true
    echo -e "${YELLOW}The NoisySensors plugin will be used for the replay: the IMU signals will be degraded following $HOME/.config/mc_rtc/plugins/NoisySensors.yaml.${RESET}"
else
    useNoisySensors=false
fi



if grep -v '^#' $projectConfig | grep -q "EnabledRobot"; then
    if [[ ! $(grep 'EnabledRobot:' $projectConfig | grep -v '^#' | sed 's/EnabledRobot://' | sed 's: ::g') ]]; then
        echo "No robot was given in the configuration file $projectConfig. Use the robot defined in $mc_rtc_yaml ?"
        select useMainConfRobot in "Yes" "No"; do
            case $useMainConfRobot in
                Yes ) 
                    main_robot=$( grep 'MainRobot:' $mc_rtc_yaml | grep -v '^#' | sed 's/MainRobot: //');
                    break;;
                No ) 
                    echo "Please enter the name of the robot to add to $projectConfig: "; 
                    read main_robot;
                    break;;
            esac
        done
        sed -i "s/EnabledRobot:/& $main_robot/" $projectConfig
    else
        main_robot=$(grep 'EnabledRobot:' $projectConfig | grep -v '^#' | sed 's/EnabledRobot://' | sed 's: ::g');
        mc_rtc_robot=$(grep 'MainRobot:' $mc_rtc_yaml | grep -v '^#' | sed 's/MainRobot: //')
        if [[ "$main_robot" != "$mc_rtc_robot" ]]; then
            echo
            echo "WARNING: The robot defined in the configuration of the project in $projectConfig ($main_robot) doesn't match the one in $mc_rtc_yaml that will be used for the replay ($mc_rtc_robot) !!!"
            echo
        fi
    fi
else
    echo "No robot was given in the configuration file $projectConfig. Use the robot defined in $mc_rtc_yaml ?"
    select useMainConfRobot in "Yes" "No"; do
        case $useMainConfRobot in
            Yes ) 
                main_robot=$( grep 'MainRobot:' $mc_rtc_yaml | grep -v '^#' | sed 's/MainRobot: //');
                break;;
            No ) 
                echo "Please enter the name of the robot to add to $projectConfig: "; 
                read main_robot;
                break;;
        esac
    done
    if [ -s $projectConfig ]; then
        sed -i "1i\\EnabledRobot: $main_robot" "$projectConfig"
    else
        #echo "EnabledRobot: $main_robot" > $projectConfig
        echo -e "\nEnabledRobot: $main_robot" >> $projectConfig
    fi
fi

############################ Checking if a mocap body was given to select the mocap markers ############################

if grep -v '^#' $projectConfig | grep -q "EnabledBody"; then
    if [[ ! $(grep 'EnabledBody:' $projectConfig | grep -v '^#' | sed 's/EnabledBody://' | sed 's: ::g') ]]; then
        echo "No mocap body was given in the configuration file $projectConfig. Please enter the name of the body to add to $projectConfig: "; 
        echo "Available bodies for robot $main_robot:"
        yq -r ".robots[] | select(.name == \"$main_robot\") | .bodies[].name" $mocapMarkers_yaml
        read body;

        sed -i "s/EnabledBody:/& $body/" $projectConfig
    fi
else
    echo "No mocap body was given in the configuration file $projectConfig. Please enter the name of the body to add to $projectConfig: "; 
    echo "Available bodies for robot $main_robot:"
    yq -r ".robots[] | select(.name == \"$main_robot\") | .bodies[].name" $mocapMarkers_yaml
    read body;
    if [ -s $projectConfig ]; then
        sed -i "1i\\EnabledBody: $body" "$projectConfig"
    else
        #echo "EnabledBody: $body" > $projectConfig
        echo -e "\nEnabledBody: $body" >> $projectConfig
    fi
    
fi

bodyName=$(grep 'EnabledBody:' $projectConfig | grep -v '^#' | sed 's/EnabledBody://' | sed 's: ::g');


if grep -v '^#' $projectConfig | grep -q "Body_vel_eval"; then
    if [[ ! $(grep 'Body_vel_eval:' $projectConfig | grep -v '^#' | sed 's/Body_vel_eval://' | sed 's: ::g') ]]; then
        echo "No body for the linear velocity evaluation was given in the configuration file $projectConfig. Please enter the name of the body to add to $projectConfig: "; 
        read body;

        sed -i "s/Body_vel_eval:/& $body/" $projectConfig
    fi
else
    echo "No body for the linear velocity evaluation was given in the configuration file $projectConfig. Please enter the name of the body to add to $projectConfig: "; 
    read body;
    if [ -s $projectConfig ]; then
        sed -i "1i\\Body_vel_eval: $body" "$projectConfig"
    else
        echo -e "\Body_vel_eval: $body" >> $projectConfig
    fi
fi


# Changing the name of the project in the replay's configuration
sed -i "/^\([[:space:]]*projectName: \).*/s//\1"$projectName"/" $replay_yaml

############################ Fetching the mocap's log ############################

mocapLog="$rawDataPath/mocapData.csv"
if [ -f "$mocapLog" ]; then
    echo "The log file of the mocap was found."
else
    echo "The log file of the mocap does not exist or is not named as expected. Expected: $mocapLog."
    exit
fi

mcrtcLog="$rawDataPath/controllerLog.bin"

if [[ -f "$mcrtcLog" ]]; then
  fileSize=$(stat -c%s "$mcrtcLog")
  if (( fileSize > 8589934592 )); then
    echo -e "${YELLOW}The log is larger than 8 GB. Keeping only the necessary data.${RESET}"

    cd $rawDataPath

    mv $mcrtcLog originalLog.bin
    $scriptsPath/routine_scripts/lightenBin.sh originalLog.bin controllerLog.bin "t" "qIn" "JointSensor*" "ground_Default*" "qOut*" "FloatingBase_*" "Accelerometer_*" "tauIn*" "RightFootForceSensor*" "LeftFootForceSensor*" "LeftHandForceSensor*" "RightHandForceSensor*" "alphaIn*" "ff*" "perf_GlobalRun"

    heavy_log=true
  fi
fi


############################ Handling mc_rtc's log ############################


if [ -f "$logReplayCSV" ] && [[ "$runFromZero" == "false" ]]; then
    echo "The csv file of the replay with the observers has been found."
else
    if [ -f "$logReplayBin" ] && [[ "$runFromZero" == "false" ]]; then
        cd $scriptsPath
        echo "The bin file of the replay with the observers has been found. Removing useless columns."

        eval $scriptsPath/routine_scripts/lightenBin.sh $outputDataPath/logReplay.bin $outputDataPath/logReplay.bin $(python lightenOutputBin.py "$projectPath" "$outputDataPath/logReplay.bin")

        cd $outputDataPath
        echo " Converting to csv."
        mc_bin_to_log logReplay.bin 
    else
        if [ -f "$mcrtcLog" ]; then
            echo "The log file of the controller was found. Replaying the log with the observer."
            if ! grep -q -E "^\s*update: true\s*$" "$replay_yaml"; then
                echo "The pipeline needs at least one estimator to be used with update: true, please modify the "Passthrough.yaml" file accordingly."
                exit
            fi
            if grep -v '^#' $mc_rtc_yaml | grep "Plugins" | grep -v "MocapAligner"; then
                    echo "The plugin MocapAligner conflicts with another plugin in $mc_rtc_yaml. Please remove the conflicting plugin or add manually MocapAligner to the existing list."
                    exit
            fi

            if [ ! -f "$mocapPlugin_yaml" ]; then
                mkdir -p $HOME/.config/mc_rtc/plugins 
                touch $mocapPlugin_yaml
            fi
            
            if grep -v '^#' $mocapPlugin_yaml | grep -q "bodyName"; then
                sed -i "s/bodyName:.*/bodyName: $bodyName/" $mocapPlugin_yaml
            else
                if [ -s $mocapPlugin_yaml ]; then
                    sed -i "1ibodyName: $bodyName" "$mocapPlugin_yaml"
                else
                    echo "bodyName: $bodyName" > $mocapPlugin_yaml
                fi
            fi

            # Plugins the replay needs, MocapAligner being the mandatory one.
            replayPlugins="MocapAligner"
            if $useHartley; then
                replayPlugins="$replayPlugins, HartleyIEKF"
            fi
            if $useNoisySensors; then
                replayPlugins="$replayPlugins, NoisySensors"
            fi

            # Plugins we add to $mc_rtc_yaml and must remove once the replay is over.
            addedPlugins=()

            pluginWasActivated=true
            if ! grep -v '^#' $mc_rtc_yaml | grep -q "MocapAligner"; then
                pluginWasActivated=false
                echo "The plugin MocapAligner was not activated. Activating it for the replay (Plugins: [$replayPlugins])."
                sed -i "1i\Plugins: [$replayPlugins]" "$mc_rtc_yaml"
            else
                # MocapAligner is already enabled: append the other required
                # plugins to the existing list rather than overwriting it.
                for plugin in HartleyIEKF NoisySensors; do
                    case ", $replayPlugins," in
                        *", $plugin,"*) ;;
                        *) continue;;
                    esac
                    if ! grep -v '^#' $mc_rtc_yaml | grep -q "$plugin"; then
                        echo "Adding the plugin $plugin to $mc_rtc_yaml for the replay."
                        sed -i "0,/^\([[:space:]]*Plugins:[[:space:]]*\[[^]]*\)\]/s//\1, $plugin]/" $mc_rtc_yaml
                        addedPlugins+=("$plugin")
                    fi
                done
            fi
            
            
            sed -i "/^\([[:space:]]*firstRun: \).*/s//\1"true"/" $replay_yaml
            mc_rtc_ticker --no-sync --replay-outputs -e -l $mcrtcLog
            cd /tmp
            LOG=$(find . -maxdepth 1 -type f -readable -name "mc-control*Passthrough*.bin" ! -name "*latest*" | sort | tail -1)
            echo "Copying the replay's bin file ($LOG) to the output_data folder as logReplay.bin"
            mv $LOG $logReplayBin

            cd $scriptsPath
            echo "Removing useless columns from the replayed log."

            eval $scriptsPath/routine_scripts/lightenBin.sh $outputDataPath/logReplay.bin $outputDataPath/logReplay.bin $(python lightenOutputBin.py "$projectPath" "$outputDataPath/logReplay.bin")

            cd $outputDataPath
            
            mc_bin_to_log logReplay.bin
            cd $cwd

            if ! $pluginWasActivated; then
                sed -i '1d' $mc_rtc_yaml
            elif (( ${#addedPlugins[@]} > 0 )); then
                for plugin in "${addedPlugins[@]}"; do
                    sed -i "0,/^\([[:space:]]*Plugins:[[:space:]]*\[[^]]*\), $plugin\]/s//\1]/" $mc_rtc_yaml
                done
            fi
        else
            echo "The log file of the controller does not exist or is not named as expected. Expected: $mcrtcLog."
            exit
        fi
    fi
fi


############################ Handling mocap's data ############################

echo "WESH2"

cd $cwd

HartleyOutputCSV="$outputDataPath/HartleyOutputCSV.csv" 
if [[ "$runFromZero" == "false" ]] && [[ -f "$HartleyOutputCSV" ]]; then
    echo "The csv file containing the results of Hartley's observer already exists. Working with this data."
else
    echo "WESH3"
    if $useHartley; then
        echo "WESH4"
        # HARTLEY_DIR can be set to skip the search. Otherwise try locate (fast
        # but its database usually does not index $HOME), then fall back to find.
        if [ -n "$HARTLEY_DIR" ]; then
            hartleyRoutine="$HARTLEY_DIR/runLogsRoutine.sh"
        else
            # The '|| true' are required: routine.sh runs under 'set -e', and a
            # command substitution whose pipeline fails (grep matching nothing)
            # kills the whole routine silently.
            hartleyRoutine=$(locate -b '\runLogsRoutine.sh' 2>/dev/null | grep Hartley | head -1 || true)
            if [ -z "$hartleyRoutine" ]; then
                hartleyRoutine=$(find "$HOME" -maxdepth 6 -type f -name 'runLogsRoutine.sh' -path '*Hartley*' -print -quit 2>/dev/null || true)
            fi
        fi

        if [ ! -f "$hartleyRoutine" ]; then
            echo "Could not find Hartley's runLogsRoutine.sh. Set HARTLEY_DIR to the directory containing it, or disable Use_HartleyIEKF in the project configuration."
            exit 1
        fi

        hartleyDir=$(dirname "$hartleyRoutine")
        echo "Using Hartley's routine from $hartleyDir."

        cd "$hartleyDir"
        mkdir -p data
        if find data -mindepth 1 -maxdepth 1 | read; then
            rm data/*
        fi

        cp "/tmp/HartleyInput.txt" "data/HartleyInput.txt"

        cd "$hartleyDir"
        ./runLogsRoutine.sh "anything"

        cd $cwd

        cp "$hartleyDir/data/HartleyOutput.csv" $HartleyOutputCSV
        echo "WESH5"
    fi
fi
echo "WESH3"

if [[ "$runFromZero" == "false" ]] && [[ -f "$lightData" ]]; then
    echo "The light version of the observer's data has already been extracted. Using the existing data."
else
    cd $scriptsPath
    echo "Starting the extraction of the light version of the observer's data."
    python extractLightReplayVersion.py "$projectPath"
    echo "Extraction of the light version of the observer's data completed."
    runScript=true
fi

cd $rawDataPath

if [ ! -f "$outputDataPath/perf_GlobalRun_log.csv" ]; then
    mc_bin_utils extract "controllerLog.bin" "perf_GlobalRun_log.bin" --keys perf_GlobalRun
    mc_bin_to_log perf_GlobalRun_log.bin $outputDataPath/perf_GlobalRun_log.csv
    rm perf_GlobalRun_log.bin
fi


if [ -f "$outputDataPath/repairedSkipped_mc_rtc_iters.csv" ] && [[ "$runFromZero" == "false" ]] ; then
    if $debug; then
        echo "Do you want to correct again the skipped iterations in the mc_rtc log?"
        select rerun in "No" "Yes"; do
        case $rerun in
            Yes ) echo "Repairing the missing iterations"; cd $scriptsPath; python repair_mc_rtc_skipped_iters.py "$timeStep" "$projectPath"; echo "Finished repairing the missing iterations."; break;;
            No ) break;;
        esac
        done
    else
        echo "The skipped iterations in mc_rtc log have already been repaired."
    fi
else
    echo "Repairing the missing iterations"
    cd $scriptsPath
    python repair_mc_rtc_skipped_iters.py "$timeStep" "$projectPath"
    echo "Finished repairing the missing iterations."
    runScript=true
fi


cd $scriptsPath

heavy_log=true
python initialize_datas.py "$timeStep" "$projectPath" "$heavy_log"


if [ -f "$resampledMocapData" ]; then
    if $debug; then
        echo "Do you want to run again the mocap data's resampling with the dynamic plots?"
        select rerunResample in "No" "Yes"; do
        case $rerunResample in
            Yes ) cd $scriptsPath; python resampleMocapAndExtractPose.py "$displayLogs" "y" "$projectPath"; break;;
            No ) break;;
        esac
        done
    else
        echo "The mocap's data has already been resampled. Using the existing data."
    fi
else
    cd $scriptsPath

    echo "Starting the resampling of the mocap's signal."
    python resampleMocapAndExtractPose.py "$displayLogs" "y" "$projectPath"
    echo "Resampling of the mocap's signal completed."
    runScript=true
fi

cd $cwd

if [ -f "$synchronizedObserversMocapData" ] && ! $runScript && [[ "$runFromZero" == "false" ]]; then
    if $debug; then
        echo "Do you want to run again the temporal data alignement with the dynamic plots?"
        select rerunResample in "No" "Yes"; do
            case $rerunResample in
                Yes )   cd $scriptsPath;
                        python crossCorrelation.py "$displayLogs" "y" "$projectPath"; break;;
                No ) break;;
            esac
        done
    else 
        echo "The temporally aligned version of the mocap's data already exists. Using the existing data."
    fi
else
    echo "Starting the cross correlation for temporal data alignement."
    cd $scriptsPath
    python crossCorrelation.py "$displayLogs" "y" "$projectPath"
    echo "Temporal alignement of the mocap's data with the observer's data completed."
    runScript=true
fi

cd $cwd

finalDataCSV="$outputDataPath/finalDataCSV.csv"

if [ -f "$finalDataCSV" ] && ! $runScript && [[ "$runFromZero" == "false" ]]; then
    if $debug; then
        echo "Do you want to run again the spatial data alignement with the dynamic plots?"
        select rerunResample in "No" "Yes"; do
        case $rerunResample in
            Yes )   echo "Please enter the time at which you want the pose of the mocap and the one of the observer must match: "
                    read matchTime
                    cd $scriptsPath
                    python matchInitPose.py "$matchTime" "$displayLogs" "y" "$projectPath" ; break;;
            No ) break;;
        esac
        done
    fi
else
    # Prompt the user for input
    echo "Matching the initial pose of the mocap with the one of the observer."
    cd $scriptsPath
    python matchInitPose.py 0 "$displayLogs" "y" "$projectPath"
    echo "Matching of the pose of the mocap with the pose of the observer completed."
fi

cd $cwd


############################ Replaying the final result ############################


if [[ "$runFromZero" == "false" ]]; then
    echo "Do you want to plot the resulting estimations?"
    select plotResults in "Yes" "No"; do
        case $plotResults in
            Yes ) plotResults=true; break;;
            No )  plotResults=false; break;;
        esac
    done  
fi

if [[ "$runFromZero" == "false" ]]; then
    echo "Do you want to compute the evalutation metrics for all the estimators?"
    select computeMetrics in "No" "Yes"; do
        case $computeMetrics in
            No )  computeMetrics=false; break;;
            Yes ) computeMetrics=true; break;;
        esac
    done
else
    computeMetrics=true
fi

if [[ "$computeMetrics" == "true" ]]; then
    if [[ "$runFromZero" == "false" ]] && [[ -d "$outputDataPath/evals/" ]]; then
        echo "It seems that the estimator evaluation metrics have already been computed, do you want to compute them again?"
        select recomputeMetrics in "No" "Yes"; do
            case "$recomputeMetrics" in
                No )    cd "$scriptsPath"
                        python plotAndFormatResults.py "$plotResults" "$projectPath" "False"; 
                        break;;
                Yes )   cd "$scriptsPath"; source routine_scripts/computeMetrics.sh;
                        break;;
            esac
        done   
        
    else
        echo "Formatting the results to evaluate the performances of the observers."; 
        cd $scriptsPath
        source routine_scripts/computeMetrics.sh
        echo "Formatting finished."; 
    fi
elif $plotResults; then
    echo "Plotting the observer results."; 
    cd "$scriptsPath"
    python plotAndFormatResults.py "$plotResults" "$projectPath" "False"; 
fi


cd $cwd

if [[ "$runFromZero" == "false" ]]; then 
    echo "Do you want to replay the log with the obtained mocap's data?"
    select replayWithMocap in "Yes" "No"; do
        case $replayWithMocap in
            Yes ) mcrtcLog="$rawDataPath/controllerLog.bin"; sed -i "/^\([[:space:]]*firstRun: \).*/s//\1"false"/" $replay_yaml; sed -i "/^\([[:space:]]*mocapBodyName: \).*/s//\1"$bodyName"/" $replay_yaml; mc_rtc_ticker --no-sync --replay-outputs -e -l $mcrtcLog; break;;
            No ) break;;
        esac
    done    
fi

echo "The pipeline finished without any issue. If you are not satisfied with the result, please re-run the scripts one by one and help yourself with the logs for the debug. Please also make sure that the time you set for the matching of the mocap and the observer is correct."
