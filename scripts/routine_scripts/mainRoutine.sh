############################ Configuration files test ############################


if [ ! -f "$replay_yaml" ]; then
    echo "The scripts excepts to find a configuration file named $replay_yaml."
    exit
else
    if ! grep -q "MocapVisualizer" $replay_yaml || ! grep -q "firstRun:" $replay_yaml; then
        # Execute your action here if both patterns are found
        echo "Please add the MocapVisualizer observer to the list of the observers in $replay_yaml."
        exit
    fi
    if ! (grep -v '^#' $replay_yaml | grep -q "firstRun:"); then
        echo "Please add the boolean firstRun to the configuration of the MocapVisualizer in $replay_yaml."
        exit
    fi
    if ! (grep -v '^#' $replay_yaml | grep -q "projectName:"); then
        echo "Please add the variable projectName to the configuration of the MocapVisualizer in $replay_yaml."
        exit
    fi
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
        awk -i inplace -v robot="$main_robot" 'FNR==1 {print "EnabledRobot:", robot}1' $projectConfig
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
        awk -i inplace -v body="$body" 'FNR==1 {print "EnabledBody:", body}1' $projectConfig;
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
        awk -i inplace -v body="$body" 'FNR==1 {print "Body_vel_eval:", body}1' $projectConfig;
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
    mc_bin_utils originalLog.bin controllerLog --keys "t" "qIn" "JointSensor*" "ground_Default*" "qOut*" "FloatingBase_*" "Accelerometer_*" "tauIn*" "RightFootForceSensor*" "LeftFootForceSensor*" "LeftHandForceSensor*" "RightHandForceSensor*" "alphaIn*" "ff*" "perf_GlobalRun"

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

        eval mc_bin_utils extract $outputDataPath/logReplay.bin $outputDataPath/logReplay --keys $(python lightenOutputBin.py "$projectPath")

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
                    awk -i inplace -v name="$bodyName" 'FNR==1 {print "bodyName:", name}1' $mocapPlugin_yaml
                else
                    echo "bodyName: $bodyName" > $mocapPlugin_yaml
                fi
            fi

            pluginWasActivated=true
            if ! grep -v '^#' $mc_rtc_yaml | grep -q "MocapAligner"; then
                pluginWasActivated=false
                echo "The plugin MocapAligner was not activated. Activating it for the replay."
                
                if $useHartley; then
                    awk -i inplace 'FNR==1 {print "Plugins: [MocapAligner, HartleyIEKF] \n"}1' $mc_rtc_yaml
                else
                    awk -i inplace 'FNR==1 {print "Plugins: [MocapAligner] \n"}1' $mc_rtc_yaml
                fi
            else
                if ! grep -v '^#' $mc_rtc_yaml | grep -q "HartleyIEKF"; then
                    if $useHartley; then
                        grep -v '^#' $mc_rtc_yaml | grep -q "MocapAligner" | sed -i "s/Plugins:.*/Plugins: [MocapAligner, HartleyIEKF]/" $mc_rtc_yaml
                    fi
                fi
            fi
            
            
            sed -i "/^\([[:space:]]*firstRun: \).*/s//\1"true"/" $replay_yaml
            mc_rtc_ticker --no-sync --replay-outputs -e -l $mcrtcLog
            cd /tmp
            LOG=$(find -iname "mc-control*" | grep "Passthrough" | grep -v "latest" | grep ".bin" | sort | tail -1)
            echo "Copying the replay's bin file ($LOG) to the output_data folder as logReplay.bin"
            mv $LOG $logReplayBin

            cd $scriptsPath
            echo "Removing useless columns from the replayed log."

            eval mc_bin_utils extract $outputDataPath/logReplay.bin $outputDataPath/logReplay --keys $(python lightenOutputBin.py "$projectPath")

            cd $outputDataPath
            
            mc_bin_to_log logReplay.bin
            cd $cwd

            if ! $pluginWasActivated; then
                sed -i '1d' $mc_rtc_yaml
            fi

        else
            echo "The log file of the controller does not exist or is not named as expected. Expected: $mcrtcLog."
            exit
        fi
    fi
fi


############################ Handling mocap's data ############################



cd $cwd

HartleyOutputCSV="$outputDataPath/HartleyOutputCSV.csv" 
if [[ "$runFromZero" == "false" ]] && [[ -f "$HartleyOutputCSV" ]]; then
    echo "The csv file containing the results of Hartley's observer already exists. Working with this data."
else
    if $useHartley; then
        hartleyRoutine=$(locate runLogsRoutine.sh | grep Hartley)
        hartleyDir=$(dirname "$hartleyRoutine")

        cd "$hartleyDir"
        if find data -mindepth 1 -maxdepth 1 | read; then
            rm data/*
        fi

        cp "/tmp/HartleyInput.txt" "data/HartleyInput.txt"
        
        cd "$hartleyDir"
        ./runLogsRoutine.sh "anything"

        cd $cwd

        cp "$hartleyDir/data/HartleyOutput.csv" $HartleyOutputCSV
    fi
fi


if [[ "$runFromZero" == "false" ]] && [[ -f "$lightData" ]]; then
    echo "The light version of the observer's data has already been extracted. Using the existing data."
else
    cd $scriptsPath
    echo "Starting the extraction of the light version of the observer's data."
    python extractLightReplayVersion.py "$projectPath"
    echo "Extraction of the light version of the observer's data completed."
    python extractLightReplayVersion.py "$projectPath"
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

if [ -f "$synchronizedMocapLimbData" ] && ! $runScript && [[ "$runFromZero" == "false" ]]; then
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

observerResultsCSV="$outputDataPath/observerResultsCSV.csv"

if [ -f "$observerResultsCSV" ] && ! $runScript && [[ "$runFromZero" == "false" ]]; then
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
