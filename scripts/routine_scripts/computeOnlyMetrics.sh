#!/bin/zsh

run_analysis() {
    local observerName=$1
    local num_samples_rel_error=$2
    shift 2  # Remove the first two arguments

    local predefined_sublengths=("$@")  # Remaining arguments are the array

    cmd="python rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py \"$outputDataPath/evals/$observerName\" --recalculate_errors --no_plot --estimator_name \"$observerName\""
    
    if [ ${#predefined_sublengths[@]} -gt 0 ]; then
        cmd="$cmd --predefined_sublengths ${predefined_sublengths[@]}"
    fi
    
    if [[ -n "$num_samples_rel_error" && "$num_samples_rel_error" =~ ^[0-9]+$ ]]; then
        cmd="$cmd --num_samples_rel_error $num_samples_rel_error"
    fi

    eval "$cmd &"
}

compute_metrics() {
    cd $cwd
    
    mocapFormattedResults="$outputDataPath/formattedMocap_Traj.txt"
    if [ -f "$mocapFormattedResults" ]; then
        predefined_sublengths=($(yq eval '.predefined_sublengths[]' $projectConfig))
        
        num_samples_rel_error=($(yq eval '.num_samples_rel_error' $projectConfig))
        
        if [ ${#predefined_sublengths[@]} -eq 0 ]; then
            echo "Please give the list of lengths of the sub-trajectories for the relative error in the file $projectConfig"
            exit
        fi
        mkdir -p "$outputDataPath/evals"

        # Function to clean up background jobs on exit
        cleanup() {
            echo "Stopping background processes..."
            # Stops all the background processes
            kill %${(k)^jobstates}
            wait
            exit
        }

        # Set the trap for SIGINT (Ctrl+C)
        trap cleanup SIGINT
        
        # Define an array of observer names
        observers=("KineticsObserver" "KO_APC" "KO_ASC" "KO_Disabled" "Vanyte" "Tilt" "Controller" "Hartley")

        mv "$outputDataPath/mocap_x_y_z_traj.pickle" "$outputDataPath/evals/mocap_x_y_z_traj.pickle"
        mv "$outputDataPath/mocap_loc_vel.pickle" "$outputDataPath/evals/mocap_loc_vel.pickle"

        for observer in "${observers[@]}"; do
            formattedTrajVar="formatted${observer}_Traj.txt"
            if [ -f "$outputDataPath/$formattedTrajVar" ]; then
                mkdir -p "$outputDataPath/evals/$observer/saved_results/traj_est/cached"
                if ! [ -f "$outputDataPath/evals/$observer/eval_cfg.yaml" ]; then
                    touch "$outputDataPath/evals/$observer/eval_cfg.yaml"
                    echo "align_type: posyaw" >> "$outputDataPath/evals/$observer/eval_cfg.yaml"
                    echo "align_num_frames: -1" >> "$outputDataPath/evals/$observer/eval_cfg.yaml"
                fi

                cp $mocapFormattedResults "$outputDataPath/evals/$observer/stamped_groundtruth.txt"
                mv "$outputDataPath/$formattedTrajVar" "$outputDataPath/evals/$observer/stamped_traj_estimate.txt"
                mv "$outputDataPath/${observer}_x_y_z_traj.pickle" "$outputDataPath/evals/$observer/saved_results/traj_est/cached/x_y_z_traj.pickle"
                mv "$outputDataPath/${observer}_loc_vel.pickle" "$outputDataPath/evals/$observer/saved_results/traj_est/cached/loc_vel.pickle"
                

                # Call the run_analysis function for each observer
                run_analysis "$observer" "$num_samples_rel_error" "${predefined_sublengths[@]}"
            fi
        done
        
        rm $mocapFormattedResults
    else
        echo "Cannot compute the metrics without the ground truth"
        exit
    fi

    # Wait for all background processes to finish
    wait
    echo "Metrics computation finished"
}


if $compute_metrics_only; then
    onPress=true

    # Change to your project directory
    cd Projects

    # Get folders containing the specific file, remove './' prefix, and sort them alphabetically
    folders=($(find . -type d -exec test -e '{}/output_data/observerResultsCSV.csv' \; -print | sed 's|^\./||' | grep -v '^$' | sort))

    # Check if no folders found
    if [[ ${#folders[@]} -eq 0 ]]; then
        exit 1
    fi

    selected_folders=()

    # Function to select folders
    select_folders() {
    local folder_count=${#folders[@]}

    while true; do
        clear
        echo "Select folders (Type the numbers to select/unselect, 'y' to confirm. If you want to select folders above 9, please press 't' to type the number and press Enter):"
        # Display the folders with selection indication
        for ((i = 1; i <= folder_count; i++)); do
        folder="${folders[$((i))]}"
        if [[ " ${selected_folders[@]} " == *" $folder "* ]]; then
            printf "\033[31m* %d: %s\033[0m\n" "$i" "$folder"  # Selected folders in red
        else
            printf "  %d: %s\n" "$i" "$folder"  # Unselected folders
        fi
        done

        # Read user input for folder selection
        if $onPress; then
            read -rs -k1 input  # Read a single character without echoing it
        else
            read input
        fi
        # If input is 'y', confirm selection
        if [[ $input == "y" ]]; then
            break
        fi
        # If input is 'y', confirm selection
        if [[ $input == "t" ]]; then
            onPress=false
            continue
        fi
        if [[ -n $input && $input =~ ^[0-9]+$ ]]; then
            index=$input
        else
            continue
        fi

        # Toggle selection if the input corresponds to a valid index
        if (( 1 <= index && index <= folder_count )); then
        folder="${folders[$((index))]}"
        if [[ " ${selected_folders[@]} " == *" $folder "* ]]; then
            # Unselect
            selected_folders=(${(@)selected_folders:#$folder})
        else
            # Select
            selected_folders+=("$folder")
        fi
        fi
    done

    # Print final selected folders
    echo "Final selected folders:"
    for folder in "${selected_folders[@]}"; do
        echo "$folder"
    done
    }



    echo "Use the timestep defined in $mc_rtc_yaml ?"
    select useMainConfRobot in "Yes" "No"; do
        case $useMainConfRobot in
            Yes ) 
                timeStep=$( grep 'Timestep:' $mc_rtc_yaml | grep -v '^#' | sed 's/Timestep: //'); break;;
            No ) 
                echo "Please enter the timestep of the controller in milliseconds: "
                read timeStep ; break;;
        esac
    done


    # Call the function
    select_folders

    cd $cwd

    for project in "${selected_folders[@]}"; do
        ## Relative paths initialization ##
        # main folders
        projectPath="$cwd/Projects/$project"
        rawDataPath="$projectPath/raw_data"
        outputDataPath="$projectPath/output_data"
        scriptsPath="/scripts"

        #main files
        projectConfig="$projectPath/projectConfig.yaml"

        # files of the resulting data after each step
        resampledMocapData="$outputDataPath/resampledMocapData.csv"
        lightData="$outputDataPath/lightData.csv"
        realignedMocapLimbData="$outputDataPath/realignedMocapLimbData.csv"
        resultMocapLimbData="$outputDataPath/resultMocapLimbData.csv"

        ## Formatting ##
        cd $cwd/scripts
        echo "Starting the formatting for $project."; 
        python plotAndFormatResults.py "$timeStep" "False" "$projectPath" "True"; 
        echo "Formatting for $project finished."; 
        compute_metrics
        echo "Computation of the metrics for $project finished."; 
    done
    exit
fi

