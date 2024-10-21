#!/bin/zsh


compute_metrics() {
    cd $cwd

    mocapFormattedResults="$outputDataPath/formattedMocapTraj.txt"
    if [ -f "$mocapFormattedResults" ]; then
        mkdir -p "$outputDataPath/evals"
        if [ -f "$outputDataPath/formattedKoTraj.txt" ]; then
            mkdir -p "$outputDataPath/evals/KineticsObserver"
            if ! [ -f "$outputDataPath/evals/KineticsObserver/eval_cfg.yaml" ]; then
                touch "$outputDataPath/evals/KineticsObserver/eval_cfg.yaml"
                echo "align_type: none" >> "$outputDataPath/evals/KineticsObserver/eval_cfg.yaml"
                echo "align_num_frames: -1" >> "$outputDataPath/evals/KineticsObserver/eval_cfg.yaml"
            fi
            cp $mocapFormattedResults "$outputDataPath/evals/KineticsObserver/stamped_groundtruth.txt"
            mv "$outputDataPath/formattedKoTraj.txt" "$outputDataPath/evals/KineticsObserver/stamped_traj_estimate.txt"
            python rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py "$outputDataPath/evals/KineticsObserver" --recalculate_errors --estimator_name "Kinetics Observer" &
        fi
        if [ -f "$outputDataPath/formattedVanyteTraj.txt" ]; then
            mkdir -p "$outputDataPath/evals/Vanyte"
            if ! [ -f "$outputDataPath/evals/Vanyte/eval_cfg.yaml" ]; then
                touch "$outputDataPath/evals/Vanyte/eval_cfg.yaml"
                echo "align_type: none" >> "$outputDataPath/evals/Vanyte/eval_cfg.yaml"
                echo "align_num_frames: -1" >> "$outputDataPath/evals/Vanyte/eval_cfg.yaml"
            fi
            cp $mocapFormattedResults "$outputDataPath/evals/Vanyte/stamped_groundtruth.txt"
            mv "$outputDataPath/formattedVanyteTraj.txt" "$outputDataPath/evals/Vanyte/stamped_traj_estimate.txt"
            python rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py "$outputDataPath/evals/Vanyte" --recalculate_errors --estimator_name "Vanyte" &
        fi
        if [ -f "$outputDataPath/formattedTiltTraj.txt" ]; then
            mkdir -p "$outputDataPath/evals/Tilt"
            if ! [ -f "$outputDataPath/evals/Tilt/eval_cfg.yaml" ]; then
                touch "$outputDataPath/evals/Tilt/eval_cfg.yaml"
                echo "align_type: none" >> "$outputDataPath/evals/Tilt/eval_cfg.yaml"
                echo "align_num_frames: -1" >> "$outputDataPath/evals/Tilt/eval_cfg.yaml"
            fi
            cp $mocapFormattedResults "$outputDataPath/evals/Tilt/stamped_groundtruth.txt"
            mv "$outputDataPath/formattedTiltTraj.txt" "$outputDataPath/evals/Tilt/stamped_traj_estimate.txt"
            python rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py "$outputDataPath/evals/Tilt" --recalculate_errors --estimator_name "Tilt" &
        fi
        if [ -f "$outputDataPath/formattedControllerTraj.txt" ]; then
            mkdir -p "$outputDataPath/evals/Controller"
            if ! [ -f "$outputDataPath/evals/Controller/eval_cfg.yaml" ]; then
                touch "$outputDataPath/evals/Controller/eval_cfg.yaml"
                echo "align_type: none" >> "$outputDataPath/evals/Controller/eval_cfg.yaml"
                echo "align_num_frames: -1" >> "$outputDataPath/evals/Controller/eval_cfg.yaml"
            fi
            cp $mocapFormattedResults "$outputDataPath/evals/Controller/stamped_groundtruth.txt"
            mv "$outputDataPath/formattedControllerTraj.txt" "$outputDataPath/evals/Controller/stamped_traj_estimate.txt"
            python rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py "$outputDataPath/evals/Controller" --recalculate_errors --estimator_name "Controller" &
        fi
        if [ -f "$outputDataPath/formattedHartleyTraj.txt" ]; then
            mkdir -p "$outputDataPath/evals/Hartley"
            if ! [ -f "$outputDataPath/evals/Hartley/eval_cfg.yaml" ]; then
                touch "$outputDataPath/evals/Hartley/eval_cfg.yaml"
                echo "align_type: none" >> "$outputDataPath/evals/Hartley/eval_cfg.yaml"
                echo "align_num_frames: -1" >> "$outputDataPath/evals/Hartley/eval_cfg.yaml"
            fi
            cp $mocapFormattedResults "$outputDataPath/evals/Hartley/stamped_groundtruth.txt"
            mv "$outputDataPath/formattedHartleyTraj.txt" "$outputDataPath/evals/Hartley/stamped_traj_estimate.txt"
            python rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py "$outputDataPath/evals/Hartley"  --recalculate_errors --estimator_name "Hartley" &
        fi
        rm $mocapFormattedResults
    fi

    # Wait for all background processes to finish
    wait
}


if $compute_metrics_only; then
    onPress=true

    # Change to your project directory
    cd Projects

    # Get folders containing the specific file, remove './' prefix, and sort them alphabetically
    folders=($(find . -type d -exec test -e '{}/output_data/observerResultsCSV.csv' \; -print | sed 's|^\./||' | grep -v '^$' | sort))

    # Check if no folders found
    if [[ ${#folders[@]} -eq 0 ]]; then
    echo "No folders contain the specified file."
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

