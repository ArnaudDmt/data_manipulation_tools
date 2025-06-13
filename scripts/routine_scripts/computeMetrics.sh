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
        observers=("KO" "KO_APC" "KO_ASC" "KO_ZPC" "KOWithoutWrenchSensors"  "Tilt" "Control" "Vanyte" "RI-EKF") 

        mv "$outputDataPath/mocap_x_y_z_traj.pickle" "$outputDataPath/evals/mocap_x_y_z_traj.pickle"
        mv "$outputDataPath/mocap_loc_vel.pickle" "$outputDataPath/evals/mocap_loc_vel.pickle"

        for observer in "${observers[@]}"; do
            formattedTrajVar="formatted_${observer}_Traj.txt"
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



cd $cwd/scripts
echo "Starting the formatting for $projectName."; 
python plotAndFormatResults.py "$timeStep" $plotResults "$projectPath" "True"; 
echo "Formatting for $projectName finished."; 
compute_metrics
echo "Computation of the metrics for $projectName finished."; 