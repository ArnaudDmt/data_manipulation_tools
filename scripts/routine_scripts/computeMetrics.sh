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
    analysis_pid=$!
}

compute_metrics() {
    local analysis_pids=()
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
            if (( ${#analysis_pids[@]} )); then
                kill "${analysis_pids[@]}" 2>/dev/null
                wait "${analysis_pids[@]}" 2>/dev/null
            fi
            exit
        }

        # Set the trap for SIGINT (Ctrl+C)
        trap cleanup SIGINT
        
        # Observers whose metrtics are evaluated
        observers=($(yq -r '.observers[].abbreviation' observersInfos.yaml))
        echo "Observers: $observers"

        mv "$outputDataPath/mocap_x_y_z_traj.pickle" "$outputDataPath/evals/mocap_x_y_z_traj.pickle"
        mv "$outputDataPath/mocap_loc_vel.pickle" "$outputDataPath/evals/mocap_loc_vel.pickle"

        for observer in "${observers[@]}"; do
            # The mocap trajectory is formatted under a different name than the observers'.
            if [[ "$observer" == "Mocap" ]]; then
                formattedTraj="$mocapFormattedResults"
            else
                formattedTraj="$outputDataPath/formatted_${observer}_Traj.txt"
            fi
            if [ -f "$formattedTraj" ]; then
                mkdir -p "$outputDataPath/evals/$observer/saved_results/traj_est/cached"
                if ! [ -f "$outputDataPath/evals/$observer/eval_cfg.yaml" ]; then
                    touch "$outputDataPath/evals/$observer/eval_cfg.yaml"
                    echo "align_type: posyaw" >> "$outputDataPath/evals/$observer/eval_cfg.yaml"
                    echo "align_num_frames: -1" >> "$outputDataPath/evals/$observer/eval_cfg.yaml"
                fi

                cp $mocapFormattedResults "$outputDataPath/evals/$observer/stamped_groundtruth.txt"
                if [[ "$observer" == "Mocap" ]]; then
                    # Still needed by the other observers, and removed after the loop.
                    cp "$formattedTraj" "$outputDataPath/evals/$observer/stamped_traj_estimate.txt"
                else
                    mv "$formattedTraj" "$outputDataPath/evals/$observer/stamped_traj_estimate.txt"
                fi
                for pickleName in x_y_z_traj loc_vel; do
                    sourcePickle="$outputDataPath/${observer}_${pickleName}.pickle"
                    destPickle="$outputDataPath/evals/$observer/saved_results/traj_est/cached/${pickleName}.pickle"
                    if [ -f "$sourcePickle" ]; then
                        mv "$sourcePickle" "$destPickle"
                    elif [[ "$observer" == "Mocap" && -f "$outputDataPath/evals/mocap_${pickleName}.pickle" ]]; then
                        # Not exported a second time under the observer's name: reuse the ground truth's.
                        cp "$outputDataPath/evals/mocap_${pickleName}.pickle" "$destPickle"
                    else
                        # Never leave results of a previous run behind: they would be
                        # inconsistent with the freshly generated ground truth.
                        rm -f "$destPickle"
                    fi
                done


                # Call the run_analysis function for each observer
                run_analysis "$observer" "$num_samples_rel_error" "${predefined_sublengths[@]}"
                analysis_pids+=("$analysis_pid")
            fi
        done
        
        rm $mocapFormattedResults
    else
        echo "Cannot compute the metrics without the ground truth"
        exit
    fi

    # Wait only for metric workers launched above.
    if (( ${#analysis_pids[@]} )); then
        wait "${analysis_pids[@]}"
    fi
    echo "Metrics computation finished"
}



cd $cwd/scripts
echo "Starting the formatting for $projectName."; 
python plotAndFormatResults.py $plotResults "$projectPath" "True"; 
echo "Formatting for $projectName finished."; 
compute_metrics
echo "Computation of the metrics for $projectName finished."; 