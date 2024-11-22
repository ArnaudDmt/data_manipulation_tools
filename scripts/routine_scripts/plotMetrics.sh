#!/bin/zsh

cd $cwd

python scripts/plots_scripts/generate_metrics_plots.py --exps_to_merge "${selected_folders[@]}" 