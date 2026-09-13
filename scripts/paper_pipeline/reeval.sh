#!/bin/bash
# Rebuild evals/<observer>/ from the freshly reformatted trajectories and re-run the RPG
# evaluation.  Mirrors computeMetrics.sh, minus the plotting, and copies instead of moving so
# the formatted files stay in place.
set -u
cd /home/arnaud/devel/src/data_manipulation_tools || exit 1
project=$1; shift
observers=("$@")
out="Projects/$project/output_data"
lengths=$(.venv/bin/python -c "
import yaml,sys
print(' '.join(str(v) for v in yaml.safe_load(open('Projects/$project/projectConfig.yaml'))['predefined_sublengths']))")
for obs in "${observers[@]}"; do
  dir="$out/evals/$obs"
  mkdir -p "$dir/saved_results/traj_est/cached"
  printf 'align_type: posyaw\nalign_num_frames: -1\n' > "$dir/eval_cfg.yaml"
  cp "$out/formattedMocap_Traj.txt" "$dir/stamped_groundtruth.txt"
  cp "$out/formatted_${obs}_Traj.txt" "$dir/stamped_traj_estimate.txt"
  rm -f "$dir/saved_results/traj_est/cached/"*.pickle
  echo "--- $project / $obs  (sublengths $lengths)"
  .venv/bin/python rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py \
      "$dir" --recalculate_errors --no_plot --estimator_name "$obs" \
      --predefined_sublengths $lengths 2>&1 | tail -3
done
