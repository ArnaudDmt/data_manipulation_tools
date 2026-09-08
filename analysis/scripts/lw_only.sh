#!/bin/bash
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
timeout 2400 ../.venv/bin/python kinetics_eval.py --projects HRP5P_LongWalk \
  --covariance-overlay "$SP/tuned_a1_posx30.yaml" run --label W_LW --no-plots --no-open --no-latest \
  || echo "=== LW FAIL ==="
echo "=== LW DONE ==="
