#!/bin/bash
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=KO_TRO2024_RHPS1_1
echo "=== BASELINE (installed config, no calibration) ==="
../.venv/bin/python kinetics_eval.py --projects $P prepare --force >/dev/null 2>&1 || echo PREP_FAIL
../.venv/bin/python kinetics_eval.py --projects $P --covariance-overlay "$SP/tuned_a1_posx30.yaml" \
  run --label WB_base --no-open --no-latest 2>&1 | tail -2
echo "=== CALIBRATED ==="
../.venv/bin/python kinetics_eval.py --projects $P --observer-config "$SP/cal/MCKineticsObserver.yaml" \
  prepare --force >/dev/null 2>&1 || echo PREP_FAIL
../.venv/bin/python kinetics_eval.py --projects $P --covariance-overlay "$SP/tuned_a1_posx30.yaml" \
  --observer-config "$SP/cal/MCKineticsObserver.yaml" run --label WB_cal --no-open --no-latest 2>&1 | tail -2
echo "=== WB DONE ==="
