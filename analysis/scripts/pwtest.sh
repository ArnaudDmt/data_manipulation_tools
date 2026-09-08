#!/bin/bash
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,HRP5_MultiContact_4,KO_TRO2024_RHPS1_1,KO_TRO2024_RHPS1_2,KO_TRO2024_RHPS1_3,KO_TRO2024_RHPS1_4,KO_TRO2024_RHPS1_5,KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3
echo "=== BASELINE prepare ==="
../.venv/bin/python kinetics_eval.py --projects "$P" prepare --force 2>&1 | grep -c prepared
echo "=== BASELINE run ==="
timeout 2400 ../.venv/bin/python kinetics_eval.py --projects "$P" \
  --covariance-overlay "$SP/tuned_a1_posx30.yaml" run --label PW_base --no-open --no-latest 2>&1 | tail -1
echo "=== CALIBRATED prepare ==="
../.venv/bin/python kinetics_eval.py --projects "$P" \
  --observer-config "$SP/cal/MCKineticsObserver.yaml" prepare --force 2>&1 | grep -c prepared
echo "=== CALIBRATED run ==="
timeout 2400 ../.venv/bin/python kinetics_eval.py --projects "$P" \
  --covariance-overlay "$SP/tuned_a1_posx30.yaml" --observer-config "$SP/cal/MCKineticsObserver.yaml" \
  run --label PW_cal --no-open --no-latest 2>&1 | tail -1
echo "=== PW DONE ==="
