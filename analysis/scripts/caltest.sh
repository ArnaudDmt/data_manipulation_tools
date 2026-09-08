#!/bin/bash
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,HRP5_MultiContact_4,KO_TRO2024_RHPS1_1,KO_TRO2024_RHPS1_2,KO_TRO2024_RHPS1_3,KO_TRO2024_RHPS1_4,KO_TRO2024_RHPS1_5,KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3
echo "=== PREPARE (calibrated config) ==="
../.venv/bin/python kinetics_eval.py --projects "$P" \
  --observer-config "$SP/cal/MCKineticsObserver.yaml" prepare --force || { echo "=== PREPARE FAILED ==="; exit 1; }
echo "=== RUN ==="
timeout 1800 ../.venv/bin/python kinetics_eval.py --projects "$P" \
  --covariance-overlay "$SP/tuned_a1_posx30.yaml" --observer-config "$SP/cal/MCKineticsObserver.yaml" \
  run --label "C_wrenchcal" --no-plots --no-open --no-latest || echo "=== RUN FAILED ==="
echo "=== CAL DONE ==="
