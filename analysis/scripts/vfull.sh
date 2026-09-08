#!/bin/bash
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,HRP5_MultiContact_4,HRP5P_LongWalk,KO_TRO2024_RHPS1_1,KO_TRO2024_RHPS1_2,KO_TRO2024_RHPS1_3,KO_TRO2024_RHPS1_4,KO_TRO2024_RHPS1_5,KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3
timeout 1800 ../.venv/bin/python kinetics_eval.py --projects "$P" prepare || exit 1
echo "=== PREPARE DONE ==="
for cfg in tuned_a0.0 tuned_a1_posx100 tuned_yaw100 tuned_yaw100_rp; do
  echo "=== RUN $cfg ==="
  timeout 2700 ../.venv/bin/python kinetics_eval.py --projects "$P" \
    --covariance-overlay "$SP/$cfg.yaml" run --label "V_$cfg" --no-plots --no-open --no-latest \
    || echo "=== FAIL $cfg ==="
done
echo "=== VFULL DONE ==="
