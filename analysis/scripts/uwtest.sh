#!/bin/bash
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,HRP5_MultiContact_4,KO_TRO2024_RHPS1_1,KO_TRO2024_RHPS1_2,KO_TRO2024_RHPS1_3,KO_TRO2024_RHPS1_4,KO_TRO2024_RHPS1_5,KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3
for cfg in tuned_uw1e-2 tuned_uw1e-4; do
  echo "=== RUN $cfg ==="
  timeout 900 ../.venv/bin/python kinetics_eval.py --projects "$P" \
    --covariance-overlay "$SP/$cfg.yaml" run --label "U_$cfg" --no-plots --no-open --no-latest \
    || echo "=== FAIL $cfg ==="
done
echo "=== UW DONE ==="
