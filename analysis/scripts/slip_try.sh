#!/bin/bash
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3,KO_TRO2024_RHPS1_1
for cfg in tuned_a1.0 tuned_a1_posx100; do
  echo "=== RUN $cfg ==="
  timeout 1800 ../.venv/bin/python kinetics_eval.py --projects "$P" \
    --covariance-overlay "$SP/$cfg.yaml" run --label "sl_$cfg" --no-plots --no-open --no-latest \
    || echo "=== TIMEOUT/FAIL $cfg ==="
done
echo "=== SLIP TRY DONE ==="
