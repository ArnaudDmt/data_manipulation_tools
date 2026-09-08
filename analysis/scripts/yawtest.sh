#!/bin/bash
SP="$(dirname "$0")"
while pgrep -f 'verify1[3]' >/dev/null; do sleep 30; done
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3,KO_TRO2024_RHPS1_1
for cfg in tuned_yaw100 tuned_yaw100_rp; do
  echo "=== RUN $cfg ==="
  timeout 1800 ../.venv/bin/python kinetics_eval.py --projects "$P" \
    --covariance-overlay "$SP/$cfg.yaml" run --label "yw_$cfg" --no-plots --no-open --no-latest \
    || echo "=== FAIL $cfg ==="
done
echo "=== YAWTEST DONE ==="
