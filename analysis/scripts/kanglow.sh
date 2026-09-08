#!/bin/bash
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=KO_TRO2024_RHPS1_1,KO_TRO2024_RHPS1_2,KO_TRO2024_RHPS1_3,KO_TRO2024_RHPS1_4,KO_TRO2024_RHPS1_5,KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3
for k in 300 100; do
  echo "=== kang $k ==="
  timeout 1200 ../.venv/bin/python kinetics_eval.py --projects "$P" \
    --covariance-overlay "$SP/kang$k.yaml" run --label "KANG_$k" --no-plots --no-open --no-latest 2>&1 | tail -1 || echo "FAIL $k"
done
echo "=== KANGLOW DONE ==="
