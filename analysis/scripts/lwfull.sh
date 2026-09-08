#!/bin/bash
set -e
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,HRP5_MultiContact_4,HRP5P_LongWalk,KO_TRO2024_RHPS1_1,KO_TRO2024_RHPS1_2,KO_TRO2024_RHPS1_3,KO_TRO2024_RHPS1_4,KO_TRO2024_RHPS1_5,KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3
../.venv/bin/python kinetics_eval.py --projects "$P" prepare
echo "=== PREPARE DONE ==="
for a in 0.0 0.5 1.0 2.0; do
  echo "=== RUN alpha=$a ==="
  ../.venv/bin/python kinetics_eval.py --projects "$P" --covariance-overlay "$SP/lwa$a.yaml" \
    run --label "lwa$a" --no-plots --no-open --no-latest || echo "=== FAILED alpha=$a ==="
done
echo "=== ALL DONE ==="
