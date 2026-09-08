#!/bin/bash
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,HRP5_MultiContact_4,KO_TRO2024_RHPS1_1,KO_TRO2024_RHPS1_2,KO_TRO2024_RHPS1_3,KO_TRO2024_RHPS1_4,KO_TRO2024_RHPS1_5,KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3
for v in 1.0 3.0; do
  echo "=== RUN cw$v ==="
  timeout 1800 ../.venv/bin/python kinetics_eval.py --projects "$P" \
    --covariance-overlay "$SP/cal_cw$v.yaml" --observer-config "$SP/cal/MCKineticsObserver.yaml" \
    run --label "CW_$v" --no-plots --no-open --no-latest 2>&1 | tail -1 || echo "FAIL $v"
done
echo "=== CW DONE ==="
