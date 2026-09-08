#!/bin/bash
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,HRP5_MultiContact_4,KO_TRO2024_RHPS1_1,KO_TRO2024_RHPS1_2,KO_TRO2024_RHPS1_3,KO_TRO2024_RHPS1_4,KO_TRO2024_RHPS1_5,KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3
for mu in 0.8 0.5; do
  echo "=== mu=$mu prepare ==="
  ../.venv/bin/python kinetics_eval.py --projects "$P" --observer-config "$SP/mu$mu/MCKineticsObserver.yaml" \
    prepare --force 2>&1 | grep -c prepared
  echo "=== mu=$mu run ==="
  timeout 1800 ../.venv/bin/python kinetics_eval.py --projects "$P" \
    --covariance-overlay "$SP/tuned_a1_posx30.yaml" --observer-config "$SP/mu$mu/MCKineticsObserver.yaml" \
    run --label "MU_$mu" --no-plots --no-open --no-latest 2>&1 | tail -1 || echo "FAIL $mu"
done
echo "=== MU DONE ==="
