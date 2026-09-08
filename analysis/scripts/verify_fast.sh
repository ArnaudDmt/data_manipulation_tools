#!/bin/bash
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
RH=KO_TRO2024_RHPS1_1,KO_TRO2024_RHPS1_2,KO_TRO2024_RHPS1_3,KO_TRO2024_RHPS1_4,KO_TRO2024_RHPS1_5,KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3
MC=HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,HRP5_MultiContact_4
# RHPS1 first (slippage + nominal, the criteria under test), then MultiContact. LongWalk is left
# out: one replay of it costs ~50 min, more than everything else combined, and it answers nothing
# that the other eleven datasets do not.
for grp in RH MC; do
  eval "P=\$$grp"
  for cfg in tuned_a0.0 tuned_a1_posx100; do
    echo "=== $grp $cfg ==="
    timeout 2700 ../.venv/bin/python kinetics_eval.py --projects "$P" \
      --covariance-overlay "$SP/$cfg.yaml" run --label "v_${grp}_$cfg" --no-plots --no-open --no-latest \
      || echo "=== FAIL $grp $cfg ==="
  done
done
echo "=== FAST VERIFY DONE ==="
