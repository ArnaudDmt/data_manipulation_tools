#!/bin/bash
SP="$(dirname "$0")"
until grep -q "KANGLOW DONE" "$SP/kanglow.log" 2>/dev/null; do sleep 20; done
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=KO_TRO2024_RHPS1_1,KO_TRO2024_RHPS1_2,KO_TRO2024_RHPS1_3,KO_TRO2024_RHPS1_4,KO_TRO2024_RHPS1_5,KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3
for tag in adamp5 adamp60 lstiff1e4 lstiff1e5 ldamp40 ldamp600; do
  echo "=== $tag ==="
  timeout 1200 ../.venv/bin/python kinetics_eval.py --projects "$P" \
    --covariance-overlay "$SP/flex_$tag.yaml" run --label "FLEX_$tag" --no-plots --no-open --no-latest 2>&1 | tail -1 || echo "FAIL $tag"
done
echo "=== FLEX DONE ==="
