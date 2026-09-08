#!/bin/bash
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,HRP5_MultiContact_4
for a in 0.0 1.0; do
  echo "=== TUNED alpha=$a ==="
  timeout 1800 ../.venv/bin/python kinetics_eval.py --projects "$P" \
    --covariance-overlay "$SP/tuned_a$a.yaml" run --label "tn$a" --no-plots --no-open --no-latest \
    || echo "=== TIMEOUT/FAIL alpha=$a ==="
done
echo "=== TUNED2 DONE ==="
