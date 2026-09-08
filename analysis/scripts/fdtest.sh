#!/bin/bash
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=KO_TRO2024_RHPS1_1,KO_TRO_2024_RHPS1_SLIPPAGE_1
for cfg in mufd; do
  echo "=== $cfg prepare ==="
  ../.venv/bin/python kinetics_eval.py --projects "$P" --observer-config "$SP/$cfg/MCKineticsObserver.yaml" \
    prepare --force 2>&1 | grep -c prepared
  echo "=== $cfg run (finite differences: slow) ==="
  timeout 3000 ../.venv/bin/python kinetics_eval.py --projects "$P" \
    --covariance-overlay "$SP/tuned_a1_posx30.yaml" --observer-config "$SP/$cfg/MCKineticsObserver.yaml" \
    run --label "FD_mu0.8" --no-plots --no-open --no-latest 2>&1 | tail -1 || echo "FAIL $cfg"
done
echo "=== FD DONE ==="
