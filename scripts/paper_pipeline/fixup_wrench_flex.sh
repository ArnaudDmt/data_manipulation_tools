#!/bin/bash
# Table 7 of the paper: disturbance-wrench error for the three flexibility tunings.
#
# Its numbers were computed on 2026-09-13, before the disturbance-wrench process moved to 0.09, so
# its "Init flex" row contradicts tab:ExtWrenchErrors -- the same quantity, same configuration,
# two different values eleven pages apart. The init row is the hidehand run already in
# results/paper-rebuild/runs/hidehand; only the two flexibility variants need rerunning.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK="$ROOT/results/paper-rebuild"
cd "$ROOT" || exit 1
PROJS="HRP5_MultiContact_1_WO_LeftHand HRP5_MultiContact_2_WO_LeftHand \
HRP5_MultiContact_3_WO_LeftHand HRP5_MultiContact_4_WO_LeftHand"

for flex in flexdiv10 flexmul10; do
  echo "############ hidehand + $flex"
  env/bin/python - "$flex" <<'PY' || exit 1
import sys
sys.path.insert(0, "scripts/paper_pipeline")
import variant_install as vi
vi.main("hidehand")                 # restores the retained tuning, then hides the left hand
vi.scale_flexibilities(0.1 if sys.argv[1] == "flexdiv10" else 10.0)
print(f"installed hidehand + {sys.argv[1]}")
PY
  store="$WORK/runs/hidehand-$flex"
  for p in $PROJS; do
    echo "================ $flex / $p"
    mkdir -p "$store/$p"
    "$HERE/chain.sh" "$p" || { echo "[$flex/$p] ABANDONNE"; continue; }
    env/bin/python "$HERE/extract_wrench.py" \
      "Projects/$p/output_data/logReplay.csv" "$store/$p/wrench.csv" || echo "[$flex/$p] EXTRACTION KO"
  done
done
env/bin/python "$HERE/variant_install.py" clean
echo WRENCH_FLEX_DONE
