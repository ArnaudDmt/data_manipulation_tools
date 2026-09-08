#!/bin/bash
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
SP="$(dirname "$0")"
# prepare WITHOUT --force: reuses each cached full log, only regenerates the replay configs so
# they carry the new flag. Forcing regeneration is what broke LongWalk's compact/merge step.
../.venv/bin/python kinetics_eval.py prepare || exit 1
echo "PREPARE DONE"
../.venv/bin/python kinetics_eval.py run --label pinbase --no-plots --no-open --no-latest || exit 1
echo "CONTROL DONE"
../.venv/bin/python kinetics_eval.py --covariance-overlay "$SP/pin_overlay.yaml" \
  run --label freepin --no-plots --no-open --no-latest
echo "PIN EXPERIMENT DONE rc=$?"
