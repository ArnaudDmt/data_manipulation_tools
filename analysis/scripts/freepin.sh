#!/bin/bash
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
SP="$(dirname "$0")"
# control: stock config (flag defaults to false), all 13 datasets
../.venv/bin/python kinetics_eval.py prepare --force || exit 1
../.venv/bin/python kinetics_eval.py run --label pinbase --no-plots --no-open --no-latest || exit 1
echo "CONTROL DONE"
# experiment: same everything, pin released
../.venv/bin/python kinetics_eval.py --observer-config "$SP/freepin_cfg/MCKineticsObserver.yaml" \
  prepare --force || exit 1
../.venv/bin/python kinetics_eval.py --observer-config "$SP/freepin_cfg/MCKineticsObserver.yaml" \
  run --label freepin --no-plots --no-open --no-latest
echo "FREEPIN EXPERIMENT DONE rc=$?"
