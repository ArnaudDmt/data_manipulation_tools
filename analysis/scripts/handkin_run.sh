#!/bin/bash
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
SP="$(dirname "$0")"
P=HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,HRP5_MultiContact_4
CFG="$SP/handkin_cfg/MCKineticsObserver.yaml"
../.venv/bin/python kinetics_eval.py --observer-config "$CFG" --projects "$P" prepare --force \
  && ../.venv/bin/python kinetics_eval.py --observer-config "$CFG" --projects "$P" \
       run --label handkin --no-plots --no-open --no-latest
echo "EXPERIMENT DONE rc=$?"
