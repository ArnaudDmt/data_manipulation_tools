#!/bin/bash
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
SP="$(dirname "$0")"
P=HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,HRP5_MultiContact_4
# regenerate against the stock observer config, so the only difference under test is the overlay
../.venv/bin/python kinetics_eval.py --projects "$P" prepare --force || exit 1
../.venv/bin/python kinetics_eval.py --projects "$P" \
  run --label handbase2 --no-plots --no-open --no-latest || exit 1
../.venv/bin/python kinetics_eval.py --covariance-overlay "$SP/hand_overlay.yaml" --projects "$P" \
  run --label handwrench --no-plots --no-open --no-latest
echo "HAND EXPERIMENT DONE rc=$?"
