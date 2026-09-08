#!/bin/bash
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
SP="$(dirname "$0")"
# Three slippage runs plus two nominal: the smallest set that shows both the intended gain and its
# cost. Analytical Jacobians (the default) -- finite differences perturb all 51 states each step.
P=KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3,KO_TRO2024_RHPS1_1,KO_TRO2024_RHPS1_3
../.venv/bin/python kinetics_eval.py --projects "$P" prepare --force || exit 1
echo "PREPARE DONE"
../.venv/bin/python kinetics_eval.py --projects "$P" run --label slipvel --no-plots --no-open --no-latest
echo "SLIPVEL DONE rc=$?"
