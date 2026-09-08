#!/bin/bash
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
SP="/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/287daf9e-ad1c-4931-a07e-923142412c80/scratchpad"
P=HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,HRP5_MultiContact_4,KO_TRO2024_RHPS1_1,KO_TRO2024_RHPS1_2,KO_TRO2024_RHPS1_3,KO_TRO2024_RHPS1_4,KO_TRO2024_RHPS1_5,KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3
../.venv/bin/python kinetics_eval.py --projects "$P" prepare || exit 1
echo "PREPARE DONE"
../.venv/bin/python kinetics_eval.py --projects "$P" run --label pinbase --no-plots --no-open --no-latest || exit 1
echo "CONTROL DONE"
../.venv/bin/python kinetics_eval.py --projects "$P" --covariance-overlay "$SP/pin_overlay.yaml"   run --label freepin --no-plots --no-open --no-latest
echo "PIN EXPERIMENT DONE rc=$?"
