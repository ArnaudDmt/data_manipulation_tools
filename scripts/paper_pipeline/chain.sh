#!/bin/bash
# Routine chain for one project, from a fresh re-tick with whatever observer config is installed.
#
# Three things here were learned the hard way and must not be simplified away:
#  - the timestep comes from the project, never pinned (LongWalk ticks at 0.002, the rest 0.005);
#  - plotAndFormatResults is asked to write its outputs, without which the evaluation copies stale
#    trajectories and the *_loc_vel.pickle velocity files are never refreshed;
#  - the RI-EKF baseline is the OFFLINE parse, not the in-tick plugin, which emits one row every
#    two iterations on the 500 Hz LongWalk log and leaves zero-norm quaternions in the merge.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK="$ROOT/results/paper-rebuild"
cd "$ROOT" || exit 1
p=$1
proj="$ROOT/Projects/$p"
out="$proj/output_data"
stage="$WORK/retick/$p.bin"
mkdir -p "$WORK/retick"
step() { echo "   [$(date +%H:%M:%S)] $*"; }

ts=$(.venv/bin/python -c "import sys;sys.path.insert(0,'scripts');import kinetics_eval as ke;print(ke.project_timestep(ke.ROOT/'Projects'/'$p'))") || exit 1
echo "[$(date +%H:%M:%S)] $p (timestep $ts)"

# The KO-ZPC curve needs a second observer instance in the controller pipeline; see
# manifest.NEEDS_KOZPC for why it is limited to one dataset. Always restored, including on
# failure: a leftover second instance would silently double every later tick's logs.
controller="$WORK/configs/controllers/plain.yaml"
if env/bin/python -c "
import sys; sys.path.insert(0,'scripts/paper_pipeline'); import manifest as m
sys.exit(0 if '$p' in m.NEEDS_KOZPC else 1)"; then
  controller="$WORK/configs/controllers/kozpc.yaml"
  step "instance KOZPC activee dans le controleur"
fi
target=$(env/bin/python -c "
import sys; sys.path.insert(0,'scripts/paper_pipeline'); import manifest as m; print(m.CONTROLLER)")
cp "$controller" "$target" || exit 1
restore_controller() { cp "$WORK/configs/controllers/plain.yaml" "$target" 2>/dev/null; }
trap restore_controller EXIT

step "re-tick"
.venv/bin/python "$HERE/retick_routine.py" "$p" "$stage" || { echo "[$p] RE-TICK FAILED"; exit 1; }

n=$(mc_bin_utils show "$stage" 2>/dev/null | grep -c "MocapAligner")
[ "$n" -gt 0 ] || { echo "[$p] ABANDON: le log re-tique n'a pas la mocap"; exit 1; }

step "allegement -> logReplay.bin"
keys=$(cd scripts && ../env/bin/python lightenOutputBin.py "$proj" "$stage") || exit 1
eval scripts/routine_scripts/lightenBin.sh "$stage" "$out/logReplay.bin" $keys > /dev/null \
  || { echo "[$p] LIGHTEN FAILED"; exit 1; }

step "mc_bin_to_log"
( cd "$out" && rm -f logReplay.csv && mc_bin_to_log logReplay.bin ) > /dev/null 2>&1 \
  || { echo "[$p] BIN_TO_LOG FAILED"; exit 1; }

# The orientation-error project re-ticks the very controllerLog.bin of HRP5_MultiContact_1 (same
# md5, same row count) and the RI-EKF does not depend on the Kinetics Observer's tuning, so that
# project's parse is exactly this one's. It cannot simply be dropped: plotAndFormatResults needs
# posFbImu, which it only defines inside its RI-EKF block.
# Verified by md5: each *_WO_LeftHand project re-ticks byte-identical raw data to its numbered
# twin, and the orientation-error project re-ticks MultiContact_1's.
source=${p%_WO_LeftHand}
[ "$p" = "HRP5_MultiContact_ContactInitOriError" ] && source=HRP5_MultiContact_1
offline="$WORK/hartley/$source-HartleyOutput.csv"
if [ -f "$offline" ]; then
  step "RI-EKF hors ligne ($source)"
  cp "$offline" "$out/HartleyOutputCSV.csv" || exit 1
else
  echo "[$p] ABANDON: pas de parse RI-EKF hors ligne pour $source"; exit 1
fi

cd scripts || exit 1
for cmd in \
  "extractLightReplayVersion.py $proj" \
  "repair_mc_rtc_skipped_iters.py $ts $proj" \
  "initialize_datas.py $ts $proj true" \
  "resampleMocapAndExtractPose.py false y $proj" \
  "crossCorrelation.py false y $proj" \
  "matchInitPose.py 0 false y $proj" \
  "plotAndFormatResults.py false $proj True" ; do
  step "${cmd%% *}"
  ../env/bin/python $cmd > "$WORK/retick/$p.${cmd%%.py*}.log" 2>&1 \
    || { echo "[$p] ${cmd%% *} FAILED - voir $WORK/retick/$p.${cmd%%.py*}.log"
         tail -6 "$WORK/retick/$p.${cmd%%.py*}.log"; exit 1; }
done
cd "$ROOT" || exit 1

# Only the thirteen scored datasets carry predefined sublengths; the hand-removal and
# orientation-error projects feed a figure or a table instead.
if grep -q "predefined_sublengths" "$proj/projectConfig.yaml" 2>/dev/null; then
  step "RPG (KO + RI-EKF)"
  "$HERE/reeval.sh" "$p" Hartley KO > "$WORK/retick/$p.rpg.log" 2>&1 \
    || { echo "[$p] RPG FAILED"; exit 1; }
fi
echo "[$(date +%H:%M:%S)] $p OK"
