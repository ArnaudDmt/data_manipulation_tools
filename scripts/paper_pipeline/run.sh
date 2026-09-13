#!/bin/bash
# Rebuild the paper's measured content end to end.
#
#   ./run.sh                 every stage, in order
#   ./run.sh routine         one stage
#   ./run.sh routine hidehand    one stage, one variant
#   ./run.sh figures poseAndVel  one stage, one figure
#
# Stages, in dependency order:
#   replay   kinetics_eval over the thirteen datasets per variant -> results/var-<label>-*
#            (the relative errors)
#   routine  the mc_rtc chain per variant -> velocity pickles, RI-EKF baseline, figure inputs,
#            disturbance-wrench logs
#   metrics  pool all three families into macros and fold them into metrics_results.tex
#   figures  regenerate every data figure, install it, and export PNG + PDF for all of them
#
# Each stage is independently runnable and leaves its snapshots behind, so a failed stage can be
# rerun without redoing the ones before it.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK="$ROOT/results/paper-rebuild"
cd "$ROOT" || exit 1
mkdir -p "$WORK"

stage=${1:-all}
argument=${2:-all}
started=$(date +%s)
say() { echo; echo "##################################################"; echo "# $*  [$(date +%H:%M:%S)]"; echo "##################################################"; }

run_replay()  { say "replay";  "$HERE/stage_replay.sh"; }
run_routine() { say "routine"; "$HERE/stage_routine.sh" "$argument"; }
run_metrics() { say "metrics"; env/bin/python "$HERE/metrics.py"; }
run_figures() { say "figures"; env/bin/python "$HERE/figures.py" "$argument"; }

case "$stage" in
  replay)  run_replay ;;
  routine) run_routine ;;
  metrics) run_metrics ;;
  figures) run_figures ;;
  all)
    run_replay  || { echo "ABANDON a l'etage replay"; exit 1; }
    run_routine || { echo "ABANDON a l'etage routine"; exit 1; }
    run_metrics || { echo "ABANDON a l'etage metrics"; exit 1; }
    run_figures || { echo "ABANDON a l'etage figures"; exit 1; }
    ;;
  *) echo "etage inconnu: $stage (replay | routine | metrics | figures | all)"; exit 1 ;;
esac

echo
echo "termine en $(( ($(date +%s) - started) / 60 )) min"
