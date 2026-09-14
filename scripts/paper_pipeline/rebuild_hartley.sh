#!/bin/bash
# Re-run the offline RI-EKF on the input the last tick produced, and refresh the pipeline's cache.
#
# chain.sh does NOT run the RI-EKF: it copies a cached parse from results/paper-rebuild/hartley/,
# because the baseline does not depend on the Kinetics Observer's tuning. That holds for every
# variant except one that changes what the SENSORS deliver -- the synthetic gyrometer noise. Run
# without this step, such an experiment compares a degraded KO against an untouched RI-EKF, which
# is exactly what happened on 2026-09-14 at 20:05.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK="$ROOT/results/paper-rebuild"
HARTLEY=${HARTLEY_DIR:-$HOME/Documents/HartleyIEKF_WithPlots}
project=${1:?usage: rebuild_hartley.sh <project> <robot>}
robot=${2:?usage: rebuild_hartley.sh <project> <robot>}

[ -f /tmp/HartleyInput.txt ] || { echo "[$project] ABANDON: /tmp/HartleyInput.txt absent"; exit 1; }
# The plugin rewrites it at every tick, so a stale file means the tick did not run the plugin.
age=$(( $(date +%s) - $(date -r /tmp/HartleyInput.txt +%s) ))
[ "$age" -lt 3600 ] || { echo "[$project] ABANDON: /tmp/HartleyInput.txt a $age s, le tick ne l'a pas ecrit"; exit 1; }

cp /tmp/HartleyInput.txt "$HARTLEY/data/HartleyInput.txt" || exit 1
rm -f "$HARTLEY/data/HartleyOutput.csv"
( cd "$HARTLEY/bin" && HARTLEY_ROBOT="$robot" ./InEkfLogParser ) > /tmp/inekf_$project.log 2>&1 \
  || { echo "[$project] ABANDON: InEkfLogParser a echoue, voir /tmp/inekf_$project.log"; exit 1; }
[ -s "$HARTLEY/data/HartleyOutput.csv" ] || { echo "[$project] ABANDON: sortie vide"; exit 1; }

mkdir -p "$WORK/hartley"
cp "$HARTLEY/data/HartleyOutput.csv" "$WORK/hartley/$project-HartleyOutput.csv" || exit 1
echo "[$project] parse RI-EKF refait ($(wc -l < "$WORK/hartley/$project-HartleyOutput.csv") lignes)"
