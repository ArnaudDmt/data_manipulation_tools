#!/bin/bash
# Rescore VALINOR and capture its velocity, then refresh everything downstream.
#
# VALINOR does not read the Kinetics Observer's tuning, so its estimator was never affected by the
# retuning -- but its scored numbers were: the chain regenerates its trajectory on every run while
# never re-running its RPG evaluation, so the paper carried August values against a mocap, a wrench
# calibration and (on LongWalk) an evaluation rate that had all moved since. Measured: 11 of its
# 16 relative-error macros changed, up to 75%.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
REPORT="$ROOT/results/paper-rebuild/overnight-report.md"
cd "$ROOT" || exit 1
note() { printf '%s\n' "$*" >> "$REPORT"; }

note ""; note "## Correctif: VALINOR rejoue et reintegre"
echo "=== etage routine, variante clean [$(date +%H:%M:%S)]"
if "$HERE/run.sh" routine clean; then note "- OK variante clean rejouee (VALINOR evalue et capture)"
else note "- ECHEC de la variante clean"; exit 1; fi

echo "=== metrics [$(date +%H:%M:%S)]"
env/bin/python "$HERE/metrics.py" && note "- OK metriques regenerees" || note "- ECHEC metrics"
env/bin/python "$HERE/rebold.py" > /tmp/rebold_valinor.log 2>&1 \
  && note "- OK gras reattribue ($(grep -o '[0-9]* bold markers moved' /tmp/rebold_valinor.log))" \
  || note "- ECHEC rebold"
env/bin/python "$HERE/distill.py" > /dev/null && note "- OK resume redistille"
echo "FIXUP_VALINOR_DONE"
note "_correctif VALINOR termine a $(date '+%H:%M')_"
