#!/bin/bash
# Attend la fin de l'etape 1, enchaine l'etape 2, puis ecrit le rapport final.
SP=/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/287daf9e-ad1c-4931-a07e-923142412c80/scratchpad
ROOT=/home/arnaud/devel/src/data_manipulation_tools
# Attente du PID exact de l etape 1. Un pgrep sur le nom du script attrapait aussi les
# shells interactifs qui le mentionnent, ce qui aurait pu bloquer l enchainement.
while kill -0 3975265 2>/dev/null; do sleep 30; done
cd "$ROOT" || exit 1
.venv/bin/python "$SP/grid/stage2.py" > "$SP/grid/stage2.log" 2>&1
{ echo "=== balayage, glissements, $(date '+%Y-%m-%d %H:%M') ==="; echo
  .venv/bin/python "$SP/grid/report2.py"
  echo; echo "=== criblage de l etape 2 ==="; cat "$SP/grid/stage2.log"
  echo; echo "=== incidents ==="; grep -hE "ABANDON|MANQUE" "$SP/grid/sweep.log" "$SP/grid/stage2.log" | tail -30
} > "$ROOT/results/paper-rebuild/grid-report.txt" 2>&1
cp "$SP/grid/sweep.log" "$ROOT/results/paper-rebuild/grid-sweep.log" 2>/dev/null
cp "$SP/grid/stage2.log" "$ROOT/results/paper-rebuild/grid-stage2.log" 2>/dev/null
