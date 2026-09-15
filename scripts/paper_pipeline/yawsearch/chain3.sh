#!/bin/bash
# Attend la fin de l'enchainement etapes 1-2, puis valide les candidates :
# crible sur les quatre multicontacts, puis les neuf datasets restants pour les survivantes.
# chain2.sh n'est pas modifie -- bash relit un script en cours d'execution, l'editer le corromprait.
SP=/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/287daf9e-ad1c-4931-a07e-923142412c80/scratchpad
ROOT=/home/arnaud/devel/src/data_manipulation_tools
while kill -0 4002560 2>/dev/null; do sleep 30; done
cd "$ROOT" || exit 1
.venv/bin/python "$SP/grid/stage3.py" > "$SP/grid/stage3.log" 2>&1
{ echo "=== balayage et validation, $(date '+%Y-%m-%d %H:%M') ==="; echo
  echo "### GLISSEMENTS : toutes les configurations"; echo
  .venv/bin/python "$SP/grid/report2.py"
  echo; echo "### VALIDATION : crible multicontact puis les neuf restants"; echo
  cat "$SP/grid/stage3.log"
  echo; echo "### criblage de l etape 2"; cat "$SP/grid/stage2.log"
  echo; echo "### incidents"
  grep -hE "ABANDON|MANQUE" "$SP/grid/sweep.log" "$SP/grid/stage2.log" "$SP/grid/stage3.log" 2>/dev/null | tail -30
} > "$ROOT/results/paper-rebuild/grid-report.txt" 2>&1
for f in sweep stage2 stage3; do cp "$SP/grid/$f.log" "$ROOT/results/paper-rebuild/grid-$f.log" 2>/dev/null; done
