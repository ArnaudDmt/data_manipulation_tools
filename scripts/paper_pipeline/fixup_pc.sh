#!/bin/bash
# Re-run the pinContacts variant on the corrected configuration, then refresh what depends on it.
#
# pc is the one variant that cannot ride a covariance overlay -- it changes the state dimension --
# so it runs from a standalone observer configuration. That file was a frozen copy and kept the
# previous disturbance-wrench process while every overlay followed the new one, which would have
# put the "KO without wrench sensors" column on a different tuning from every other column.
# make_overlays.py now derives it; this rebuilds the run that was already in flight when it did.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK="$ROOT/results/paper-rebuild"
REPORT="$WORK/overnight-report.md"
cd "$ROOT" || exit 1
note() { printf '%s\n' "$*" >> "$REPORT"; }

echo "attente de la fin du rebuild de nuit"
while pgrep -f "paper_pipeline/overnight.sh" > /dev/null; do sleep 30; done

ALL=$(env/bin/python -c "
import sys; sys.path.insert(0,'scripts/paper_pipeline'); import manifest as m; print(','.join(m.ALL))")
count=$(env/bin/python -c "
import sys; sys.path.insert(0,'scripts/paper_pipeline'); import manifest as m; print(len(m.ALL))")
config="$WORK/configs/pc/MCKineticsObserver.yaml"

note ""
note "## Correctif: variante pc rejouee sur la configuration corrigee"
echo "=== prepare pc [$(date +%H:%M:%S)]"
.venv/bin/python scripts/kinetics_eval.py --projects "$ALL" --cache-suffix _pc \
  --observer-config "$config" prepare --force > "$WORK/fixup_pc_prepare.log" 2>&1
n=$(grep -c ": prepared" "$WORK/fixup_pc_prepare.log")
echo "prepare pc: $n/$count"
if [ "$n" -ne "$count" ]; then
  note "- ECHEC preparation pc ($n/$count); la colonne KO sans capteurs reste sur l'ancien reglage"
  tail -5 "$WORK/fixup_pc_prepare.log"; exit 1
fi

echo "=== run pc [$(date +%H:%M:%S)]"
if .venv/bin/python scripts/kinetics_eval.py --projects "$ALL" --cache-suffix _pc \
     --observer-config "$config" run --label var-pc --no-plots --no-open --no-latest 2>&1 \
     | grep -E "^Results:|Traceback|Error"; then
  note "- OK pc rejouee sur uw corrige"
else
  note "- ECHEC du run pc"; exit 1
fi

# Everything downstream of the tables has to be recomputed now that one column moved.
echo "=== metrics [$(date +%H:%M:%S)]"
env/bin/python "$HERE/metrics.py" && note "- OK metrics regenerees" || note "- ECHEC metrics apres correctif"
env/bin/python "$HERE/distill.py" && note "- OK resume redistille" || note "- ECHEC distillation"

{ note ""; note "## Tableaux definitifs (apres correctif pc)"; note ""; note '```'
  env/bin/python "$HERE/report_tables.py" 2>&1; note '```'; } >> "$REPORT" 2>&1
echo "FIXUP_PC_DONE"
note ""
note "_correctif pc termine a $(date '+%H:%M')_"
