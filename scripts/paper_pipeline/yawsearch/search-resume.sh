#!/bin/bash
# Reprend la recherche ciblee lacet la ou elle s'est arretee.
#
# Autonome : ne depend ni du scratchpad (/tmp, efface au redemarrage) ni de la session Claude.
# Les dimensions sont figees dans search-dims.txt, l'etude Optuna dans results/kinetics-retuning.
# Le tuner recalcule `remaining = trials - completed`, donc relancer ne refait rien de deja fait.
#
#   ./search-resume.sh          reprend en tache de fond et rend la main
#   ./search-resume.sh --fg     reprend au premier plan, pour voir les essais defiler
set -u
ROOT=/home/arnaud/devel/src/data_manipulation_tools
HERE="$ROOT/results/paper-rebuild/grid-scripts"
LOG="$ROOT/results/paper-rebuild/grid-yawsearch.log"
DB="$ROOT/results/kinetics-retuning/study.db"
DATASETS=KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3,HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,KO_TRO2024_RHPS1_1

cd "$ROOT" || exit 1
if pgrep -f "yawsearch.py" > /dev/null; then
  echo "la recherche tourne deja, rien a faire"
  exit 0
fi
DIMS=$(cat "$HERE/search-dims.txt") || { echo "ABANDON: search-dims.txt introuvable"; exit 1; }
done_trials=0
[ -f "$DB" ] && done_trials=$(sqlite3 "$DB" "select count(*) from trials where state='COMPLETE'" 2>/dev/null || echo 0)
echo "reprise : $done_trials essais deja termines sur 1000, $(( 1000 - done_trials )) restants"
echo "journal : $LOG"

run() {
  {
    echo
    echo "=== reprise $(date '+%Y-%m-%d %H:%M'), $done_trials essais deja faits"
  } >> "$LOG"
  .venv/bin/python "$HERE/yawsearch.py" --only $DIMS --datasets "$DATASETS" \
    --screen-trials 1000 --full-trials 60 --promote 12 --workers 12 \
    --prefix yawsearch >> "$LOG" 2>&1
  {
    echo; echo "### RECHERCHE CIBLEE LACET, fin $(date '+%H:%M')"; echo
    tail -60 "$LOG"
  } >> "$ROOT/results/paper-rebuild/grid-report.txt"
}

if [ "${1:-}" = "--fg" ]; then
  run
else
  # declare -f n'emporte que le corps de la fonction : sans export, le bash detache
  # verrait ROOT, HERE, LOG, DIMS et DATASETS vides et ecrirait a cote.
  export ROOT HERE LOG DB DATASETS DIMS done_trials
  setsid bash -c "$(declare -f run); run" < /dev/null > /dev/null 2>&1 &
  disown 2>/dev/null
  sleep 5
  pid=$(pgrep -f 'yawsearch.py' | head -1)
  [ -n "$pid" ] && echo "relancee en tache de fond, PID $pid" || echo "ATTENTION: rien ne tourne, voir $LOG"
fi
