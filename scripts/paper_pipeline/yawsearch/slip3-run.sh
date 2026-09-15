#!/bin/bash
# Recherche de cause sur le pire glissement en lacet (SLIPPAGE_3, rapport KO/RI-EKF 1.116).
# Lance, reprend, et s'arrete proprement -- meme mecanique que search-resume.sh : l'etude est
# SQLite avec load_if_exists, et le tuner recalcule `remaining = trials - completed`, donc
# relancer ne refait jamais un essai deja fait. Changer TRIALS ci-dessous prolonge la campagne.
#
#   ./slip3-run.sh            lance ou reprend en tache de fond
#   ./slip3-run.sh --fg       au premier plan
#   ./slip3-run.sh --stop     met en pause sans rien perdre
set -u
ROOT=/home/arnaud/devel/src/data_manipulation_tools
HERE="$ROOT/results/paper-rebuild/grid-scripts"
TRACK="$ROOT/results/yaw-slip3"
LOG="$ROOT/results/paper-rebuild/slip3-search.log"
DB="$TRACK/study.db"
TRIALS=500
DATASET=KO_TRO_2024_RHPS1_SLIPPAGE_3

cd "$ROOT" || exit 1

if [ "${1:-}" = "--stop" ]; then
  n=0; [ -f "$DB" ] && n=$(sqlite3 "$DB" "select count(*) from trials where state='COMPLETE'" 2>/dev/null || echo '?')
  pkill -f "yawslip3.py" 2>/dev/null; sleep 3
  pkill -9 -f "yawslip3.py" 2>/dev/null; pkill -f "kinetics_eval.py" 2>/dev/null
  echo "en pause, $n essais conserves ; reprendre avec $0"
  exit 0
fi

if pgrep -f "yawslip3.py" > /dev/null; then
  echo "la recherche tourne deja, rien a faire"; exit 0
fi
done_trials=0
[ -f "$DB" ] && done_trials=$(sqlite3 "$DB" "select count(*) from trials where state='COMPLETE'" 2>/dev/null || echo 0)
echo "$done_trials/$TRIALS essais deja termines, $(( TRIALS - done_trials )) restants"
echo "journal : $LOG"

run() {
  echo "=== $(date '+%Y-%m-%d %H:%M'), $done_trials essais deja faits" >> "$LOG"
  .venv/bin/python "$HERE/yawslip3.py" --only $(cd "$ROOT" && .venv/bin/python -c "
import sys,importlib.util; sys.argv=['x']
s=importlib.util.spec_from_file_location('y','$HERE/yawslip3.py')
m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(' '.join(m.DIMS))") \
    --datasets "$DATASET" --tracking "$TRACK" \
    --screen-trials "$TRIALS" --full-trials 0 --workers 12 \
    --prefix slip3 >> "$LOG" 2>&1
}

export ROOT HERE LOG DB TRACK TRIALS DATASET done_trials
if [ "${1:-}" = "--fg" ]; then
  run
else
  setsid bash -c "$(declare -f run); run" < /dev/null > /dev/null 2>&1 &
  disown 2>/dev/null
  sleep 6
  pid=$(pgrep -f 'yawslip3.py' | head -1)
  [ -n "$pid" ] && echo "lancee, PID $pid" || echo "ATTENTION: rien ne tourne, voir $LOG"
fi
