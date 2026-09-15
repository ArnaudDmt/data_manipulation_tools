#!/bin/bash
# Met en pause la recherche ciblee lacet, sans rien perdre.
#
# Les essais termines sont dans results/kinetics-retuning/study.db (SQLite). Le tuner rouvre cette
# etude avec load_if_exists=True et recalcule `remaining = trials - completed`, donc reprendre
# revient a relancer la meme commande : les essais deja faits ne sont pas refaits.
# Seul l'essai en cours au moment de la pause est perdu, soit ~2 minutes.
set -u
ROOT=/home/arnaud/devel/src/data_manipulation_tools
DB="$ROOT/results/kinetics-retuning/study.db"

before=0
[ -f "$DB" ] && before=$(sqlite3 "$DB" \
  "select count(*) from trials where state='COMPLETE'" 2>/dev/null || echo '?')

pkill -f "grid-scripts/chain4.sh" 2>/dev/null
pkill -f "scratchpad/grid/chain4.sh" 2>/dev/null
pkill -f "yawsearch.py" 2>/dev/null
sleep 3
pkill -9 -f "yawsearch.py" 2>/dev/null
# Les replays lances par les ouvriers meurent avec eux ; on nettoie les rescapes.
pkill -f "kinetics_eval.py" 2>/dev/null
sleep 1

alive=$(pgrep -cf "yawsearch.py" 2>/dev/null || echo 0)
echo "recherche en pause"
echo "  essais termines et conserves : $before"
echo "  processus restants           : $alive"
echo "  reprendre avec               : $ROOT/results/paper-rebuild/grid-scripts/search-resume.sh"
