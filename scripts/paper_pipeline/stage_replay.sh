#!/bin/bash
# Relative-error side of the rebuild: the replay pipeline (kinetics_eval.py) scores each variant
# over the thirteen datasets and leaves results/var-<label>-<hash>/ behind, which is what the
# metrics stage pools.
#
# Routine and replay agree to under a micrometre, so either could produce these; they are kept
# separate because the replay is the one that can run a whole variant without re-ticking, and the
# routine is the one that produces the velocities, the RI-EKF baseline and the figures.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK="$ROOT/results/paper-rebuild"
cd "$ROOT" || exit 1
stamp() { date '+%H:%M:%S'; }

ALL=$(env/bin/python -c "
import sys; sys.path.insert(0,'scripts/paper_pipeline'); import manifest as m
print(','.join(m.ALL))")
count=$(env/bin/python -c "
import sys; sys.path.insert(0,'scripts/paper_pipeline'); import manifest as m
print(len(m.ALL))")

# Conversion leftovers from an interrupted run can fill the disk on their own.
for d in /tmp/mc_rtc_convert_*; do
  [ -d "$d" ] && ! pgrep -af "$d" > /dev/null 2>&1 && rm -rf "$d"
done
free=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
echo "[$(stamp)] $free Go libres"
[ "$free" -ge 40 ] || { echo "ABANDON: espace insuffisant"; exit 1; }

echo "[$(stamp)] installation de la configuration retenue"
cp "$WORK/configs/clean/MCKineticsObserver.yaml" ~/.config/mc_rtc/observers/MCKineticsObserver.yaml

# The variant overlays are derived from the retained tuning, never frozen copies of it: change a
# covariance and the variants must follow, or the tables silently compare two different tunings.
echo "[$(stamp)] regeneration des overlays de variantes"
env/bin/python "$HERE/make_overlays.py" | tail -3

echo "[$(stamp)] prepare --force"
.venv/bin/python scripts/kinetics_eval.py --projects "$ALL" prepare --force \
  > "$WORK/replay_prepare.log" 2>&1
n=$(grep -c ": prepared" "$WORK/replay_prepare.log")
echo "[$(stamp)] prepare: $n/$count"
# A partial preparation still lets the prepared datasets be scored; only a total failure is fatal.
if [ "$n" -ne "$count" ]; then
  echo "AVERTISSEMENT: preparation incomplete ($n/$count), on continue sur ce qui est pret"
  tail -5 "$WORK/replay_prepare.log"
  [ "$n" -gt 0 ] || { echo "ABANDON: aucun dataset prepare"; exit 1; }
fi

echo "[$(stamp)] reference : la configuration retenue elle-meme"
.venv/bin/python scripts/kinetics_eval.py --projects "$ALL" \
  run --label var-clean-ref --no-plots --no-open --no-latest 2>&1 | grep -E "^Results:|Traceback|Error"

# noconstraint is an analysis variant (the projector M alone); it rides the same prepared caches.
for v in zpc noconstraint flex-div10 flex-mul10; do
  echo "[$(stamp)] variante $v"
  # One variant failing must not cost the others: they are independent runs over the same caches.
  .venv/bin/python scripts/kinetics_eval.py --projects "$ALL" \
    --covariance-overlay "results/var-$v-overlay.yaml" \
    run --label "var-$v" --no-plots --no-open --no-latest 2>&1 | grep -E "^Results:|Traceback|Error" \
    || echo "[$v] ECHEC, on passe a la suivante"
done

# pinContacts changes the state dimension, so it cannot ride the shared cache.
echo "[$(stamp)] variante pc (pinContacts, cache isole)"
.venv/bin/python scripts/kinetics_eval.py --projects "$ALL" --cache-suffix _pc \
  --observer-config "$WORK/configs/pc/MCKineticsObserver.yaml" prepare --force \
  > "$WORK/replay_prepare_pc.log" 2>&1
n=$(grep -c ": prepared" "$WORK/replay_prepare_pc.log"); echo "[$(stamp)] prepare pc: $n/$count"
if [ "$n" -eq "$count" ]; then
  .venv/bin/python scripts/kinetics_eval.py --projects "$ALL" --cache-suffix _pc \
    --observer-config "$WORK/configs/pc/MCKineticsObserver.yaml" \
    run --label var-pc --no-plots --no-open --no-latest 2>&1 | grep -E "^Results:|Traceback|Error"
else
  echo "  pc: preparation incomplete"; tail -8 "$WORK/replay_prepare_pc.log"
fi
echo "STAGE_REPLAY_DONE"
