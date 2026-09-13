#!/bin/bash
# Unattended rebuild: settle the disturbance-wrench process, recompute everything, run the
# constraint experiment, and leave a report behind.
#
# Deliberately not `set -e`. A stage that fails must not take the rest of the night with it: each
# step records its outcome and the next one runs if its own inputs exist. The report says plainly
# what did not happen.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK="$ROOT/results/paper-rebuild"
REPORT="$WORK/overnight-report.md"
SWEEP=/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/287daf9e-ad1c-4931-a07e-923142412c80/scratchpad
cd "$ROOT" || exit 1
mkdir -p "$WORK"
: > "$REPORT"

say()  { echo; echo "===== $* [$(date +%H:%M:%S)]"; }
note() { printf '%s\n' "$*" >> "$REPORT"; }
step() {                       # step <label> <command...>
  local label=$1; shift
  say "$label"
  if "$@"; then note "- OK **$label**"; return 0
  else local code=$?; note "- ECHEC **$label** (code $code)"; return $code; fi
}

note "# Rebuild de nuit — $(date '+%Y-%m-%d %H:%M')"
note ""

# ---------------------------------------------------------------- 1. attendre le sweep
say "attente de la fin du sweep de vitesses"
while pgrep -f "uw_cost.sh|uw_vel.sh" > /dev/null; do sleep 20; done
note "- sweep de vitesses termine ($(grep -c 'OK$' "$SWEEP/uw-vel.log" 2>/dev/null || echo 0)/26 chaines)"
rm -f "$SWEEP/retick-staging"/*.bin

# ---------------------------------------------------------------- 2. choisir uw
if step "choix du process de wrench" env/bin/python "$HERE/decide_uw.py"; then
  CHOICE=$(cat "$WORK/uw_choice/value")
  cat "$WORK/uw_choice/report.md" >> "$REPORT"
else
  CHOICE=4
  note "- repli sur uw=4, la configuration actuelle"
fi
note ""
say "valeur retenue: $CHOICE"

# ---------------------------------------------------------------- 3. poser la configuration
if [ "$CHOICE" != "4" ]; then
  step "installation de uw=$CHOICE dans la config de reference" \
    env/bin/python "$HERE/set_unmodeled.py" "$CHOICE" || exit 1
  step "regeneration des overlays" env/bin/python "$HERE/make_overlays.py"
  grep -A1 "unmodeled_wrench_process" results/var-zpc-overlay.yaml | tail -1 \
    | grep -q -- "- $CHOICE" \
    && note "- overlays synchronises sur uw=$CHOICE" \
    || note "- **ATTENTION** les overlays ne portent pas uw=$CHOICE"
else
  note "- configuration inchangee, pas de reinstallation"
fi
note ""

# ---------------------------------------------------------------- 4. reconstruction
#
# What genuinely depends on what:
#   replay   -> nothing. A separate pipeline; runs even if the routine chain is broken.
#   routine  -> the smoke test. If one short dataset cannot go through the chain, none will.
#   metrics  -> replay for the relative errors, routine for the velocities and the wrench.
#               The three families are independent inside metrics.py, so it runs if EITHER ran.
#   figures  -> routine, and only routine: they read Projects/*/output_data. Skipped when the
#               routine stage did not run, because they would otherwise be regenerated from the
#               previous tuning's data and look current.
#   distill / tables -> whatever exists.
replay_ok=0; routine_ok=0

step "etage replay" "$HERE/run.sh" replay && replay_ok=1

if step "rodage: etage routine sur la variante hidehand" "$HERE/run.sh" routine hidehand; then
  step "etage routine (7 variantes)" "$HERE/run.sh" routine && routine_ok=1
else
  note "- routine complet **saute**: le rodage sur un dataset court a echoue"
fi

if [ "$replay_ok" = 1 ] || [ "$routine_ok" = 1 ]; then
  step "etage metrics" "$HERE/run.sh" metrics
else
  note "- metrics **saute**: ni replay ni routine n'ont produit de donnees"
fi

if [ "$routine_ok" = 1 ]; then
  step "etage figures" "$HERE/run.sh" figures
else
  note "- figures **sautees**: sans etage routine elles viendraient de l'ancien reglage"
  step "export des figures existantes" env/bin/python "$HERE/figures.py" --export-only
fi

step "distillation du resume" env/bin/python "$HERE/distill.py"
note ""

# ---------------------------------------------------------------- 6. resultats
say "tableaux"
{ note "## Erreurs relatives par categorie"; note ""; note '```'
  env/bin/python "$HERE/report_tables.py" 2>&1; note '```'; } >> "$REPORT" 2>&1
note ""

echo "OVERNIGHT_DONE"
note "_termine a $(date '+%H:%M')_"
