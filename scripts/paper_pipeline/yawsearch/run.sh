#!/bin/bash
# Balayage croise de quatre reglages du KO sur les trois experiences de glissement, via le REPLAY
# (kinetics_eval.py) : le bag est deja converti, donc une combinaison coute ~30 s au lieu des
# ~110 s d'un re-tick routine. state-observation et mc_state_observation sont en RelWithDebInfo.
#
#   stateAngVelProcessVariance          1e-10 (ref) / 1e-8 / 1e-12
#   contactOrientationProcessVariance z 1e-4  (ref) / 1e-2 / 1e-6
#   contactPositionProcessVariance  x,y 1e-5  (ref) / 1e-4 / 1e-6
#   contactOriInitVarianceNewContacts   2e-4  (ref) / 1e-2 / 1e-3 / 1e-5 / 1e-6
#
# 135 combinaisons. Reprenable : une cellule dont les trois pickles sont deja copies est sautee.
# La cellule de reference sert de controle, elle doit reproduire runs/clean.
set -u
SP=/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/287daf9e-ad1c-4931-a07e-923142412c80/scratchpad
ROOT=/home/arnaud/devel/src/data_manipulation_tools
OUT="$ROOT/results/paper-rebuild/runs/grid"
PROJECTS="KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3"
cd "$ROOT" || exit 1

ANGVEL="1e-10 1e-8 1e-12"; ORIYAW="0.0001 1e-2 1e-6"; POSXY="1e-05 1e-4 1e-6"; ORIINIT="0.0002 1e-2 1e-3 1e-5 1e-6"
total=$(( 3 * 3 * 3 * 5 )); i=0; fail=0; started=$(date +%s)
echo "[$(date +%H:%M:%S)] debut, $total combinaisons via le replay"
for av in $ANGVEL; do for oy in $ORIYAW; do for px in $POSXY; do for oi in $ORIINIT; do
  i=$((i+1))
  tag="av${av}_oy${oy}_px${px}_oi${oi}"
  store="$OUT/$tag"
  done_all=1
  for p in ${PROJECTS//,/ }; do [ -s "$store/$p/cached_rel_err.pickle" ] || done_all=0; done
  [ "$done_all" = 1 ] && { echo "[$i/$total] $tag deja fait"; continue; }
  el=$(( $(date +%s) - started ))
  eta=$([ $i -gt 1 ] && echo $(( el * (total - i + 1) / (i - 1) / 60 )) || echo '?')
  echo "[$i/$total] $tag  ($(date +%H:%M:%S), reste ~${eta} min)"
  ov="$SP/grid/overlays/$tag.yaml"; mkdir -p "$(dirname "$ov")"
  if ! .venv/bin/python "$SP/grid/overlay.py" "$av" "$oy" "$px" "$oi" "$ov" >/dev/null; then
    echo "  ABANDON overlay"; fail=$((fail+1)); continue
  fi
  label="grid$i"
  if ! .venv/bin/python scripts/kinetics_eval.py --projects "$PROJECTS" --covariance-overlay "$ov" \
        run --label "$label" --no-plots --no-latest --no-open > "$SP/grid/logs/$tag.log" 2>&1; then
    echo "  ABANDON run, voir $SP/grid/logs/$tag.log"; fail=$((fail+1)); continue
  fi
  d=$(ls -d results/${label}-* 2>/dev/null | head -1)
  if [ -z "$d" ]; then echo "  ABANDON: pas de repertoire de resultats"; fail=$((fail+1)); continue; fi
  for p in ${PROJECTS//,/ }; do
    c="$d/$p/eval/saved_results/traj_est/cached/cached_rel_err.pickle"
    mkdir -p "$store/$p"
    if [ -s "$c" ]; then cp "$c" "$store/$p/cached_rel_err.pickle"; else echo "  MANQUE $p"; fail=$((fail+1)); fi
  done
  # Le repertoire de sortie du replay ne sert plus une fois le pickle recopie, et 135 d'entre eux
  # remplissent le disque.
  rm -rf "$d"
done; done; done; done
echo "[$(date +%H:%M:%S)] BALAYAGE TERMINE en $(( ($(date +%s)-started)/60 )) min, $fail echecs"
