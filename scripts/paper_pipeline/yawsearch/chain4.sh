#!/bin/bash
# Recherche ciblee sur le LACET, lancee seulement quand tout le reste est fini.
#
# Nouveauté par rapport aux campagnes precedentes : l'objectif du tuner lit maintenant des
# MOYENNES et non des medianes. Sur les medianes le KO battait deja le RI-EKF en lacet partout,
# ce qui est la raison pour laquelle "~1400 evaluations n'ont jamais ameliore le lacet" -- l'objectif
# ne le voyait pas. Avec les moyennes, il le voit.
#
# Les dimensions sont choisies a partir du criblage de l'etape 2 : un axe mesure inerte est retire.
SP=/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/287daf9e-ad1c-4931-a07e-923142412c80/scratchpad
ROOT=/home/arnaud/devel/src/data_manipulation_tools
# TOUS les glissements et trois multicontacts sur quatre : un seul essai par categorie
# surajuste a cet essai-la, et les trois glissements different deja de 0.33 a 0.42 deg en
# lacet entre eux. Les deux ajouts sont les plus petits datasets du lot, donc ils coutent peu.
DATASETS=KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3,HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,KO_TRO2024_RHPS1_1
while kill -0 4005811 2>/dev/null; do sleep 60; done
cd "$ROOT" || exit 1
DIMS=$(.venv/bin/python "$SP/grid/plan_search.py" 2>"$SP/grid/plan.log")
[ -n "$DIMS" ] || { echo "ABANDON: aucune dimension" > "$SP/grid/search.log"; exit 1; }
{
  echo "=== recherche ciblee lacet, $(date '+%Y-%m-%d %H:%M')"
  cat "$SP/grid/plan.log"
  echo "dimensions : $DIMS"
  echo "datasets   : $DATASETS"
  echo "             (3 glissements, 3 multicontacts, 1 marche ; LongWalk exclu)"
  echo
} > "$SP/grid/search.log"
# yawsearch.py impose les bornes d'Arnaud et des barreaux a mantisses 1 et 2 (sans le 5),
# puis appelle kinetics_tune.main(). Sans lui, le tuner proposerait dans SES decades avec
# ses mantisses 1/2/5 -- et son echelle du lacet de contact s'arrete a 1e-4, donc le 1e-2
# demande serait hors plage.
.venv/bin/python "$SP/grid/yawsearch.py" --only $DIMS --datasets "$DATASETS" \
  --screen-trials 1400 --full-trials 60 --promote 12 --workers 12 \
  --prefix yawsearch >> "$SP/grid/search.log" 2>&1
{
  echo; echo "### RECHERCHE CIBLEE LACET"; echo
  tail -60 "$SP/grid/search.log"
  echo; echo "resultats complets : $SP/grid/search.log et $ROOT/results/yawsearch-*"
} >> "$ROOT/results/paper-rebuild/grid-report.txt"
cp "$SP/grid/search.log" "$ROOT/results/paper-rebuild/grid-yawsearch.log" 2>/dev/null
