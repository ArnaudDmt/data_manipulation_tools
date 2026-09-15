# Balayage et recherche sur le lacet de RHPS1

Ce que ces scripts ont produit le 2026-09-15, et comment les relancer.

## Le plan factoriel (`run.sh`, `apply.py`, `overlay.py`, `report.py`)

135 combinaisons croisees de quatre reglages sur les trois glissements, via le replay. La cellule
de reference reproduit `runs/clean` EXACTEMENT sur les quatre metriques : c'est le controle qui
valide le harnais, a verifier avant de croire quoi que ce soit d'autre.

Lire les **effets marginaux** par axe, pas le classement : a 135 cellules sur trois datasets, le
meilleur point est un tirage. Un reglage n'est retenu que s'il est monotone sur ses niveaux, qu'on
sait raconter son mecanisme, et qu'il a le meme signe sur les deux robots.

## Le criblage et la validation (`stage2.py`, `stage3.py`, `score_promoted.py`)

`stage2` crible chaque nouveau reglage SEUL autour du meilleur point, puis ne croise que les axes
actifs -- croiser moins de facteurs a pleine resolution vaut mieux que tronquer un produit
cartesien, qui n'explorerait qu'un coin de l'espace.

`stage3` passe les candidates sur les quatre multicontacts AVANT les cinq marches : les deux robots
reagissent en sens inverse a chaque parametre de contact, donc un gain sur une categorie ne prouve
rien. Le lacet y est une contrainte dure, les autres metriques ont un seuil de 2 %.

## La recherche TPE (`yawsearch.py`, `yawslip3.py`, `plan_search.py`)

Lanceurs de `kinetics_tune.py` qui imposent des bornes et des barreaux choisis, decoupent le wrench
non modelise en quatre dimensions que le tuner n'a pas, et remplacent `SCREEN_PROJECTS` -- sa liste
est CODEE EN DUR et ignore `--datasets`, ce qui fait echouer tous les essais si un dataset n'est pas
prepare. `search-pause.sh` et `search-resume.sh` arretent et reprennent sans perdre d'essai.

## Le resultat

Aucune configuration ne bat le reglage publie sur les quatre categories. Les deux meilleurs points
de la recherche gagnent 4 % de translation sur les glissements et s'effondrent sur les marches
(+21 % et +388 % de lacet). Le reglage actuel est un point d'equilibre etroit entre categories.
