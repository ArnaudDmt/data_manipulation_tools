#!/bin/bash
# Run the routine chain for every variant the paper needs, and snapshot what each pass produces
# before the next one overwrites it in place.
#
# Everything under Projects/<p>/output_data is rewritten by every pass, so a result that is not
# copied out here is lost. That is how the velocity pickles were dropped from the first
# disturbance-wrench sweep.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK="$ROOT/results/paper-rebuild"
cd "$ROOT" || exit 1
wanted=${1:-all}
failed=0

# Every variant runs under its own private HOME (config_home.py): the real ~/.config/mc_rtc is
# never written, so a crash cannot leave a variant installed and there is nothing to restore.
# KO_LIVE_CONFIG=1 brings back the former behaviour (install into ~/.config, restore at exit).
if [ -n "${KO_LIVE_CONFIG:-}" ]; then
  trap 'env/bin/python "$HERE/variant_install.py" clean >/dev/null 2>&1' EXIT INT TERM
fi

mapfile -t plan < <(env/bin/python - "$wanted" <<'PY'
import os
import sys
sys.path.insert(0, "scripts/paper_pipeline")
import manifest as m
wanted = sys.argv[1]
# Opt-in: restrict the pass to named datasets. Without KO_PROJECTS every dataset of the variant
# runs, as before. It exists because a pass is sometimes needed on a subset and nothing else can
# express that -- re-ticking the three datasets whose outputs a variant run overwrote would
# otherwise mean re-ticking all thirteen. An unknown name is refused rather than silently
# dropped, which would produce a pass that quietly did nothing.
known = set(m.ALL) | set(m.NO_LEFT_HAND) | set(m.ORI_ERROR)
only = os.environ.get("KO_PROJECTS", "").replace(",", " ").split()
unknown = [p for p in only if p not in known]
if unknown:
    sys.exit(f"ABANDON: jeu de donnees inconnu dans KO_PROJECTS: {' '.join(unknown)}")
for name, (argument, datasets, _) in m.VARIANTS.items():
    if wanted in ("all", name):
        if name in m.SHARED_ROUTINE_OBSERVERS:
            if name != "clean" and wanted == "all":
                continue
            name, argument, datasets = "clean", "clean", m.ALL
        if only:
            datasets = [p for p in datasets if p in only]
            if not datasets:
                continue
        print(f"{name}\t{argument}\t{' '.join(datasets)}")
PY
)
[ "${#plan[@]}" -gt 0 ] || { echo "ABANDON: aucune variante ne correspond a '$wanted'"; exit 1; }

for line in "${plan[@]}"; do
  IFS=$'\t' read -r name argument datasets <<< "$line"
  echo "############ variante $name ($argument)"
  store="$WORK/runs/$name"
  if [ -n "${KO_LIVE_CONFIG:-}" ]; then
    unset KO_CONFIG_HOME
    installed=$(env/bin/python "$HERE/variant_install.py" "$argument")
  else
    export KO_CONFIG_HOME="$WORK/homes/$name"
    installed=$(cd "$HERE" && ../../env/bin/python variant_install.py --home "$KO_CONFIG_HOME" "$argument")
  fi
  if [ $? -ne 0 ]; then echo "[$name] INSTALLATION ECHOUEE, variante ignoree"; failed=1; continue; fi
  echo "$installed"
  for p in $datasets; do
    echo "================ $name / $p"
    mkdir -p "$store/$p"
    # Opt-in: a change to the RI-EKF's own configuration makes its cached parse stale. With
    # KO_REBUILD_HARTLEY_CLEAN set, the clean pass rebuilds it from this very tick's input
    # (chain.sh -> rebuild_hartley.sh) and every later variant copies the refreshed parse.
    if [ "$name" = "clean" ] && [ -n "${KO_REBUILD_HARTLEY_CLEAN:-}" ]; then
      case "$p" in HRP5*) export KO_REBUILD_HARTLEY=hrp5_p ;; *) export KO_REBUILD_HARTLEY=rhps1 ;; esac
    else
      unset KO_REBUILD_HARTLEY
    fi
    if ! "$HERE/chain.sh" "$p"; then echo "[$name/$p] ABANDONNE"; failed=1; continue; fi
    if [ "$name" = "clean" ]; then
      env/bin/python "$HERE/snapshot_shared.py" "$p" || exit 1
      continue
    fi
    out="Projects/$p/output_data"
    cache="$out/evals/KO/saved_results/traj_est/cached/cached_rel_err.pickle"
    # Provenance: the exact configuration files and their digest travel with the results.
    if [ -n "${KO_CONFIG_HOME:-}" ]; then
      (cd "$HERE" && ../../env/bin/python -c "import config_home as c; c.keep_provenance('$KO_CONFIG_HOME', '$store/$p')") \
        || echo "[$name/$p] PROVENANCE ECHOUEE"
    fi
    [ -f "$cache" ] && cp "$cache" "$store/$p/cached_rel_err.pickle"
    # The mocap travels with the estimate: the routine resynchronises the ground truth on
    # every run, so pairing a run's velocities with another run's mocap is wrong.
    for f in KO_loc_vel Hartley_loc_vel Tilt_loc_vel mocap_loc_vel; do
      [ -f "$out/$f.pickle" ] && cp "$out/$f.pickle" "$store/$p/$f.pickle"
    done
    # A decimated trajectory, so the SHAPE of a run's error stays answerable: the full file is
    # 70 MB and the next variant overwrites it, and the error cache holds statistics, not a signal.
    env/bin/python "$HERE/keep_traj.py" "$out/evals" "$store/$p" \
      || echo "[$name/$p] TRAJECTOIRE DECIMEE ECHOUEE"
    # The disturbance-wrench table needs the estimated wrench against the hand sensor. Keep only
    # those columns: the full log is 35 MB a trial and nothing else here reads it.
    if [ "$name" = "hidehand" ]; then
      env/bin/python "$HERE/extract_wrench.py" "$out/logReplay.csv" "$store/$p/wrench.csv" \
        || echo "[$name/$p] EXTRACTION WRENCH ECHOUEE"
    fi
  done
done

# Live mode only: never leave a variant installed in ~/.config.
[ -n "${KO_LIVE_CONFIG:-}" ] && env/bin/python "$HERE/variant_install.py" clean
echo "STAGE_ROUTINE_DONE"
exit "$failed"
