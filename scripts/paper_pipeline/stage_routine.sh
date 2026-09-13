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

# A variant left installed -- a hidden hand, a 30 deg orientation error -- silently
# contaminates every later tick, so restore the retained tuning whatever happens.
trap 'env/bin/python "$HERE/variant_install.py" clean >/dev/null 2>&1' EXIT INT TERM

mapfile -t plan < <(env/bin/python - "$wanted" <<'PY'
import sys
sys.path.insert(0, "scripts/paper_pipeline")
import manifest as m
wanted = sys.argv[1]
for name, (argument, datasets, _) in m.VARIANTS.items():
    if wanted in ("all", name):
        print(f"{name}\t{argument}\t{' '.join(datasets)}")
PY
)
[ "${#plan[@]}" -gt 0 ] || { echo "ABANDON: aucune variante ne correspond a '$wanted'"; exit 1; }

for line in "${plan[@]}"; do
  IFS=$'\t' read -r name argument datasets <<< "$line"
  echo "############ variante $name ($argument)"
  if ! env/bin/python "$HERE/variant_install.py" "$argument"; then
    echo "[$name] INSTALLATION ECHOUEE, variante ignoree"; continue
  fi
  store="$WORK/runs/$name"
  for p in $datasets; do
    echo "================ $name / $p"
    mkdir -p "$store/$p"
    if ! "$HERE/chain.sh" "$p"; then echo "[$name/$p] ABANDONNE"; continue; fi
    out="Projects/$p/output_data"
    cache="$out/evals/KO/saved_results/traj_est/cached/cached_rel_err.pickle"
    [ -f "$cache" ] && cp "$cache" "$store/$p/cached_rel_err.pickle"
    # The mocap travels with the estimate: the routine resynchronises the ground truth on
    # every run, so pairing a run's velocities with another run's mocap is wrong.
    for f in KO_loc_vel Hartley_loc_vel mocap_loc_vel; do
      [ -f "$out/$f.pickle" ] && cp "$out/$f.pickle" "$store/$p/$f.pickle"
    done
    # The disturbance-wrench table needs the estimated wrench against the hand sensor. Keep only
    # those columns: the full log is 35 MB a trial and nothing else here reads it.
    if [ "$name" = "hidehand" ]; then
      env/bin/python "$HERE/extract_wrench.py" "$out/logReplay.csv" "$store/$p/wrench.csv" \
        || echo "[$name/$p] EXTRACTION WRENCH ECHOUEE"
    fi
  done
done

# Never leave a variant installed: a stale 30 deg orientation error or a hidden hand would
# silently contaminate every later tick.
env/bin/python "$HERE/variant_install.py" clean
echo "STAGE_ROUTINE_DONE"
