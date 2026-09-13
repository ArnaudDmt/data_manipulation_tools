#!/bin/bash
# Re-tick the slippage dataset with the second observer instance, then redraw the two trajectory
# figures that failed.
#
# slipping-odom-traj is supposed to carry the KO-ZPC curve -- the published figure never got it --
# and that curve comes from a second MCKineticsObserver instance in the controller pipeline, which
# only chain.sh installs and only for the projects manifest.NEEDS_KOZPC names. friends-traj simply
# asked for a curve it was never meant to have; its estimator list is now explicit.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
WORK="$ROOT/results/paper-rebuild"
REPORT="$WORK/overnight-report.md"
cd "$ROOT" || exit 1
note() { printf '%s\n' "$*" >> "$REPORT"; }

echo "attente de la fin du correctif pc"
while pgrep -f "paper_pipeline/fixup_pc.sh" > /dev/null; do sleep 30; done

note ""
note "## Correctif: figures de trajectoire"

P=KO_TRO_2024_RHPS1_SLIPPAGE_1
echo "=== re-tick de $P avec l'instance KOZPC [$(date +%H:%M:%S)]"
if "$HERE/chain.sh" "$P"; then
  note "- OK $P re-tique avec la seconde instance"
  # The snapshot has to follow: the metrics must come from the same tick as the figure.
  store="$WORK/runs/clean/$P"; mkdir -p "$store"
  out="Projects/$P/output_data"
  cp "$out/evals/KO/saved_results/traj_est/cached/cached_rel_err.pickle" "$store/cached_rel_err.pickle" 2>/dev/null
  for f in KO_loc_vel Hartley_loc_vel mocap_loc_vel; do
    cp "$out/$f.pickle" "$store/$f.pickle" 2>/dev/null
  done
else
  note "- ECHEC du re-tick de $P; slipping-odom-traj restera sans KO-ZPC"
fi

echo "=== figures [$(date +%H:%M:%S)]"
for figure in slipping-odom-traj friends-traj; do
  if env/bin/python "$HERE/figures.py" "$figure"; then
    note "- OK figure $figure"
  else
    note "- ECHEC figure $figure"
  fi
done

env/bin/python "$HERE/distill.py" > /dev/null && note "- OK resume redistille"
echo "FIXUP_FIGURES_DONE"
note "_correctif figures termine a $(date '+%H:%M')_"
