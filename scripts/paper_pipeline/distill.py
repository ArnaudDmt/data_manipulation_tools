"""Reduce the evaluation caches to something small enough to keep under version control.

The caches hold one value per sub-trajectory -- 469597 of them for LongWalk alone, 330 MB over all
variants -- and the paper never reports those individually. What it reports is, per category, the
mean of the absolute values pooled across the category's trials, and their population standard
deviation.

Both are exactly recoverable from three numbers per trial and metric: the count, the sum of the
absolute values, and the sum of the squares. Pooling is then a weighted combination:

    mean  = sum(sum_abs) / sum(count)
    std   = sqrt(sum(sum_sq) / sum(count) - mean^2)

so this is a lossless summary of what the paper prints, not an approximation of it. Only per-
sub-trajectory distributions -- histograms, quantiles, the shape of the tail -- are lost, and
rebuilding those means rerunning the evaluation anyway.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as m

SUMMARY = m.ROOT / "results_summary"
METRICS = ("rel_trans_x_y_norm", "rel_trans_z", "rel_tilt", "rel_yaw", "rel_gravity", "rel_rot")


def moments(values):
    values = np.abs(np.asarray(values, dtype=float))
    return {"count": int(values.size),
            "sum_abs": float(values.sum()),
            "sum_sq": float(np.square(values).sum())}


def relative_errors():
    out = {}
    for cache in sorted(m.ROOT.glob("results/var-*/*/eval/saved_results/traj_est/cached/cached_rel_err.pickle")):
        variant, project = cache.parts[-7], cache.parts[-6]
        data = pickle.load(cache.open("rb"))
        for length, block in data.items():
            for metric in METRICS:
                if metric not in block:
                    continue
                out.setdefault(variant, {}).setdefault(project, {}).setdefault(
                    str(length), {})[metric] = moments(block[metric])
    # The RI-EKF baseline lives with the projects, not with a variant: it is the same offline
    # parse for every one of them.
    for project in m.ALL:
        cache = (m.ROOT / "Projects" / project
                 / "output_data/evals/Hartley/saved_results/traj_est/cached/cached_rel_err.pickle")
        if not cache.exists():
            continue
        data = pickle.load(cache.open("rb"))
        for length, block in data.items():
            for metric in METRICS:
                if metric in block:
                    out.setdefault("riekf", {}).setdefault(project, {}).setdefault(
                        str(length), {})[metric] = moments(block[metric])
    return out


def velocities():
    """Local linear velocity error, summarised the same way: xy norm and z."""
    out = {}
    for pickled in sorted(m.WORK.glob("runs/*/*/*_loc_vel.pickle")):
        variant, project = pickled.parts[-3], pickled.parts[-2]
        prefix = pickled.name[: -len("_loc_vel.pickle")]
        if prefix == "mocap":
            continue
        mocap = pickled.with_name("mocap_loc_vel.pickle")
        if not mocap.exists():
            continue
        estimate = pickle.load(pickled.open("rb"))["estimate"]
        truth = pickle.load(mocap.open("rb"))["estimate"]
        error = {a: np.abs(np.asarray(estimate[a]) - np.asarray(truth[a])) for a in "xyz"}
        out.setdefault(variant, {}).setdefault(project, {})[prefix] = {
            "xy": moments(np.linalg.norm(np.stack([error["x"], error["y"]], axis=-1), axis=-1)),
            "z": moments(error["z"])}
    return out


def main():
    SUMMARY.mkdir(parents=True, exist_ok=True)
    rpe = relative_errors()
    (SUMMARY / "relative_errors.json").write_text(json.dumps(rpe, indent=1, sort_keys=True))
    trials = sum(len(v) for v in rpe.values())
    print(f"erreurs relatives : {len(rpe)} variantes, {trials} essais")

    velocity = velocities()
    if velocity:
        (SUMMARY / "velocities.json").write_text(json.dumps(velocity, indent=1, sort_keys=True))
        print(f"vitesses : {sum(len(v) for v in velocity.values())} essais")
    else:
        print("vitesses : aucun pickle, l'etage routine n'a pas encore tourne")

    for source, name in ((m.WORK / "configs/clean/MCKineticsObserver.yaml", "MCKineticsObserver.yaml"),):
        if source.exists():
            (SUMMARY / name).write_text(source.read_text())
    overlays = SUMMARY / "overlays"
    overlays.mkdir(exist_ok=True)
    for overlay in sorted(m.ROOT.glob("results/var-*-overlay.yaml")):
        (overlays / overlay.name).write_text(overlay.read_text())
    macros = m.WORK / "macros"
    if macros.exists():
        target = SUMMARY / "macros"
        target.mkdir(exist_ok=True)
        for produced in sorted(macros.glob("*.tex")):
            (target / produced.name).write_text(produced.read_text())
    total = sum(p.stat().st_size for p in SUMMARY.rglob("*") if p.is_file())
    print(f"{SUMMARY.relative_to(m.ROOT)} : {total / 1024:.0f} Ko")


if __name__ == "__main__":
    main()
