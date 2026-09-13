"""Rebuild every measured number the paper prints, from the snapshots the run stages left.

Three families, all pooled the way generate_metrics_plots pools them -- concatenate the raw
per-sample arrays across a category's trials, then take the mean of the absolute values and the
population standard deviation:

  relative error  RPE from the replay caches (results/var-<label>-*), RI-EKF from the routine;
  velocity        local linear velocity, from the routine's pickles, each run with its own mocap;
  disturbance wrench  the hand-removal variant, against the sensor still logged at the hand.

Writes one macro file per family, then folds them into the paper's metrics_results.tex.
"""
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as m

MACROS = m.WORK / "macros"
RPE_METRICS = {"rel_trans_x_y_norm": "Transxy", "rel_trans_z": "Transz",
               "rel_tilt": "Tilt", "rel_yaw": "Yaw"}
ANGLES = {"Tilt", "Yaw"}


def latest_variant_dir(label):
    """results/var-<label>-<hash>/ -- the hash changes with the configuration, so resolve it."""
    candidates = sorted((m.ROOT / "results").glob(f"{label}-*"),
                        key=lambda p: p.stat().st_mtime)
    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        raise SystemExit(f"aucun repertoire de resultats pour {label}")
    return candidates[-1]


def pool(arrays):
    values = np.abs(np.concatenate(arrays))
    return float(np.mean(values)), float(np.std(values))


# --- relative error ---------------------------------------------------------------------------

def rpe_macros():
    lines, report = [], []
    replay = {name: latest_variant_dir(label) for name, label in m.REPLAY_LABELS.items()}

    def cache(directory, project):
        return directory / project / "eval/saved_results/traj_est/cached/cached_rel_err.pickle"

    def riekf(project):
        return (m.ROOT / "Projects" / project
                / "output_data/evals/Hartley/saved_results/traj_est/cached/cached_rel_err.pickle")

    def stats(paths, distance):
        bags = {key: [] for key in RPE_METRICS}
        for path in paths:
            data = pickle.load(path.open("rb"))
            if distance not in data:
                raise SystemExit(f"{path} n'a pas la longueur {distance}: {sorted(data)}")
            for key in bags:
                bags[key].append(np.asarray(data[distance][key], dtype=float))
        return {name: pool(bags[key]) for key, name in RPE_METRICS.items()}

    def emit(category, estimator, values):
        for name, (mean, std) in values.items():
            digits = 2 if name in ANGLES else 3
            lines.append(f"\\newcommand{{\\{category}{estimator}Relerror{name}Meanabs}}"
                         f"{{{mean:.{digits}f}}}")
            lines.append(f"\\newcommand{{\\{category}{estimator}Relerror{name}Std}}"
                         f"{{{std:.{digits}f}}}")
        report.append((category, estimator, values))

    for category, (projects, distance) in m.CATEGORIES.items():
        for variant, (_, _, estimator) in m.VARIANTS.items():
            if estimator is None or variant not in replay:
                continue
            emit(category, estimator, stats([cache(replay[variant], p) for p in projects], distance))
        emit(category, "Hartley", stats([riekf(p) for p in projects], distance))

    # Flexibility ablation: same estimator, retuned contact stiffness, two categories only.
    for variant, suffix in m.FLEX_SUFFIX.items():
        for category in m.FLEX_CATEGORIES:
            projects, distance = m.CATEGORIES[category]
            emit(category + suffix, "Kineticsobserver",
                 stats([cache(replay[variant], p) for p in projects], distance))
    return lines, report


# --- velocity ---------------------------------------------------------------------------------

def velocity_macros():
    lines, report, missing = [], [], []

    def estimate(path):
        with path.open("rb") as stream:
            return pickle.load(stream)["estimate"]

    def stats(projects, variant, prefix):
        xy, z = [], []
        for project in projects:
            source = m.WORK / "runs" / variant / project
            values = estimate(source / f"{prefix}_loc_vel.pickle")
            # Each run resynchronises the ground truth against its own estimate, so the mocap
            # must come from the same run: pairing across runs adds an alignment error that has
            # nothing to do with the estimator.
            mocap = estimate(source / "mocap_loc_vel.pickle")
            error = {a: np.abs(np.asarray(values[a]) - np.asarray(mocap[a])) for a in "xyz"}
            xy.append(np.linalg.norm(np.stack([error["x"], error["y"]], -1), -1))
            z.append(error["z"])
        return {"EstimateXy": pool(xy), "EstimateZ": pool(z)}

    series = [(c, e, v, p)
              for v, (_, _, e) in m.VARIANTS.items() if e is not None
              for c in m.CATEGORIES for p in ("KO",)]
    series += [(c, "Hartley", "clean", "Hartley") for c in m.CATEGORIES]
    series += [(c + s, "Kineticsobserver", v, "KO")
               for v, s in m.FLEX_SUFFIX.items() for c in m.FLEX_CATEGORIES]

    for category, estimator, variant, prefix in series:
        base = category[:-1] if category[-1] in "bc" and category[:-1] in m.CATEGORIES else category
        projects = m.CATEGORIES[base][0]
        try:
            values = stats(projects, variant, prefix)
        except FileNotFoundError as error:
            missing.append(f"{category}/{estimator}: {Path(error.filename).name}")
            continue
        for name, (mean, std) in values.items():
            lines.append(f"\\newcommand{{\\{category}{estimator}Velerror{name}Meanabs}}{{{mean:.3f}}}")
            lines.append(f"\\newcommand{{\\{category}{estimator}Velerror{name}Std}}{{{std:.3f}}}")
        report.append((category, estimator, values))
    return lines, report, missing


# --- disturbance wrench -------------------------------------------------------------------------

PREFIX = "Observers_MainObserverPipeline_MCKineticsObserver"
AXES = [("force", "extForceCentr", a) for a in "xyz"] + \
       [("torque", "extTorqueCentr", a) for a in "xyz"]


def wrench_macros():
    bags = {f"{kind[0].upper()}{axis}": [] for kind, _, axis in AXES}
    trials = 0
    for project in m.NO_LEFT_HAND:
        path = m.WORK / "runs/hidehand" / project / "wrench.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, sep=";")
        for kind, state, axis in AXES:
            reference = frame[f"{PREFIX}_debug_wrenchesInCentroid_LeftHandForceSensor_{kind}_{axis}"]
            bags[f"{kind[0].upper()}{axis}"].append(
                frame[f"{PREFIX}_MEKF_estimatedState_{state}_{axis}"].to_numpy()
                - reference.to_numpy())
        trials += 1
    if not trials:
        return [], [], 0
    lines, report = [], []
    for name, arrays in bags.items():
        mean, std = pool(arrays)
        lines.append(f"\\newcommand{{\\Extwrench{name}Meanabs}}{{{mean:.1f}}}")
        lines.append(f"\\newcommand{{\\Extwrench{name}Std}}{{{std:.1f}}}")
        report.append((name, mean, std))
    return lines, report, trials


# --- output -------------------------------------------------------------------------------------

def busy():
    """A chain in flight rewrites Projects/<p>/output_data underneath us.

    Pooling the RI-EKF while its caches were being regenerated trial by trial once produced a
    table mixing two runs, and nothing in the output said so. Refuse instead.
    """
    probe = subprocess.run(["pgrep", "-af", "chain.sh|paper_chain.sh|kinetics_eval.py"],
                           capture_output=True, text=True)
    return [line for line in probe.stdout.splitlines() if "pgrep" not in line]


def main():
    running = busy()
    if running and "--force" not in sys.argv:
        print("ABANDON: une chaine tourne encore, les caches sont en cours de reecriture:",
              file=sys.stderr)
        for line in running:
            print(f"  {line}", file=sys.stderr)
        print("  (--force pour passer outre)", file=sys.stderr)
        raise SystemExit(1)
    MACROS.mkdir(parents=True, exist_ok=True)
    written = []

    rpe, rpe_report = rpe_macros()
    (MACROS / "relerror.tex").write_text("\n".join(rpe) + "\n")
    written.append(MACROS / "relerror.tex")
    print(f"\n=== erreurs relatives ({len(rpe)} macros) ===")
    print(f"{'categorie':20} {'estimateur':24} {'trans_xy':>10} {'yaw':>8} {'tilt':>8}")
    for category, estimator, values in rpe_report:
        print(f"{category:20} {estimator:24} {values['Transxy'][0]:10.3f} "
              f"{values['Yaw'][0]:8.2f} {values['Tilt'][0]:8.2f}")

    velocity, velocity_report, missing = velocity_macros()
    if velocity:
        (MACROS / "velerror.tex").write_text("\n".join(velocity) + "\n")
        written.append(MACROS / "velerror.tex")
    print(f"\n=== vitesses ({len(velocity)} macros) ===")
    for category, estimator, values in velocity_report:
        print(f"{category:20} {estimator:24} xy {values['EstimateXy'][0]:.3f} "
              f"z {values['EstimateZ'][0]:.3f}")
    for item in missing:
        print(f"  MANQUANT {item}", file=sys.stderr)

    wrench, wrench_report, trials = wrench_macros()
    if wrench:
        (MACROS / "extwrench.tex").write_text("\n".join(wrench) + "\n")
        written.append(MACROS / "extwrench.tex")
        print(f"\n=== wrench de perturbation ({trials}/{len(m.NO_LEFT_HAND)} essais) ===")
        for name, mean, std in wrench_report:
            print(f"  {name}  MAE {mean:6.1f}   std {std:6.1f}")
    else:
        print("\n=== wrench de perturbation: aucun essai, variante hidehand non rejouee ===",
              file=sys.stderr)

    if "--no-install" in sys.argv:
        print(f"\n{len(written)} fichiers de macros ecrits sous {MACROS} (pas d'installation)")
        return
    merge = Path(__file__).resolve().parent / "merge_macros.py"
    subprocess.run([sys.executable, str(merge), *map(str, written)], check=True)


if __name__ == "__main__":
    main()
