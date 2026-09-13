"""Print the paper's tables from the distilled summary, for the overnight report.

Reads results_summary/relative_errors.json rather than the caches: the summary is exact for these
statistics and costs nothing to load.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as m

SUMMARY = m.ROOT / "results_summary"
NAMES = {"var-clean-ref": "KO", "var-zpc": "KO-ZPC", "var-noconstraint": "KO sans contrainte",
         "var-pc": "KO sans capteurs", "var-flex-div10": "flex /10", "var-flex-mul10": "flex x10",
         "riekf": "RI-EKF"}
METRICS = [("rel_trans_x_y_norm", "trans_xy", 4), ("rel_yaw", "yaw", 3), ("rel_tilt", "tilt", 3)]


def resolve(data, label):
    """The variant directories carry a configuration hash; match on the label prefix.

    Prefer a key that actually carries every dataset: an aborted run leaves a near-empty entry,
    and picking it would print a table of dashes instead of saying the run is missing.
    """
    if label in data:
        return label
    matches = sorted(k for k in data if k.startswith(label + "-"))
    complete = [k for k in matches if len(data[k]) >= len(m.ALL)]
    return (complete or matches)[-1] if matches else None


def pool(data, key, projects, length, metric):
    n = s = q = 0.0
    for project in projects:
        entry = data.get(key, {}).get(project, {}).get(str(length), {}).get(metric)
        if entry is None:
            return None
        n += entry["count"]; s += entry["sum_abs"]; q += entry["sum_sq"]
    if not n:
        return None
    mean = s / n
    return mean, (max(q / n - mean * mean, 0.0)) ** 0.5


def main():
    path = SUMMARY / "relative_errors.json"
    if not path.exists():
        print("pas de resume distille", file=sys.stderr)
        return 1
    data = json.loads(path.read_text())
    for category, (projects, length) in m.CATEGORIES.items():
        print(f"\n{category}  (sous-trajectoires {length} m)")
        header = f"{'estimateur':22s}" + "".join(f"{n:>14s}" for _, n, _ in METRICS)
        print(header); print("-" * len(header))
        reference = {}
        for label, name in NAMES.items():
            key = resolve(data, label)
            if key is None:
                continue
            row, empty = f"{name:22s}", True
            for metric, _, digits in METRICS:
                stats = pool(data, key, projects, length, metric)
                if stats is None:
                    row += f"{'-':>14s}"; continue
                empty = False
                if label == "var-clean-ref":
                    reference[metric] = stats[0]
                    row += f"{stats[0]:14.{digits}f}"
                else:
                    base = reference.get(metric)
                    mark = f" ({100 * (stats[0] / base - 1):+.0f}%)" if base else ""
                    row += f"{stats[0]:9.{digits}f}{mark:>5s}"
            if not empty:
                print(row)
    velocities = SUMMARY / "velocities.json"
    if velocities.exists():
        data = json.loads(velocities.read_text())
        print("\n\nVitesses (erreur locale moyenne, m/s)")
        print(f"{'variante':22s} {'dataset':32s} {'xy':>9s} {'z':>9s}")
        for variant in sorted(data):
            for project in sorted(data[variant]):
                for prefix, stats in sorted(data[variant][project].items()):
                    xy, z = stats["xy"], stats["z"]
                    print(f"{variant + '/' + prefix:22s} {project:32s} "
                          f"{xy['sum_abs'] / xy['count']:9.4f} {z['sum_abs'] / z['count']:9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
