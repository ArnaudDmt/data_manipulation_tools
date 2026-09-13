"""Pool the disturbance-wrench sweep's velocities and pick the process covariance to keep.

The sweep measured the relative errors already; the velocity is the metric that could still
contradict the choice, because it is the one the odometry tables do not show.

Decision rule, fixed in advance so the choice is not made to fit the numbers:
  - the default is the preferred value (0.09);
  - fall back to 0.5 if the preferred one degrades pooled vel_xy by more than TOLERANCE on any
    category AND 0.5 does materially better there;
  - keep the current 4 if every candidate degrades it by more than 2 x TOLERANCE.
TOLERANCE is 5%, the order of the relative-error costs already accepted for this trade.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as m

SWEEP = Path("/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/"
             "287daf9e-ad1c-4931-a07e-923142412c80/scratchpad")
# uw value -> directory holding <project>.<prefix>_loc_vel.pickle for that value.
SOURCES = {"4": SWEEP / "paper-variants/clean", "0.5": SWEEP / "uw-cost/uw0.5",
           "0.09": SWEEP / "uw-cost/uw0.09"}
PREFERRED, FALLBACK, CURRENT = "0.09", "0.5", "4"
TOLERANCE = 0.05


def pooled(directory, projects):
    xy, z = [], []
    for project in projects:
        estimate = pickle.load((directory / f"{project}.KO_loc_vel.pickle").open("rb"))["estimate"]
        # Each run resynchronises the ground truth against its own estimate; the mocap must come
        # from the same run or the pairing adds an alignment error of its own.
        truth = pickle.load((directory / f"{project}.mocap_loc_vel.pickle").open("rb"))["estimate"]
        error = {a: np.abs(np.asarray(estimate[a]) - np.asarray(truth[a])) for a in "xyz"}
        xy.append(np.linalg.norm(np.stack([error["x"], error["y"]], axis=-1), axis=-1))
        z.append(error["z"])
    return float(np.mean(np.concatenate(xy))), float(np.mean(np.concatenate(z)))


def main():
    table, missing = {}, []
    for value, directory in SOURCES.items():
        for category, (projects, _) in m.CATEGORIES.items():
            try:
                table[(value, category)] = pooled(directory, projects)
            except FileNotFoundError as error:
                missing.append(f"uw={value} {category}: {Path(error.filename).name}")

    lines = ["| categorie | uw | vel_xy (m/s) | vel_z (m/s) | ecart xy vs uw=4 |",
             "|---|---|---|---|---|"]
    print(f"\n{'categorie':18s} {'uw':>6s} {'vel_xy':>10s} {'vel_z':>10s} {'ecart xy':>10s}")
    worst = {}
    for category in m.CATEGORIES:
        base = table.get((CURRENT, category))
        for value in SOURCES:
            stats = table.get((value, category))
            if stats is None:
                continue
            delta = (stats[0] / base[0] - 1) if base else float("nan")
            if value != CURRENT:
                worst[value] = max(worst.get(value, -9.9), delta)
            mark = "" if value == CURRENT else f"{100 * delta:+.1f}%"
            print(f"{category:18s} {value:>6s} {stats[0]:10.4f} {stats[1]:10.4f} {mark:>10s}")
            lines.append(f"| {category} | {value} | {stats[0]:.4f} | {stats[1]:.4f} | {mark} |")

    if missing:
        print("\nMANQUANTS:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)

    # A candidate judged on a subset of the categories is not judged at all: the sweep writes its
    # snapshots dataset by dataset, so a partial run would otherwise look like a clean verdict.
    complete = {value for value in SOURCES
                if all((value, category) in table for category in m.CATEGORIES)}
    incomplete = sorted(set(SOURCES) - complete)
    if incomplete:
        print(f"\nincomplet, donc ecarte du choix: {', '.join(incomplete)}", file=sys.stderr)
    worst = {value: delta for value, delta in worst.items() if value in complete}
    if CURRENT not in complete:
        choice, why = CURRENT, ("la reference uw=4 est incomplete, aucune comparaison fiable; "
                                "rien n'est change")
    elif PREFERRED not in worst:
        choice, why = CURRENT, f"aucune donnee de vitesse pour uw={PREFERRED}; rien n'est change"
    elif worst[PREFERRED] <= TOLERANCE:
        choice, why = PREFERRED, (f"la preference tient: pire degradation vel_xy "
                                  f"{100 * worst[PREFERRED]:+.1f}% <= {100 * TOLERANCE:.0f}%")
    elif worst.get(FALLBACK, 9.9) <= TOLERANCE:
        choice, why = FALLBACK, (f"uw={PREFERRED} degrade vel_xy de {100 * worst[PREFERRED]:+.1f}%, "
                                 f"au-dela de {100 * TOLERANCE:.0f}%; uw={FALLBACK} reste a "
                                 f"{100 * worst[FALLBACK]:+.1f}%")
    elif min(worst.values()) > 2 * TOLERANCE:
        choice, why = CURRENT, (f"tous les candidats degradent vel_xy de plus de "
                                f"{200 * TOLERANCE:.0f}%; on garde la config actuelle")
    else:
        choice, why = FALLBACK, f"compromis: uw={FALLBACK} a {100 * worst[FALLBACK]:+.1f}%"

    print(f"\nCHOIX: uw = {choice}\nRAISON: {why}")
    out = m.WORK / "uw_choice"
    out.mkdir(parents=True, exist_ok=True)
    (out / "value").write_text(choice + "\n")
    (out / "report.md").write_text(
        "## Vitesses et choix du process de wrench\n\n" + "\n".join(lines)
        + f"\n\n**Choix : `uw = {choice}`** — {why}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
