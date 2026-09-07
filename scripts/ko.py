#!/usr/bin/env python3
"""One entry point for reading the Kinetics-vs-RI-EKF results.

Every question asked during a tuning run -- how far along is it, what does the best config look
like in real units, which of the goal constraints does it satisfy, how do the recorded trials rank
under a changed objective -- shares the same three inputs: the trial ledgers, the installed
tuning's evaluation, and the RI-EKF reference values. Recomputing that from scratch each time is
what makes a status check expensive.

    ko.py status                    where the running search is, and the machine's headroom
    ko.py best [--trial ID]         the leading config, per dataset, in metres/degrees/m per s
    ko.py goal                      which of the stated constraints each config satisfies
    ko.py rescore                   re-rank every recorded trial under the current objective
    ko.py slippage [--trial ID]     how much each estimator degrades when the feet slip

`best` and `slippage` default to the best trial of the newest run; pass --trial RUN#N to pick one.
"""

import argparse
import collections
import csv
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import kinetics_tune as kt

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path(__file__).parent / ".ko_cache.json"
INSTALLED_RUN = "results/baseline-newmetrics-2e07a56ca4"

# Bold only when writing to a terminal: piped into a file or a grep the escapes are noise, and
# --plain forces them off.
_COLOUR = sys.stdout.isatty() and "--plain" not in sys.argv
BOLD, PLAIN = ("\033[1m", "\033[0m") if _COLOUR else ("", "")
METRICS = ("trans_xy", "trans_z", "yaw", "tilt", "vel_xy", "vel_z")
UNITS = {"trans_xy": "m", "trans_z": "m", "yaw": "deg", "tilt": "deg",
         "vel_xy": "m/s", "vel_z": "m/s"}
RELATIVE = ("trans_xy", "trans_z", "yaw", "tilt")     # per sub-trajectory; the rest are whole-run
FAMILY = {"MultiContact": [f"HRP5_MultiContact_{i}" for i in range(1, 5)],
          "LongWalk": ["HRP5P_LongWalk"],
          "RHPS1": [f"KO_TRO2024_RHPS1_{i}" for i in range(1, 6)],
          "Slippage": [f"KO_TRO_2024_RHPS1_SLIPPAGE_{i}" for i in range(1, 4)]}
SHORT = (("HRP5_MultiContact_", "MC_"), ("KO_TRO2024_RHPS1_", "RHPS1_"),
         ("KO_TRO_2024_RHPS1_SLIPPAGE_", "SLIP_"), ("HRP5P_LongWalk", "LongWalk"))

# The goal, as stated: beat the RI-EKF on most metrics, by more than 15% on horizontal
# translation and yaw, stay within 15% on velocity and tilt, within 5% on vertical translation,
# and degrade less than the RI-EKF does when the feet slip.
TARGET = {"trans_xy": ("<=", 0.85), "yaw": ("<=", 0.85), "trans_z": ("<=", 1.05),
          "tilt": ("<=", 1.15), "vel_xy": ("<=", 1.15), "vel_z": ("<=", 1.15)}
SLIP_TARGET = 1.00


def short(project):
    for long, brief in SHORT:
        if project.startswith(long):
            return project.replace(long, brief)
    return project


def geo(values):
    return math.exp(sum(math.log(max(v, 1e-9)) for v in values) / len(values))


def reference():
    """RI-EKF error per dataset and metric, plus the installed tuning's. Cached on disk."""
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    directory = ROOT / INSTALLED_RUN
    riekf, installed, segment = collections.defaultdict(dict), collections.defaultdict(dict), {}
    for row in csv.DictReader((directory / "summary.csv").open()):
        if row["statistic"] == "mean" and row["metric"] in RELATIVE:
            target = riekf if row["estimator"] == "RI-EKF" else installed
            target[row["project"]][row["metric"]] = float(row["value"])
            segment[row["project"]] = float(row["distance"])
    import numpy as np
    cache = kt.reference_velocities(kt.PROJECTS)
    for name in kt.PROJECTS:
        grid, mocap, baseline, settings = cache[name]
        pose = np.loadtxt(directory / name / "kinetics.txt", comments="#", ndmin=2)
        velocity = np.loadtxt(directory / name / "kinetics_velocity.txt", comments="#", ndmin=2)
        own = kt.velocity_errors(kt.estimated_local_velocity(pose, velocity, grid, settings), mocap)
        for metric, mine, theirs in (("vel_xy", own[0], baseline[0]), ("vel_z", own[1], baseline[1])):
            installed[name][metric] = float(mine)
            riekf[name][metric] = float(theirs)
    data = {"riekf": riekf, "installed": installed, "segment": segment}
    CACHE.write_text(json.dumps(data))
    return data


def trials(pattern="results/kinetics-*/trials.csv"):
    """Every successful full-set trial across every recorded search, newest run last."""
    out = []
    for path in sorted(glob.glob(str(ROOT / pattern))):
        run = Path(path).parent.name
        for row in csv.DictReader(open(path)):
            if row["status"] != "ok" or row["ratios"].strip() in ("", "{}"):
                continue
            ratios = json.loads(row["ratios"])
            if len(ratios) < 70:
                continue                      # a subset study, not comparable
            out.append({"id": f"{run.split('-')[-1]}#{row['trial']}", "run": run,
                        "trial": row["trial"], "J": float(row["objective"]), "ratios": ratios})
    return out


def by_project(ratios):
    out = {}
    for case, value in ratios.items():
        project, metric = case.split("|", 1)
        out.setdefault(project, {})[metric] = value
    return out


def per_metric(ratios):
    out = collections.defaultdict(list)
    for case, value in ratios.items():
        out[metric_name(case)].append(value)
    return out


def metric_is_relative(metric):
    """Relative metrics accumulate over a sub-trajectory; velocity is an RMS over the whole run."""
    return metric in RELATIVE


def metric_name(case):
    return case.split("|", 1)[1]


def pick(rows, wanted):
    if wanted:
        match = [r for r in rows if r["id"] == wanted or r["trial"] == wanted]
        if not match:
            sys.exit(f"no trial matching {wanted!r}")
        return match[0]
    newest = max(r["run"] for r in rows)
    return min((r for r in rows if r["run"] == newest), key=lambda r: r["J"])


def slippage(ratios, metric):
    """KO degradation / RI-EKF degradation under slippage; below 1 means the KO degrades less."""
    grouped = by_project(ratios)
    nominal = [grouped[p][metric] for p in FAMILY["RHPS1"] if metric in grouped.get(p, {})]
    slipping = [grouped[p][metric] for p in FAMILY["Slippage"] if metric in grouped.get(p, {})]
    return geo(slipping) / geo(nominal) if nominal and slipping else float("nan")


# --------------------------------------------------------------------------- commands

def cmd_status(args):
    running = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
    # only the python process counts: watchdog and guard scripts carry the same words
    live = [l for l in running.splitlines()
            if "kinetics_tune.py --tracking" in l and "shell-snapshot" not in l
            and ("python" in l.split("kinetics_tune.py")[0])]
    # Follow the running search if there is one: its ledger may not exist yet, in which case the
    # newest on disk belongs to a previous run and reporting it would be misleading.
    tracked = None
    for line in live:
        parts = line.split()
        if "--tracking" in parts:
            tracked = Path(parts[parts.index("--tracking") + 1]) / "trials.csv"
    ledgers = sorted(glob.glob(str(ROOT / "results/kinetics-*/trials.csv")), key=os.path.getmtime)
    if tracked is not None:
        ledgers = [str(tracked)] if tracked.exists() else []
        if not ledgers:
            print(f"run         {tracked.parent.name}  (no trials finished yet)")
    print(f"search      {'RUNNING' if live else 'not running'}")
    if ledgers:
        latest = Path(ledgers[-1])
        rows = list(csv.DictReader(latest.open()))
        states = collections.Counter(r["status"] for r in rows)
        secs = [float(r["seconds"]) for r in rows] or [0]
        print(f"run         {latest.parent.name}")
        print(f"trials      {len(rows)}  {dict(states)}   mean {sum(secs)/len(secs):.0f}s/trial")
        ok = [r for r in rows if r["status"] == "ok"]
        if ok:
            best = min(ok, key=lambda r: float(r["objective"]))
            print(f"best        #{best['trial']}  J={float(best['objective']):+.4f}   "
                  f"wins {best['wins']}/{best['cases']}")
        print(f"updated     {time.strftime('%H:%M:%S', time.localtime(latest.stat().st_mtime))}"
              f"   (now {time.strftime('%H:%M:%S')})")
    free = shutil.disk_usage("/").free / 2**30
    mem = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.splitlines()[1].split()
    recorders = running.count("ros2 bag record")
    print(f"machine     disk {free:.0f}GB free   mem {mem[6]}GB avail   recorders {recorders}")


def cmd_best(args):
    reference_data = reference()
    riekf, installed, segment = reference_data["riekf"], reference_data["installed"], reference_data["segment"]
    rows = trials()
    if not rows:
        sys.exit("no trials recorded yet")
    row = pick(rows, args.trial)
    grouped, metrics = by_project(row["ratios"]), per_metric(row["ratios"])
    print(f"{row['id']}   J={row['J']:+.4f}   ({len(row['ratios'])} cases)")
    print("error per estimator; the lowest of each three is in bold\n")
    est = ("tuned", "installed", "RI-EKF")
    w = 11

    def line_for(label, seg, values):
        """One line: three estimators per metric, the smallest of each triple in bold."""
        line = f"{label:<17}{seg:<10}"
        for triple, fmt in values:
            best = min(triple)
            for value in triple:
                text = f"{value:{fmt}}"
                line += (BOLD + text + PLAIN).ljust(w + len(BOLD) + len(PLAIN)) if value == best \
                        else text.ljust(w)
        return line.rstrip()

    def header(metrics, first):
        top = f"{first:<17}{'':<10}" + "".join(f"{m + ' [' + UNITS[m] + ']':<{3 * w}}" for m in metrics)
        sub = f"{'':<17}{'segment':<10}" + "".join("".join(f"{e:<{w}}" for e in est) for _ in metrics)
        return top + "\n" + sub + "\n" + "-" * len(sub)

    for group in (METRICS[:3], METRICS[3:]):
        print(header(group, "experiment"))
        for project in kt.PROJECTS:
            seg = f"{segment[project]:g} m" if metric_is_relative(group[0]) else "whole run"
            values = []
            for metric in group:
                base = riekf[project][metric]
                fmt = ".4f" if metric.startswith(("trans", "vel")) else ".3f"
                values.append(((base * grouped[project][metric], installed[project][metric], base), fmt))
            print(line_for(short(project), seg, values))
        print()

    # Mean error within each family. Averaging inside a family is meaningful because its datasets
    # share a sub-trajectory length; across families it would not be. LongWalk is a single run, so
    # its mean is that run.
    print("MEAN ERROR BY EXPERIMENT FAMILY\n")
    for group in (METRICS[:3], METRICS[3:]):
        print(header(group, "family"))
        for family, members in FAMILY.items():
            mean = lambda values: sum(values) / len(values)
            values = []
            for metric in group:
                fmt = ".4f" if metric.startswith(("trans", "vel")) else ".3f"
                values.append(((mean([riekf[p][metric] * grouped[p][metric] for p in members]),
                                mean([installed[p][metric] for p in members]),
                                mean([riekf[p][metric] for p in members])), fmt))
            print(line_for(f"{family} ({len(members)})", "", values))
        print()
    print(f"{'metric':<14}{'geomean':>9}{'wins':>8}{'target':>9}{'status':>9}")
    for metric in METRICS:
        value, wins = geo(metrics[metric]), sum(v < 1 for v in metrics[metric])
        op, bound = TARGET[metric]
        print(f"{metric:<14}{value:9.3f}{wins:>5}/13{bound:9.2f}"
              f"{'ok' if value <= bound else 'MISSES':>9}")
    for metric in ("trans_xy", "yaw"):
        value = slippage(row["ratios"], metric)
        print(f"{'slip ' + metric:<14}{value:9.3f}{'':>8}{SLIP_TARGET:9.2f}"
              f"{'ok' if value <= SLIP_TARGET else 'MISSES':>9}")


def cmd_goal(args):
    rows = trials()
    print(f"{len(rows)} full-set configs. Per-family geomean must beat the RI-EKF on the two "
          f"metrics the goal names,\nand the capped metrics must stay inside their bound.\n")
    checks = [(f"{family}.{metric}", lambda r, f=family, m=metric:
               geo([by_project(r)[p][m] for p in FAMILY[f] if p in by_project(r)]), 1.0)
              for family in FAMILY for metric in ("trans_xy", "yaw")]
    checks += [(f"cap.{m}", lambda r, m=m: geo(per_metric(r)[m]), TARGET[m][1])
               for m in ("trans_z", "tilt", "vel_xy", "vel_z")]
    checks += [(f"slip.{m}", lambda r, m=m: slippage(r, m), SLIP_TARGET) for m in ("trans_xy", "yaw")]
    print(f"{'constraint':24}{'satisfied':>12}{'best':>9}{'by':>18}")
    feasible = [r for r in rows]
    for label, fn, bound in checks:
        values = [(fn(r["ratios"]), r["id"]) for r in rows]
        met = sum(1 for v, _ in values if v <= bound)
        best, who = min(values)
        print(f"{label:24}{met:>6}/{len(rows):<5}{best:9.3f}{who:>18}"
              + ("" if met else "   NEVER"))
        feasible = [r for r in feasible if fn(r["ratios"]) <= bound]
    print(f"\nconfigs satisfying every constraint: {len(feasible)}")
    for r in feasible[:5]:
        print(f"   {r['id']}  J={r['J']:+.4f}")


def constraints(ratios):
    """Every requirement the tuning was given, as {name: (value, limit)}.

    Ranking by J alone is not enough at selection time: the objective can buy a cap with a large
    regression on a priority metric, which is how the J-best config ended up 19% worse on yaw.
    """
    byp, per = by_project(ratios), per_metric(ratios)
    out = {}
    for family, members in FAMILY.items():
        for metric in ("trans_xy", "yaw"):
            values = [byp[p][metric] for p in members if p in byp and metric in byp[p]]
            if values:
                out[f"{family}.{metric}"] = (geo(values), 1.0)
    for metric in ("trans_xy", "yaw"):
        out[f"slippage.{metric}"] = (slippage(ratios, metric), SLIP_TARGET)
    for metric, (_, limit) in TARGET.items():
        if metric in per and metric in ("trans_z", "tilt", "vel_xy", "vel_z"):
            out[f"cap.{metric}"] = (geo(per[metric]), limit)
    return out


def cmd_select(args):
    """Rank configurations by how many requirements they meet, not by the objective."""
    rows = trials()
    if not rows:
        sys.exit("no trials yet")
    if args.run:
        rows = [r for r in rows if args.run in r["run"]]
    scored = []
    for row in rows:
        c = constraints(row["ratios"])
        met = [k for k, (v, lim) in c.items() if v <= lim]
        missed = {k: v / lim for k, (v, lim) in c.items() if v > lim}
        per = per_metric(row["ratios"])
        # tie-break on the two priority metrics, then on how badly the misses miss
        priority = geo([geo(per["trans_xy"]), geo(per["yaw"])])
        scored.append((-len(met), priority, max(missed.values(), default=1.0), row, met, missed))
    scored.sort(key=lambda s: (s[0], s[1], s[2]))
    total = len(constraints(rows[0]["ratios"]))
    print(f"{len(rows)} configurations, ranked by requirements met (of {total}), "
          f"then by trans_xy x yaw\n")
    print(f"{'rank':>4} {'trial':>16} {'met':>5} {'J':>9} {'trans_xy':>9}{'yaw':>7}"
          f"{'trans_z':>8}{'tilt':>7}{'vel_xy':>8}{'vel_z':>7}   worst miss")
    for i, (neg, prio, worst, row, met, missed) in enumerate(scored[:args.top], 1):
        per = per_metric(row["ratios"])
        g = {m: geo(per[m]) for m in METRICS if m in per}
        name, factor = max(missed.items(), key=lambda kv: kv[1], default=("none", 1.0))
        print(f"{i:>4} {row['id']:>16} {-neg:>3}/{total} {row['J']:+9.4f}"
              f"{g['trans_xy']:9.3f}{g['yaw']:7.3f}{g['trans_z']:8.3f}{g['tilt']:7.3f}"
              f"{g['vel_xy']:8.3f}{g['vel_z']:7.3f}   {name} {factor:.0%}" if missed else
              f"{i:>4} {row['id']:>16} {-neg:>3}/{total} {row['J']:+9.4f}"
              f"{g['trans_xy']:9.3f}{g['yaw']:7.3f}{g['trans_z']:8.3f}{g['tilt']:7.3f}"
              f"{g['vel_xy']:8.3f}{g['vel_z']:7.3f}   all met")
    if scored:
        print(f"\nbest: {scored[0][3]['id']} meets {-scored[0][0]}/{total}")
        for k, v in sorted(scored[0][5].items(), key=lambda kv: -kv[1]):
            print(f"   misses {k:22} by {v-1:+.1%}")


def cmd_rescore(args):
    rows = trials()
    for row in rows:
        row["new"] = kt.objective(row["ratios"])
    rows.sort(key=lambda r: r["new"])
    print(f"{len(rows)} configs re-ranked under the current objective\n")
    print(f"{'rank':>4}{'trial':>18}{'J':>9}" + "".join(f"{m:>10}" for m in METRICS)
          + f"{'slipXY':>9}{'slipYaw':>9}")
    for rank, row in enumerate(rows[:args.top], 1):
        metrics = per_metric(row["ratios"])
        cells = "".join(f"{geo(metrics[m]):10.3f}" for m in METRICS)
        print(f"{rank:>4}{row['id']:>18}{row['new']:+9.4f}{cells}"
              f"{slippage(row['ratios'], 'trans_xy'):9.3f}{slippage(row['ratios'], 'yaw'):9.3f}")


def cmd_slippage(args):
    reference_data = reference()
    riekf, installed = reference_data["riekf"], reference_data["installed"]
    # Defaults to the installed tuning: the question "how much does slippage hurt us" is about
    # the shipped configuration unless a specific trial is named.
    rows = trials()
    row = pick(rows, args.trial) if args.trial else None
    print("degradation = mean error on the 3 slippage runs / mean error on the 5 nominal RHPS1 runs")
    print("both estimators face the same trajectories, so the quotient of the two isolates "
          "which one\nsuffers more; below 1.000 in the last column means the Kinetics Observer "
          "degrades less.\n")
    print(f"{'metric':10}{'KO nominal':>12}{'KO slip':>10}{'KO degr':>9}"
          f"{'RI nominal':>12}{'RI slip':>10}{'RI degr':>9}{'KO/RI':>9}")
    for metric in METRICS:
        fmt = ".4f" if metric.startswith(("trans", "vel")) else ".3f"
        def mean(projects, table):
            return sum(table[p][metric] for p in projects) / len(projects)
        if row is not None:
            grouped = by_project(row["ratios"])
            table = {p: {metric: riekf[p][metric] * grouped[p][metric]} for p in kt.PROJECTS}
        else:
            table = installed
        kn, ks = mean(FAMILY["RHPS1"], table), mean(FAMILY["Slippage"], table)
        rn, rs = mean(FAMILY["RHPS1"], riekf), mean(FAMILY["Slippage"], riekf)
        print(f"{metric:10}{kn:12{fmt}}{ks:10{fmt}}{ks/kn:9.2f}"
              f"{rn:12{fmt}}{rs:10{fmt}}{rs/rn:9.2f}{(ks/kn)/(rs/rn):9.3f}")
    print(f"\nsource: {'installed tuning' if row is None else row['id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    parser.add_argument("--plain", action="store_true", help="no bold, for piping")
    sub.add_parser("status")
    for name in ("best", "slippage"):
        p = sub.add_parser(name)
        p.add_argument("--trial", help="RUN#N, or a bare trial number in the newest run")
    sub.add_parser("goal")
    p = sub.add_parser("select")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--run", help="restrict to runs whose directory name contains this")
    p = sub.add_parser("rescore")
    p.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    {"status": cmd_status, "best": cmd_best, "goal": cmd_goal, "select": cmd_select,
     "rescore": cmd_rescore, "slippage": cmd_slippage}[args.command](args)


if __name__ == "__main__":
    main()
