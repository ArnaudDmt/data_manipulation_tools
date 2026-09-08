#!/usr/bin/env python3
"""One-at-a-time impact of each contact-model parameter, from the installed tuning.

A search tells you where the optimum is; an ablation tells you which knobs do anything at all,
and in which direction. That is the cheaper question, and it is the one that should decide what
goes into a joint search -- a dimension with no measurable effect is only diluting the sampler.
"""
import sys, json, math, collections, itertools
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "scripts")
import kinetics_tune as kt

MC = [f"HRP5_MultiContact_{i}" for i in range(1, 5)]
RH = [f"KO_TRO2024_RHPS1_{i}" for i in range(1, 6)]
SL = [f"KO_TRO_2024_RHPS1_SLIPPAGE_{i}" for i in range(1, 4)]
DATASETS = {"hrp5_p": MC, "rhps1": RH + SL}
STEPS = (-1.0, -0.5, +0.5, +1.0)          # decades from the installed value
WORKERS = 8
geo = lambda v: math.exp(sum(math.log(max(x, 1e-9)) for x in v) / len(v))

base, widths, flex = kt.base_covariances()
start = kt.base_point(base, flex)
robot = sys.argv[1]
projects = DATASETS[robot]
dims = [(n, lo, hi) for n, r, f, ax, lo, hi in kt.FLEXIBILITY_SPACE if r == robot]

jobs = []
for name, lo, hi in dims:
    for step in STEPS:
        v = start[name] + step
        if lo <= v <= hi:
            jobs.append((name, step, v))
print(f"{robot}: {len(dims)} dimensions x {len(STEPS)} steps -> {len(jobs)} runs on {len(projects)} datasets",
      flush=True)

def run(job):
    i, (name, step, value) = job
    point = {**start, name: value}
    overlay = kt.overlay_from(base, widths, point, flex)
    d, err = kt.run_evaluation(overlay, projects, f"abl-{robot}-{i:03d}", 60 + i % WORKERS,
                               "/home/arnaud/devel/src/catkin_ws", 1800)
    if d is None:
        return name, step, value, None, err[:80]
    try:
        r = kt.read_ratios(d / "summary.csv", projects)
        r.update(kt.velocity_ratios(d, projects, kt.reference_velocities(projects)))
    except Exception as e:
        return name, step, value, None, str(e)[:80]
    finally:
        import shutil; shutil.rmtree(d, ignore_errors=True)
    per = collections.defaultdict(list); byp = {}
    for k, v in r.items():
        p, m = k.split("|"); per[m].append(v); byp.setdefault(p, {})[m] = v
    out = {m: geo(per[m]) for m in per}
    if robot == "rhps1":
        out["slip_xy"] = geo([byp[p]["trans_xy"] for p in SL]) / geo([byp[p]["trans_xy"] for p in RH])
        out["slip_yaw"] = geo([byp[p]["yaw"] for p in SL]) / geo([byp[p]["yaw"] for p in RH])
    return name, step, value, out, ""

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    results = list(pool.map(run, enumerate(jobs)))
json.dump([[n, s, v, o, e] for n, s, v, o, e in results],
          open(f"/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/ablation-{robot}.json", "w"))
print("done", flush=True)
