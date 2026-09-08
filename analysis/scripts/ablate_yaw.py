#!/usr/bin/env python3
"""Does splitting roll/pitch from yaw give the search any leverage on yaw at all?

Nothing improved yaw in ~1400 evaluations, and the contact model demonstrably cannot (largest
correlation +0.23 of 16 dimensions). The hypothesis is that the state covariances were isotropic,
forcing one compromise across gravity-observable roll/pitch and unobservable yaw. This moves each
new split dimension on its own, so the answer does not depend on a sampler exploring 41 dimensions
for seven hours.

Scored on the eight RHPS1 datasets: that is the family where yaw fails (1.01-1.11 in every
config), and they are the cheap ones -- 239s a trial against 1003s for all thirteen.
"""
import sys, json, math, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "scripts")
import kinetics_tune as kt

PROJECTS = [f"KO_TRO2024_RHPS1_{i}" for i in range(1, 6)] + \
           [f"KO_TRO_2024_RHPS1_SLIPPAGE_{i}" for i in range(1, 4)]
DIMS = ("contact_new_position_xy", "contact_new_position_z",
        "contact_new_orientation_rp", "contact_new_orientation_yaw",
        "contact_process_position_xy", "contact_process_position_z_ratio",
        "contact_process_orientation_rp", "contact_process_orientation_yaw")
STEPS = (-2.0, -1.0, +1.0, +2.0)
WORKERS = 12
geo = lambda v: math.exp(sum(math.log(max(x, 1e-9)) for x in v) / len(v))

base, widths, flex = kt.base_covariances()
start = kt.base_point(base, flex)
space = {n: (lo, hi) for n, *_, lo, hi in kt.active_space(False)}

jobs = [(n, s, start[n] + s) for n in DIMS for s in STEPS
        if space[n][0] <= start[n] + s <= space[n][1]]
jobs.insert(0, ("BASELINE", 0.0, start["contact_new_position_xy"]))
print(f"{len(jobs)} runs on {len(PROJECTS)} datasets, {WORKERS} workers", flush=True)

def run(job):
    i, (name, step, value) = job
    point = dict(start) if name == "BASELINE" else {**start, name: value}
    overlay = kt.overlay_from(base, widths, point, flex)
    d, err = kt.run_evaluation(overlay, PROJECTS, f"ac-{i:03d}", 60 + i % WORKERS,
                               "/home/arnaud/devel/src/catkin_ws", 1800)
    if d is None:
        return name, step, None, err[:70]
    try:
        r = kt.read_ratios(d / "summary.csv", PROJECTS)
        r.update(kt.velocity_ratios(d, PROJECTS, kt.reference_velocities(PROJECTS)))
    except Exception as e:
        return name, step, None, str(e)[:70]
    finally:
        import shutil; shutil.rmtree(d, ignore_errors=True)
    per = collections.defaultdict(list)
    for k, v in r.items():
        per[k.split("|")[1]].append(v)
    return name, step, {m: geo(per[m]) for m in per}, ""

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    out = list(pool.map(run, enumerate(jobs)))
json.dump([[n, s, o, e] for n, s, o, e in out],
          open("/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/"
               "287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/ablation-contact.json", "w"))
print("done", flush=True)
