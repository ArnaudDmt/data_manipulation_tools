import sys, csv, collections, glob, numpy as np
sys.path.insert(0,'.')
from pathlib import Path
import kinetics_tune as kt

# scenario categories: by robot AND by scenario. LongWalk is HRP5P, like MultiContact,
# but a different scenario; the RHPS1 walks are a different platform entirely.
CATS=[("MultiContact (HRP5P)", lambda p: p.startswith("HRP5_MultiContact"), "0.3 m"),
      ("LongWalk (HRP5P)",     lambda p: p.startswith("HRP5P_LongWalk"),    "10 m"),
      ("RHPS1 walk",           lambda p: p.startswith("KO_TRO2024_RHPS1"),  "1 m"),
      ("RHPS1 slippage",       lambda p: p.startswith("KO_TRO_2024_RHPS1_SLIPPAGE"), "1 m")]
def cat(p):
    for n,f,_ in CATS:
        if f(p): return n
    return None

RUNS=[sorted(glob.glob("../results/W_tuned_a1_posx30-*"))[0],
      sorted(glob.glob("../results/W_LW-*"))[0]]
vals=collections.defaultdict(lambda: collections.defaultdict(dict))
projects=[]
for RUN in RUNS:
    for r in csv.DictReader(open(f"{RUN}/summary.csv")):
        if r["statistic"]!="mean": continue
        vals[r["project"]][r["estimator"]][r["metric"]]=float(r["value"])
    for d in Path(RUN).iterdir():
        if d.is_dir() and cat(d.name): projects.append((d.name, RUN))

vel={}
cache=kt.reference_velocities([n for n,_ in projects])
for name,RUN in projects:
    grid, mocap, riekf, settings = cache[name]
    pose=np.loadtxt(Path(RUN)/name/"kinetics.txt", comments="#", ndmin=2)
    velo=np.loadtxt(Path(RUN)/name/"kinetics_velocity.txt", comments="#", ndmin=2)
    local=kt.estimated_local_velocity(pose, velo, grid, settings)
    vel[name]={"Kinetics":kt.velocity_errors(local, mocap), "RI-EKF":tuple(riekf)}

P=[n for n,_ in projects]
print("\n| metric | " + " | ".join(f"{n} KO | {n} RI-EKF" for n,_,_ in CATS) + " |")
print("|:--|" + "".join(":--|:--|" for _ in CATS))
def emit(label, getter, scale, fmt="{:.3f}"):
    cells=[]
    for n,_,_ in CATS:
        ps=[p for p in P if cat(p)==n]
        if not ps: cells += ["n/a","n/a"]; continue
        a=np.mean([getter(p,"Kinetics") for p in ps])*scale
        b=np.mean([getter(p,"RI-EKF") for p in ps])*scale
        sa,sb=fmt.format(a),fmt.format(b)
        if a<b: sa=f"**{sa}**"
        else:   sb=f"**{sb}**"
        cells += [sa,sb]
    print(f"| {label} | " + " | ".join(cells) + " |")
for key,label,scale in (("trans_xy","pos xy RPE [mm]",1000.0),("trans_z","pos z RPE [mm]",1000.0),
                        ("yaw","yaw RPE [deg]",1.0),("tilt","tilt err [deg]",1.0)):
    emit(label, lambda p,e,k=key: vals[p][e][k], scale)
emit("vel xy RMSE [mm/s]", lambda p,e: vel[p][e][0], 1000.0)
emit("vel z RMSE [mm/s]",  lambda p,e: vel[p][e][1], 1000.0)
print("\nsub-trajectory length: " + ", ".join(f"{n} {d}" for n,_,d in CATS))
print("datasets: " + ", ".join(f"{n}={sum(1 for p in P if cat(p)==n)}" for n,_,_ in CATS))
