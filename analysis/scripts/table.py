import sys, csv, math, collections, glob, numpy as np
sys.path.insert(0,'.')
from pathlib import Path
import kinetics_tune as kt

FAM=[("MultiContact",  lambda p: p.startswith("HRP5_MultiContact")),
     ("RHPS1 nominal", lambda p: p.startswith("KO_TRO2024_RHPS1")),
     ("RHPS1 slippage",lambda p: p.startswith("KO_TRO_2024_RHPS1_SLIPPAGE"))]
def fam(p):
    for n,f in FAM:
        if f(p): return n
    return None

RUN=sorted(glob.glob("../results/W_tuned_a1_posx30-*"))[0]
P=[d.name for d in Path(RUN).iterdir() if d.is_dir()]

# --- pose metrics straight from the run's summary (absolute errors) ---
vals=collections.defaultdict(lambda: collections.defaultdict(dict))
for r in csv.DictReader(open(f"{RUN}/summary.csv")):
    if r["statistic"]!="mean": continue
    vals[r["project"]][r["estimator"]][r["metric"]]=float(r["value"])

# --- velocity: absolute RMS error for both estimators ---
cache=kt.reference_velocities(P)
vel={}
for name in P:
    grid, mocap, riekf, settings = cache[name]
    pose=np.loadtxt(Path(RUN)/name/"kinetics.txt", comments="#", ndmin=2)
    velo=np.loadtxt(Path(RUN)/name/"kinetics_velocity.txt", comments="#", ndmin=2)
    local=kt.estimated_local_velocity(pose, velo, grid, settings)
    ko=kt.velocity_errors(local, mocap)
    vel[name]={"Kinetics":ko, "RI-EKF":tuple(riekf)}

rows=[("trans_xy","pos xy RPE [mm]",1000.0),("trans_z","pos z RPE [mm]",1000.0),
      ("yaw","yaw RPE [deg]",1.0),("tilt","tilt err [deg]",1.0)]
print(f"\nKinetics Observer vs RI-EKF -- absolute errors, mean over each scenario category")
print(f"configuration: load-weighted projector (alpha=1) + contact position process x30")
print(f"{len(P)} datasets (LongWalk excluded: bag-conversion rejections unresolved)\n")
hdr=f"{'metric':22s}"
for n,_ in FAM: hdr+=f"{n:>26s}"
print(hdr)
print(f"{'':22s}" + "".join(f"{'KO':>12}{'RI-EKF':>14}" for _ in FAM))
def emit(label, getter, scale, fmt="{:.3f}"):
    line=f"{label:22s}"
    for n,_ in FAM:
        ps=[p for p in P if fam(p)==n]
        a=np.mean([getter(p,"Kinetics") for p in ps])*scale
        b=np.mean([getter(p,"RI-EKF") for p in ps])*scale
        sa,sb=fmt.format(a),fmt.format(b)
        if a<b: sa=f"**{sa}**"
        else:   sb=f"**{sb}**"
        line+=f"{sa:>12}{sb:>14}"
    print(line)
for key,label,scale in rows:
    emit(label, lambda p,e,k=key: vals[p][e][k], scale)
emit("vel xy RMSE [mm/s]", lambda p,e: vel[p][e][0], 1000.0)
emit("vel z  RMSE [mm/s]", lambda p,e: vel[p][e][1], 1000.0)
