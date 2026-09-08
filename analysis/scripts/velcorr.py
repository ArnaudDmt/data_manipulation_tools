import sys, numpy as np, glob; sys.path.insert(0,'.')
from pathlib import Path
import kinetics_tune as kt
RUN=sorted(glob.glob("../results/W_tuned_a1_posx30-*"))[0]
LW=sorted(glob.glob("../results/W_LW-*"))[0]
sets=[("MultiContact","HRP5_MultiContact_1",RUN),("RHPS1 walk","KO_TRO2024_RHPS1_1",RUN),
      ("RHPS1 slip","KO_TRO_2024_RHPS1_SLIPPAGE_1",RUN),("LongWalk","HRP5P_LongWalk",LW)]
print("Do the KO and the RI-EKF make the same velocity error?\n")
print(f"{'dataset':16s}{'KO RMS':>9}{'RI RMS':>9}{'corr(err_KO,err_RI)':>21}{'KO scale':>10}{'RI scale':>10}")
for lab,name,run in sets:
    _,directory = kt.project_paths(name)
    import json
    settings=json.loads((directory/"time_offset.json").read_text())
    mocap=np.loadtxt(directory/"reference/mocap_velocity.txt", comments="#", ndmin=2)
    ri=np.loadtxt(directory/"reference/riekf_velocity.txt", comments="#", ndmin=2)
    grid=np.loadtxt(directory/"reference/velocity_grid.txt", comments="#", ndmin=1) if (directory/"reference/velocity_grid.txt").exists() else None
    pose=np.loadtxt(Path(run)/name/"kinetics.txt", comments="#", ndmin=2)
    velo=np.loadtxt(Path(run)/name/"kinetics_velocity.txt", comments="#", ndmin=2)
    g,_m,_r,_s = kt.reference_velocities([name])[name]
    ko=kt.estimated_local_velocity(pose, velo, g, _s)
    n=min(len(ko),len(_m),len(ri))
    # columns are [timestamp vx vy vz]; the cached _m is already the velocity block
    riv=ri[:n,1:3]
    ek=(ko[:n,0:2]-_m[:n,0:2]).ravel(); er=(riv-_m[:n,0:2]).ravel()
    if er is None or len(er)!=len(ek): print(f"{lab:16s} (riekf velocity shape {ri.shape})"); continue
    c=np.corrcoef(ek,er)[0,1]
    m=_m[:n,0:2].ravel()
    ks=float(np.linalg.lstsq(m[:,None],ko[:n,0:2].ravel(),rcond=None)[0][0])
    rs=float(np.linalg.lstsq(m[:,None],riv.ravel(),rcond=None)[0][0])
    print(f"{lab:16s}{np.sqrt(np.mean(ek**2))*1000:9.1f}{np.sqrt(np.mean(er**2))*1000:9.1f}{c:21.3f}{ks:10.4f}{rs:10.4f}")
