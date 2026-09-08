import sys, numpy as np, glob, json; sys.path.insert(0,'.')
from pathlib import Path
import kinetics_tune as kt
RUN=sorted(glob.glob("../results/W_tuned_a1_posx30-*"))[0]
LW=sorted(glob.glob("../results/W_LW-*"))[0]
sets=[("MultiContact","HRP5_MultiContact_1",RUN),("RHPS1 walk","KO_TRO2024_RHPS1_1",RUN),
      ("RHPS1 slip","KO_TRO_2024_RHPS1_SLIPPAGE_1",RUN),("LongWalk","HRP5P_LongWalk",LW)]
cache=kt.reference_velocities([n for _,n,_ in sets])
print("Is the velocity error a timing lag? shift the KO velocity in time and re-measure.\n")
print(f"{'dataset':16s}{'dt grid':>9}{'RMS now':>10}{'best lag':>10}{'RMS at best':>13}{'improvement':>13}")
for lab,name,run in sets:
    grid, mocap, riekf, settings = cache[name]
    pose=np.loadtxt(Path(run)/name/"kinetics.txt", comments="#", ndmin=2)
    velo=np.loadtxt(Path(run)/name/"kinetics_velocity.txt", comments="#", ndmin=2)
    local=kt.estimated_local_velocity(pose, velo, grid, settings)
    step=float(np.median(np.diff(grid)))
    base=np.sqrt(np.mean(np.linalg.norm(local[:,0:2]-mocap[:,0:2],axis=1)**2))
    best=(base,0)
    for s in range(-40,41):
        if s>=0: a,b=local[s:,0:2], mocap[:len(local)-s,0:2]
        else:    a,b=local[:s,0:2], mocap[-s:,0:2]
        r=np.sqrt(np.mean(np.linalg.norm(a-b,axis=1)**2))
        if r<best[0]: best=(r,s)
    r,s=best
    print(f"{lab:16s}{step*1000:8.1f}m{base*1000:10.1f}{s*step*1000:9.1f}m{r*1000:13.1f}{100*(1-r/base):12.1f}%")
print("\n(RMS in mm/s, lag in ms; positive lag means the KO estimate is late)")
