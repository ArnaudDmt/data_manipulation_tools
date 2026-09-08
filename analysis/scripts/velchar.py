import sys, numpy as np, glob; sys.path.insert(0,'.')
from pathlib import Path
import kinetics_tune as kt
RUN=sorted(glob.glob("../results/W_tuned_a1_posx30-*"))[0]
LW=sorted(glob.glob("../results/W_LW-*"))[0]
sets=[("MultiContact","HRP5_MultiContact_1",RUN),("RHPS1 walk","KO_TRO2024_RHPS1_1",RUN),
      ("RHPS1 slip","KO_TRO_2024_RHPS1_SLIPPAGE_1",RUN),("LongWalk","HRP5P_LongWalk",LW)]
cache=kt.reference_velocities([n for _,n,_ in sets])
print("KO local velocity error character (xy). bias = mean, noise = std of the residual\n")
print(f"{'dataset':16s}{'RMS':>9}{'|bias|':>9}{'noise':>9}{'scale':>8}{'resid after scale':>19}{'hi-freq share':>15}")
for lab,name,run in sets:
    grid, mocap, riekf, settings = cache[name]
    pose=np.loadtxt(Path(run)/name/"kinetics.txt", comments="#", ndmin=2)
    velo=np.loadtxt(Path(run)/name/"kinetics_velocity.txt", comments="#", ndmin=2)
    local=kt.estimated_local_velocity(pose, velo, grid, settings)
    r=local[:,0:2]-mocap[:,0:2]
    rms=np.sqrt(np.mean(np.linalg.norm(r,axis=1)**2))
    bias=np.linalg.norm(r.mean(0)); noise=np.sqrt(np.mean(np.linalg.norm(r-r.mean(0),axis=1)**2))
    m=mocap[:,0:2].ravel(); e=local[:,0:2].ravel()
    k=float(np.linalg.lstsq(m[:,None],e,rcond=None)[0][0])
    res=np.sqrt(np.mean((e-k*m)**2))
    # high-frequency share: energy of the step-to-step difference vs total
    hf=np.sqrt(np.mean(np.diff(r,axis=0)**2))/np.sqrt(2)/max(np.sqrt(np.mean(r**2)),1e-12)
    print(f"{lab:16s}{rms*1000:9.1f}{bias*1000:9.1f}{noise*1000:9.1f}{k:8.4f}{res*1000:19.1f}{hf:15.2f}")
print("\n(mm/s. scale = slope of KO velocity vs mocap velocity; hi-freq share ~1 means white noise,")
print(" <<1 means slowly varying error)")
