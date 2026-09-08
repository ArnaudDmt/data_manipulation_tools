import sys, numpy as np, glob; sys.path.insert(0,'scripts')
from pathlib import Path
from mc_log_ui import read_log
import kinetics_tune as kt
B="Observers_MainObserverPipeline_MCKineticsObserver_MEKF_"
RUN=sorted(glob.glob("results/W_tuned_a1_posx30-*"))[0]
for name in ("KO_TRO2024_RHPS1_1","KO_TRO_2024_RHPS1_SLIPPAGE_1"):
    g,mocap,_,settings=kt.reference_velocities([name])[name]
    pose=np.loadtxt(Path(RUN)/name/"kinetics.txt", comments="#", ndmin=2)
    velo=np.loadtxt(Path(RUN)/name/"kinetics_velocity.txt", comments="#", ndmin=2)
    ko=kt.estimated_local_velocity(pose, velo, g, settings)
    n=min(len(ko),len(mocap))
    err=np.linalg.norm(ko[:n,0:2]-mocap[:n,0:2],axis=1)
    log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
    t=np.asarray(log["t"],float)
    uf=np.array([np.asarray(log[f"{B}estimatedState_extForceCentr_{a}"],float) for a in "xyz"])
    ufn=np.linalg.norm(uf[:2],axis=0)
    # put the unmodeled force on the evaluation grid
    ufg=np.interp(g[:n], t-t[0], ufn)
    m=np.isfinite(err)&np.isfinite(ufg)
    c=np.corrcoef(err[m],ufg[m])[0,1]
    print(f"{name:30s} vel err {np.sqrt(np.mean(err[m]**2))*1000:6.1f} mm/s   "
          f"|unmodeled force xy| median {np.median(ufg[m]):6.2f} N  p90 {np.percentile(ufg[m],90):7.2f} N"
          f"   corr = {c:+.3f}")
    del log
