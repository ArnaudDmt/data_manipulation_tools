import sys, numpy as np, glob, json; sys.path.insert(0,'.')
from pathlib import Path
import kinetics_tune as kt
LW=sorted(glob.glob("../results/W_LW-*"))[0]
RUN=sorted(glob.glob("../results/W_tuned_a1_posx30-*"))[0]
for lab,name,run in (("LongWalk (|r|=0.155)","HRP5P_LongWalk",LW),
                     ("RHPS1 walk (|r|=0.057)","KO_TRO2024_RHPS1_1",RUN)):
    g,mocap,_,settings=kt.reference_velocities([name])[name]
    pose=np.loadtxt(Path(run)/name/"kinetics.txt", comments="#", ndmin=2)
    velo=np.loadtxt(Path(run)/name/"kinetics_velocity.txt", comments="#", ndmin=2)
    ko=kt.estimated_local_velocity(pose, velo, g, settings)
    # the transported term omega x r, from the estimator's own angular rate
    from scipy.spatial.transform import Rotation, Slerp
    times=velo[:,0]
    ang=np.column_stack([np.interp(g, times, velo[:,4+a]) for a in range(3)])
    rot=Slerp(pose[:,0], Rotation.from_quat(pose[:,4:8]))(np.clip(g,pose[0,0],pose[-1,0]))
    ang_local=rot.apply(ang, inverse=True)
    transported=np.cross(ang_local, settings["pos_fb_imu"])
    n=min(len(ko),len(mocap))
    err=np.linalg.norm(ko[:n,0:2]-mocap[:n,0:2],axis=1)
    tmag=np.linalg.norm(transported[:n,0:2],axis=1)
    print(f"{lab:24s} |omega x r| median {np.median(tmag)*1000:6.1f} mm/s   p90 {np.percentile(tmag,90)*1000:7.1f}"
          f"   corr(err,|omega x r|) = {np.corrcoef(err,tmag)[0,1]:+.3f}")
