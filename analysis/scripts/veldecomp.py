import sys, numpy as np, glob, collections; sys.path.insert(0,'.')
from pathlib import Path
import kinetics_tune as kt
RUN=sorted(glob.glob("../results/W_tuned_a1_posx30-*"))[0]
LW=sorted(glob.glob("../results/W_LW-*"))[0]
CATS=[("MultiContact",[f"HRP5_MultiContact_{i}" for i in (1,2,3,4)],RUN),
      ("LongWalk",["HRP5P_LongWalk"],LW),
      ("RHPS1 walk",[f"KO_TRO2024_RHPS1_{i}" for i in range(1,6)],RUN),
      ("RHPS1 slippage",[f"KO_TRO_2024_RHPS1_SLIPPAGE_{i}" for i in (1,2,3)],RUN)]
print("Velocity error decomposition (xy), mm/s.  KO-only = the part of the KO error that is")
print("orthogonal to the RI-EKF's, i.e. what the KO adds beyond the error they share.\n")
print(f"{'category':16s}{'KO total':>10}{'RI total':>10}{'shared':>9}{'KO-only':>9}{'KO/RI':>8}{'corr':>7}")
for lab,names,run in CATS:
    EK=[];ER=[]
    for name in names:
        try:
            g,mocap,_,settings=kt.reference_velocities([name])[name]
            pose=np.loadtxt(Path(run)/name/"kinetics.txt", comments="#", ndmin=2)
            velo=np.loadtxt(Path(run)/name/"kinetics_velocity.txt", comments="#", ndmin=2)
            ko=kt.estimated_local_velocity(pose, velo, g, settings)
            _,directory=kt.project_paths(name)
            ri=np.loadtxt(directory/"reference/riekf_velocity.txt", comments="#", ndmin=2)
        except Exception: continue
        n=min(len(ko),len(mocap),len(ri))
        EK.append((ko[:n,0:2]-mocap[:n,0:2]).ravel()); ER.append((ri[:n,1:3]-mocap[:n,0:2]).ravel())
    if not EK: continue
    ek=np.concatenate(EK); er=np.concatenate(ER)
    c=float(np.corrcoef(ek,er)[0,1])
    kt_=np.sqrt(np.mean(ek**2))*1000; rt=np.sqrt(np.mean(er**2))*1000
    shared=abs(c)*kt_; only=kt_*np.sqrt(max(1-c*c,0.0))
    print(f"{lab:16s}{kt_:10.1f}{rt:10.1f}{shared:9.1f}{only:9.1f}{kt_/max(rt,1e-9):8.2f}{c:7.2f}")
