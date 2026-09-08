import sys, numpy as np, glob; sys.path.insert(0,'.')
from pathlib import Path
from scipy.signal import butter, filtfilt
import kinetics_tune as kt
RUN=sorted(glob.glob("../results/W_tuned_a1_posx30-*"))[0]
LW=sorted(glob.glob("../results/W_LW-*"))[0]
sets=[("MultiContact 200Hz","HRP5_MultiContact_1",RUN),("RHPS1 walk 200Hz","KO_TRO2024_RHPS1_1",RUN),
      ("LongWalk 250Hz","HRP5P_LongWalk",LW)]
print("If the deficit is bandwidth, band-limiting the estimate the same way should not change")
print("the 200 Hz sets much, but should move LongWalk's scale toward 1.\n")
print(f"{'dataset':22s}{'scale raw':>11}{'scale band-limited':>20}{'RMS raw':>10}{'RMS bl':>9}")
for lab,name,run in sets:
    g,mocap,_,settings=kt.reference_velocities([name])[name]
    pose=np.loadtxt(Path(run)/name/"kinetics.txt", comments="#", ndmin=2)
    velo=np.loadtxt(Path(run)/name/"kinetics_velocity.txt", comments="#", ndmin=2)
    ko=kt.estimated_local_velocity(pose, velo, g, settings)
    n=min(len(ko),len(mocap))
    m=mocap[:n,0:2]; k=ko[:n,0:2]
    dt=float(np.median(np.diff(g))); rate=1.0/dt
    # band-limit BOTH to the same physical 15 Hz using each set's own rate
    b,a=butter(2, 15.0/(0.5*rate), btype="low")
    mb=filtfilt(b,a,m,axis=0); kb=filtfilt(b,a,k,axis=0)
    s_raw=float(np.linalg.lstsq(m.ravel()[:,None],k.ravel(),rcond=None)[0][0])
    s_bl =float(np.linalg.lstsq(mb.ravel()[:,None],kb.ravel(),rcond=None)[0][0])
    r_raw=np.sqrt(np.mean(np.linalg.norm(k-m,axis=1)**2))*1000
    r_bl =np.sqrt(np.mean(np.linalg.norm(kb-mb,axis=1)**2))*1000
    print(f"{lab:22s}{s_raw:11.4f}{s_bl:20.4f}{r_raw:10.1f}{r_bl:9.1f}")
