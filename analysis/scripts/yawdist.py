"""Yaw RPE as a function of sub-trajectory length. Needs only trajectories, no logs."""
import sys, glob, os, numpy as np
from scipy.spatial.transform import Rotation as Rot
def yaw_of(R): 
    m=R.as_matrix(); return np.arctan2(m[1,0],m[0,0])
def rpe_yaw(G,E,step,D,stride=10):
    ends=np.searchsorted(step,step+D); out=[]
    for i in range(0,len(step)-1,stride):
        j=ends[i]
        if j>=len(step): break
        Rg=Rot.from_quat(G[i,3:7]).inv()*Rot.from_quat(G[j,3:7])
        Re=Rot.from_quat(E[i,3:7]).inv()*Rot.from_quat(E[j,3:7])
        out.append(abs(np.degrees(yaw_of(Rg.inv()*Re))))
    return np.array(out)
def find(proj):
    for pref in ("PW_cal","PW_base","W_LW","V_tuned_a0.0","W_tuned_a1_posx30"):
        g=glob.glob(f"results/{pref}-*/{proj}/kinetics.txt")
        if g: return g[0]
    return None
SETS=[("RHPS1 walk",["KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_2","KO_TRO2024_RHPS1_3",
                     "KO_TRO2024_RHPS1_4","KO_TRO2024_RHPS1_5"]),
      ("LongWalk",["HRP5P_LongWalk"])]
DIST=[0.5,1.0,2.0,4.0,8.0,10.0]
for lab,projs in SETS:
    acc={d:[[],[]] for d in DIST}; plen=[]
    for p in projs:
        kf=find(p)
        if not kf: print(f"  {p}: no trajectory found"); continue
        base=f"Projects/{p}/output_data/kinetics_eval"
        gt=np.loadtxt(f"{base}/reference/mocap.txt",comments="#",ndmin=2)
        ri=np.loadtxt(f"{base}/reference/riekf.txt",comments="#",ndmin=2)
        ko=np.loadtxt(kf,comments="#",ndmin=2)
        t=gt[:,0]; it=lambda a: np.column_stack([np.interp(t,a[:,0],a[:,c]) for c in range(1,8)])
        G=gt[:,1:8]; R_=it(ri); K_=it(ko)
        step=np.r_[0,np.cumsum(np.linalg.norm(np.diff(G[:,:3],axis=0),axis=1))]
        plen.append(step[-1])
        for d in DIST:
            if step[-1]<d*1.5: continue
            acc[d][0].append(rpe_yaw(G,K_,step,d)); acc[d][1].append(rpe_yaw(G,R_,step,d))
    print(f"\n=== {lab}   path length {['%.1f'%x for x in plen]} m")
    print(f"{'sub-traj':>10s}{'KO [deg]':>12s}{'RI-EKF [deg]':>14s}{'ratio':>9s}{'n':>8s}")
    for d in DIST:
        k,r=acc[d]
        if not k: continue
        K=np.concatenate(k); R=np.concatenate(r)
        rms=lambda a: float(np.sqrt(np.mean(a**2)))
        print(f"{d:9.1f}m{rms(K):12.3f}{rms(R):14.3f}{rms(K)/rms(R):9.3f}{len(K):8d}")
