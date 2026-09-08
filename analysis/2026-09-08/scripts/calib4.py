"""Minimal per-foot TILT correction (2 DOF, no yaw null-space): the smallest rotation taking the
mean measured force direction onto world-up, from single-support samples where all the load is
on that foot. Estimated per dataset (consistency check) and pooled per robot (the calibration)."""
import sys, json, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
from scipy.spatial.transform import Rotation as Rot
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"; MASS=58.32; G=9.81
def quat2R(q):
    q=q/np.linalg.norm(q,axis=1,keepdims=True); w,x,y,z=q.T
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]).transpose(2,0,1)
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
def min_rot(u,v):
    u=u/np.linalg.norm(u); v=v/np.linalg.norm(v); ax=np.cross(u,v); s=np.linalg.norm(ax)
    if s<1e-12: return np.zeros(3)
    return ax/s*np.arctan2(s,float(np.dot(u,v)))
FEET=("LeftFootCenter","RightFootCenter")
ROBOTS={"rhps1":["KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_2","KO_TRO2024_RHPS1_3","KO_TRO2024_RHPS1_4",
                 "KO_TRO2024_RHPS1_5","KO_TRO_2024_RHPS1_SLIPPAGE_1","KO_TRO_2024_RHPS1_SLIPPAGE_2",
                 "KO_TRO_2024_RHPS1_SLIPPAGE_3"],
        "hrp5_p":["HRP5_MultiContact_1","HRP5_MultiContact_2","HRP5_MultiContact_3",
                  "HRP5_MultiContact_4","HRP5P_LongWalk"]}
out={}
for robot,ds in ROBOTS.items():
    print(f"\n===== {robot}")
    pool={c:[np.zeros(3),np.zeros(3),0] for c in FEET}
    for name in ds:
        try: log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
        except Exception as e: print(f"  {name:30s} -- {type(e).__name__}"); continue
        Rwc=quat2R(vec(log,B+"estimatedState_ori","wxyz")); row=[]
        for k,c in enumerate(FEET):
            Rw=Rwc@quat2R(vec(log,P+f"debug_contactKine_{c}_inputCentroidContactKine_orientation","wxyz"))
            f=vec(log,B+f"measurements_contacts_force_{c}_measured")
            S=lambda cc: np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{cc}"]])
            ss=np.flatnonzero(S(c)&(~S(FEET[1-k]))&(f[:,2]>0.7*MASS*G))
            if len(ss)<200: row.append(f"{c[:5]}: too few"); continue
            u=(f[ss]/np.linalg.norm(f[ss],axis=1,keepdims=True)).sum(0)
            d=np.einsum('nji,j->ni',Rw[ss],np.array([0,0,1.0])).sum(0)
            pool[c][0]+=u; pool[c][1]+=d; pool[c][2]+=len(ss)
            row.append(f"{c[:5]}: {np.degrees(np.linalg.norm(min_rot(u,d))):5.2f}deg (n={len(ss)})")
        print(f"  {name:30s} " + "   ".join(row)); del log
    out[robot]={}
    for c in FEET:
        if pool[c][2]==0: continue
        rv=min_rot(pool[c][0],pool[c][1]); out[robot][c]=rv.tolist()
        print(f"  POOLED {c:18s} tilt {np.degrees(np.linalg.norm(rv)):5.2f} deg"
              f"   rotvec_deg [{np.degrees(rv)[0]:+6.3f} {np.degrees(rv)[1]:+6.3f} {np.degrees(rv)[2]:+6.3f}]  n={pool[c][2]}")
json.dump(out,open(sys.argv[1],"w"),indent=2); print(f"\nwritten -> {sys.argv[1]}")
