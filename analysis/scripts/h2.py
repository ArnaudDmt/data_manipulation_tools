"""H2: does the KO's yaw and position error accumulate at contact transitions?

Alignment-invariant: we look at per-step *increments* of the error, and ask whether the increments
occurring within a window of a contact creation/removal are larger than elsewhere.
"""
import sys, numpy as np, csv, os
sys.path.insert(0,'scripts')
from mc_log_ui import read_log
from scipy.spatial.transform import Rotation as R
S="Observers_MainObserverPipeline_MCKineticsObserver_debug_contactState_isSet_"

def yaw_of(q): return R.from_quat(q).as_euler('zyx')[:,0]

for proj in ("KO_TRO2024_RHPS1_3","HRP5_MultiContact_1"):
    csvp=f"Projects/{proj}/output_data/synchronizedObserversMocapData.csv"
    binp=f"Projects/{proj}/output_data/kinetics_eval/logReplay_full.bin"
    if not (os.path.exists(csvp) and os.path.exists(binp)): continue
    with open(csvp) as f:
        rd=csv.reader(f,delimiter=';'); head=next(rd); idx={n:i for i,n in enumerate(head)}
        need=[f"MocapAligner_worldBodyKine_ori_{a}" for a in "wxyz"]+[f"KO_orientation_{a}" for a in "wxyz"]+\
             [f"MocapAligner_worldBodyKine_position_{a}" for a in "xyz"]+[f"KO_position_{a}" for a in "xyz"]
        rows=[]
        for r in rd:
            try: rows.append([float(r[idx[n]]) for n in need])
            except (ValueError,KeyError): rows.append([np.nan]*len(need))
    d=np.array(rows)
    log=read_log(binp)
    n=min(len(d), len(log["t"]))
    d=d[:n]
    flags=[]
    for foot in ("RightFootCenter","LeftFootCenter"):
        raw=np.asarray(log[f"{S}{foot}"])[:n]
        flags.append(np.array([str(v).strip().lower().startswith("set") for v in raw]))
    nb=flags[0].astype(int)+flags[1].astype(int)
    trans=np.zeros(n,bool)
    ch=np.flatnonzero(np.diff(nb)!=0)
    W=40  # +-0.2 s at 200 Hz
    for c in ch: trans[max(0,c-W):min(n,c+W+1)]=True
    qm=d[:,0:4][:,[1,2,3,0]]; qk=d[:,4:8][:,[1,2,3,0]]
    ok=np.isfinite(qm).all(1)&np.isfinite(qk).all(1)&(np.linalg.norm(qm,axis=1)>1e-6)&(np.linalg.norm(qk,axis=1)>1e-6)
    nrm=lambda q:q/np.linalg.norm(q,axis=1,keepdims=True)
    ey=np.full(n,np.nan); ep=np.full(n,np.nan)
    ey[ok]=np.unwrap(yaw_of(nrm(qk[ok])))-np.unwrap(yaw_of(nrm(qm[ok])))
    pm=d[:,8:11]; pk=d[:,11:14]
    okp=np.isfinite(pm).all(1)&np.isfinite(pk).all(1)
    ep[okp]=np.linalg.norm((pk-pm)[okp],axis=1)
    dy=np.abs(np.diff(ey)); dp=np.abs(np.diff(ep))
    # control: how much does the robot itself move in the same windows?
    mocap_step=np.linalg.norm(np.diff(pm,axis=0),axis=1)
    mocap_yawstep=np.abs(np.diff(np.unwrap(yaw_of(nrm(qm[ok])))))
    tw=trans[1:]
    for nm,inc,ctl in (("yaw",dy,None),("position",dp,mocap_step)):
        m=np.isfinite(inc)
        a=np.nanmedian(inc[m&tw]); b=np.nanmedian(inc[m&~tw])
        line=(f"{proj:22s} {nm:9s} error incr at transitions {a:.3e}  elsewhere {b:.3e}"
              f"  ratio {a/max(b,1e-15):6.2f}")
        if ctl is not None:
            mc=np.isfinite(ctl)
            ca=np.nanmedian(ctl[mc&tw]); cb=np.nanmedian(ctl[mc&~tw])
            line += f"   | motion ratio {ca/max(cb,1e-15):5.2f}  -> normalised {a/max(b,1e-15)/max(ca/max(cb,1e-15),1e-9):6.2f}"
        print(line)
    del log
print("DONE")
