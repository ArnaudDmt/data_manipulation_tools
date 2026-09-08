import sys, numpy as np, csv; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
B="Observers_MainObserverPipeline_MCKineticsObserver_MEKF_"
for name in ("KO_TRO2024_RHPS1_1","KO_TRO_2024_RHPS1_SLIPPAGE_1"):
    log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
    n=len(log["t"])
    uf=np.array([np.asarray(log[f"{B}estimatedState_extForceCentr_{a}"],float) for a in "xyz"])
    ufxy=np.linalg.norm(uf[:2],axis=0); ufz=np.abs(uf[2])
    with open(f"Projects/{name}/output_data/synchronizedObserversMocapData.csv") as fh:
        rd=csv.reader(fh,delimiter=';'); h=next(rd); i={k:j for j,k in enumerate(h)}
        # linVel is not populated in this CSV; differentiate the position instead
        cols=[f"MocapAligner_worldBodyKine_position_{a}" for a in "xyz"]
        v=np.array([[float(r[i[c]]) if r[i[c]] not in ("","nan") else np.nan for c in cols] for r in rd])
    m=min(n,len(v))
    dt=float(np.median(np.diff(np.asarray(log["t"],float))))
    from scipy.signal import butter, filtfilt
    b,a_=butter(2, 5.0/(0.5/dt), btype="low")
    pos=v[:m]
    good=np.isfinite(pos).all(1)
    pos=np.where(good[:,None], pos, np.nan)
    pos=np.vstack([np.interp(np.arange(m), np.flatnonzero(good), pos[good,k]) for k in range(3)]).T
    vel=np.vstack((np.zeros((1,3)), np.diff(filtfilt(b,a_,pos,axis=0),axis=0)/dt))
    sp=np.linalg.norm(vel[:,:2],axis=1)
    ok=np.isfinite(sp)
    still = ok & (sp < 0.02)      # body moving slower than 2 cm/s
    moving= ok & (sp > 0.15)
    print(f"{name}")
    print(f"   samples: still {still.sum():6d}   moving {moving.sum():6d}   (of {m})")
    for lab,mask in (("STILL (<2 cm/s)",still),("MOVING (>15 cm/s)",moving)):
        if mask.sum()<200: print(f"   {lab:20s} too few samples"); continue
        print(f"   {lab:20s} |ext force xy| median {np.median(ufxy[:m][mask]):7.1f} N"
              f"   ext force z median {np.median(uf[2][:m][mask]):+8.1f} N")
    del log
