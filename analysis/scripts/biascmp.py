"""Both estimators' gyro bias, side by side. Same init variance (1e-8), but process
variance 1e-18 (KO) vs 1e-10 (RI-EKF) -- 10^8 apart."""
import sys, csv, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
KB="Observers_MainObserverPipeline_MCKineticsObserver_MEKF_estimatedState_gyroBias_Accelerometer"
for n in ("KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_3","KO_TRO_2024_RHPS1_SLIPPAGE_1"):
    with open(f"Projects/{n}/output_data/synchronizedObserversMocapData.csv") as fh:
        rd=csv.reader(fh,delimiter=';'); h=next(rd); i={k:j for j,k in enumerate(h)}
        c=[f"Hartley_IMU_gyroBias_{a}" for a in "xyz"]
        hb=np.array([[float(r[i[k]]) if r[i[k]] not in ("","nan") else np.nan for k in c] for r in rd])
    log=read_log(f"Projects/{n}/output_data/kinetics_eval/logReplay_full.bin")
    kb=np.array([np.asarray(log[f"{KB}_{a}"],float) for a in "xyz"]).T
    m=min(len(kb),len(hb)); kb,hb=kb[:m],hb[:m]
    ok=np.isfinite(hb).all(1)
    print(f"\n=== {n}   ({ok.sum()}/{m} valid)          [all values in urad/s]")
    for lab,b in (("KO   (proc 1e-18)",kb),("RI-EKF (proc 1e-10)",hb)):
        v=b[ok]*1e6
        drift=np.linalg.norm(v[-1]-v[0]); rng=np.linalg.norm(v.max(0)-v.min(0))
        # how fast it wanders: rms step over 1 s
        k=max(1,int(1.0/0.002)); dd=np.linalg.norm(v[k:]-v[:-k],axis=1)
        print(f"  {lab:20s} start->end {drift:9.1f}   peak-peak {rng:9.1f}"
              f"   |b|max {np.max(np.linalg.norm(v,axis=1)):9.1f}   wander/s {np.median(dd):7.2f}")
    del log
