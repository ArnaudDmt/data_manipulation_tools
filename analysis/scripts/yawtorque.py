"""How much yaw error does the contact yaw-torque residual imply?
The visco-elastic model ties contact yaw to torque_z through angStiffness, so a residual of
dTz implies dTz/K_ang radians of yaw the filter cannot explain. Compared per robot."""
import sys, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"
K={"rhps1":727.0,"hrp5_p":1000.0}
SETS=[("rhps1",["KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_3","KO_TRO2024_RHPS1_5"]),
      ("hrp5_p",["HRP5_MultiContact_1","HRP5_MultiContact_2","HRP5P_LongWalk"])]
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
for robot,projs in SETS:
    ka=K[robot]
    print(f"\n=== {robot}   angStiffness {ka:.0f} N.m/rad")
    print(f"{'dataset':26s}{'contact':16s}{'|dTz| p50':>11s}{'p90':>9s}"
          f"{'-> yaw p50':>12s}{'p90 [deg]':>11s}{'|dTxy| p50':>12s}{'-> tilt p50':>13s}")
    for p in projs:
        try: log=read_log(f"Projects/{p}/output_data/kinetics_eval/logReplay_full.bin")
        except Exception as e: print(f"{p:26s} -- {type(e).__name__}"); continue
        for c in ("LeftFootCenter","RightFootCenter"):
            k1=f"{B}measurements_contacts_torque_{c}_measured"
            k2=f"{B}measurements_contacts_torque_{c}_predicted"
            if f"{k1}_x" not in log or f"{k2}_x" not in log: continue
            S=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
            f=vec(log,f"{B}measurements_contacts_force_{c}_measured")
            m=S&np.isfinite(f).all(1)&(f[:,2]>50)
            d=(vec(log,k1)-vec(log,k2))[m]
            if len(d)<200: continue
            dz=np.abs(d[:,2]); dxy=np.linalg.norm(d[:,:2],axis=1)
            print(f"{p:26s}{c[:14]:16s}{np.median(dz):11.3f}{np.percentile(dz,90):9.3f}"
                  f"{np.degrees(np.median(dz)/ka):12.4f}{np.degrees(np.percentile(dz,90)/ka):11.4f}"
                  f"{np.median(dxy):12.3f}{np.degrees(np.median(dxy)/ka):13.4f}")
        del log
