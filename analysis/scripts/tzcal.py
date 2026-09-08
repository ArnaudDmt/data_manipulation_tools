"""Is the measured foot yaw torque systematically offset? During single support the whole
ground reaction passes through one foot; with low angular rate the yaw moment about that
foot must be near zero. A consistent non-zero median, same sign per foot across datasets,
is a calibration signature -- exactly as the force tilt was."""
import sys, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"; MASS=58.32; G=9.81
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
FEET=("LeftFootCenter","RightFootCenter")
SETS=[("rhps1",["KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_2","KO_TRO2024_RHPS1_3",
                "KO_TRO2024_RHPS1_4","KO_TRO2024_RHPS1_5","KO_TRO_2024_RHPS1_SLIPPAGE_1"]),
      ("hrp5_p",["HRP5_MultiContact_1","HRP5_MultiContact_2","HRP5_MultiContact_3","HRP5_MultiContact_4"])]
for robot,projs in SETS:
    print(f"\n=== {robot}     measured tau_z during single support [N.m]")
    print(f"{'dataset':30s}{'foot':7s}{'n':>7s}{'tau_z p50':>11s}{'p25':>9s}{'p75':>9s}"
          f"{'|tau_z|/fz':>12s}{'tau_x p50':>11s}{'tau_y p50':>11s}")
    for p in projs:
        try: log=read_log(f"Projects/{p}/output_data/kinetics_eval/logReplay_full.bin")
        except Exception as e: print(f"{p:30s} -- {type(e).__name__}"); continue
        wz=np.linalg.norm(vec(log,B+"estimatedState_angVel"),axis=1)
        for k,c in enumerate(FEET):
            S=lambda cc: np.array([str(s).strip().lower().startswith("set")
                                   for s in log[P+f"debug_contactState_isSet_{cc}"]])
            f=vec(log,B+f"measurements_contacts_force_{c}_measured")
            t=vec(log,B+f"measurements_contacts_torque_{c}_measured")
            m=S(c)&(~S(FEET[1-k]))&np.isfinite(f).all(1)&np.isfinite(t).all(1)&(f[:,2]>0.7*MASS*G)&(wz<0.3)
            if m.sum()<200: print(f"{p:30s}{c[:5]:7s}{m.sum():7d}   too few"); continue
            tz=t[m,2]
            print(f"{p:30s}{c[:5]:7s}{m.sum():7d}{np.median(tz):11.3f}{np.percentile(tz,25):9.3f}"
                  f"{np.percentile(tz,75):9.3f}{np.median(np.abs(tz)/f[m,2]):12.4f}"
                  f"{np.median(t[m,0]):11.3f}{np.median(t[m,1]):11.3f}")
        del log
