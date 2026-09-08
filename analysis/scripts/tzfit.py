"""Offset or friction? tau_z = a*fz + b over the full load range.
  b != 0 (load-independent)  -> additive sensor offset, subtractable without touching the model
  b ~ 0, a != 0              -> genuine friction moment, scaling with normal load"""
import sys, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
SETS=[("rhps1",["KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_2","KO_TRO2024_RHPS1_3",
                "KO_TRO2024_RHPS1_4","KO_TRO2024_RHPS1_5"]),
      ("hrp5_p",["HRP5_MultiContact_1","HRP5_MultiContact_2"])]
for robot,projs in SETS:
    print(f"\n=== {robot}")
    print(f"{'dataset':30s}{'foot':7s}{'n':>7s}{'fz range [N]':>16s}"
          f"{'slope a':>10s}{'intercept b':>13s}{'b 95% CI':>20s}{'R^2':>7s}")
    for p in projs:
        try: log=read_log(f"Projects/{p}/output_data/kinetics_eval/logReplay_full.bin")
        except Exception as e: print(f"{p:30s} -- {type(e).__name__}"); continue
        for c in ("LeftFootCenter","RightFootCenter"):
            S=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
            f=vec(log,B+f"measurements_contacts_force_{c}_measured")
            t=vec(log,B+f"measurements_contacts_torque_{c}_measured")
            m=S&np.isfinite(f).all(1)&np.isfinite(t).all(1)&(f[:,2]>20)
            if m.sum()<500: print(f"{p:30s}{c[:5]:7s}{m.sum():7d}  too few"); continue
            x=f[m,2]; y=t[m,2]
            A=np.column_stack([x,np.ones_like(x)])
            coef,res,_,_=np.linalg.lstsq(A,y,rcond=None)
            a,b=coef; pred=A@coef; resid=y-pred
            s2=float(np.sum(resid**2)/(len(x)-2)); cov=s2*np.linalg.inv(A.T@A)
            sb=np.sqrt(cov[1,1]); r2=1-np.sum(resid**2)/np.sum((y-y.mean())**2)
            print(f"{p:30s}{c[:5]:7s}{m.sum():7d}{f'{x.min():.0f}-{x.max():.0f}':>16s}"
                  f"{a:10.5f}{b:13.3f}{f'[{b-1.96*sb:+.2f},{b+1.96*sb:+.2f}]':>20s}{r2:7.3f}")
        del log
