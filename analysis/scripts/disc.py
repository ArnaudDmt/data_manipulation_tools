import sys, numpy as np, os; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
B="Observers_MainObserverPipeline_MCKineticsObserver_MEKF_"
name="KO_TRO2024_RHPS1_1"; mass=58.32
log=read_log(f"Projects/{name}/output_data/kinetics_eval/logReplay_full.bin")
def v3(pfx):
    return np.array([np.asarray(log[f"{pfx}_{a}"],float) for a in "xyz"])
acc_i = np.linalg.norm(v3(f"{B}measurements_accelerometer_Accelerometer_measured")
                       - v3(f"{B}measurements_accelerometer_Accelerometer_predicted"), axis=0)
# contact wrench prediction error, summed over contacts, expressed as an acceleration
tot=np.zeros_like(acc_i)
for c in ("LeftFootCenter","RightFootCenter"):
    try:
        d = v3(f"{B}estimatedState_contact_{c}_forces") - v3(f"{B}prediction_contact_{c}_forces")
        tot += np.linalg.norm(d,axis=0)
    except KeyError: pass
force_acc = tot/mass
# angular term: omega_dot x r is not logged directly; use angular acceleration magnitude as a proxy
try:
    angacc = np.linalg.norm(v3(f"{B}estimatedState_angAcc"),axis=0)
except KeyError:
    angacc = np.full_like(acc_i, np.nan)
m=np.isfinite(acc_i)&np.isfinite(force_acc)
print(f"{name}, n={m.sum()}")
print(f"  accelerometer innovation      median {np.median(acc_i[m]):7.3f}  std {np.std(acc_i[m]):7.3f} m/s^2")
print(f"  contact force error / mass    median {np.median(force_acc[m]):7.3f}  std {np.std(force_acc[m]):7.3f} m/s^2")
if np.isfinite(angacc).any():
    print(f"  |angular acceleration|        median {np.nanmedian(angacc[m]):7.3f} rad/s^2")
c=np.corrcoef(acc_i[m], force_acc[m])[0,1]
print(f"\n  correlation(accelerometer innovation, force error/mass) = {c:+.3f}")
