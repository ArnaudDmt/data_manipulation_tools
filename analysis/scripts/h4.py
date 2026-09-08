"""H4: is the translation error mostly cross-track, i.e. caused by heading error?

Displacement over a window, expressed in each estimate's own start frame. Decompose the error into
the component along the mocap displacement (along-track) and perpendicular to it (cross-track).
Heading error rotates the displacement, so it shows up almost entirely cross-track.
"""
import sys, numpy as np, csv, os
sys.path.insert(0,'scripts')
from scipy.spatial.transform import Rotation as R

for proj in ("KO_TRO2024_RHPS1_3","KO_TRO2024_RHPS1_1","HRP5_MultiContact_1"):
    p=f"Projects/{proj}/output_data/synchronizedObserversMocapData.csv"
    if not os.path.exists(p): continue
    with open(p) as f:
        rd=csv.reader(f,delimiter=';'); head=next(rd); idx={n:i for i,n in enumerate(head)}
        need=[f"MocapAligner_worldBodyKine_position_{a}" for a in "xyz"]+[f"KO_position_{a}" for a in "xyz"]+\
             [f"MocapAligner_worldBodyKine_ori_{a}" for a in "wxyz"]+[f"KO_orientation_{a}" for a in "wxyz"]+\
             [f"Hartley_IMU_position_{a}" for a in "xyz"]+[f"Hartley_IMU_orientation_{a}" for a in "xyzw"]
        rows=[]
        for r in rd:
            try: rows.append([float(r[idx[n]]) for n in need])
            except (ValueError,KeyError): rows.append([np.nan]*len(need))
    d=np.array(rows)
    pm=d[:,0:3]; qm=d[:,6:10][:,[1,2,3,0]]
    variants={"KO":(d[:,3:6], d[:,10:14][:,[1,2,3,0]]), "RI-EKF":(d[:,14:17], d[:,17:21])}
    W=200
    for vname,(pk,qk) in variants.items():
        ok=np.isfinite(d).all(1)&(np.linalg.norm(qm,axis=1)>1e-6)&(np.linalg.norm(qk,axis=1)>1e-6)
        along,cross=[],[]
        for s0 in range(0,len(d)-W,W):
            sl=slice(s0,s0+W)
            if not ok[sl].all(): continue
            Rm=R.from_quat(qm[s0]/np.linalg.norm(qm[s0])); Rk=R.from_quat(qk[s0]/np.linalg.norm(qk[s0]))
            vm=Rm.inv().apply(pm[s0+W-1]-pm[s0]); vk=Rk.inv().apply(pk[s0+W-1]-pk[s0])
            L=np.linalg.norm(vm[:2])
            if L<0.02: continue
            u=vm[:2]/L; n2=np.array([-u[1],u[0]]); e=vk[:2]-vm[:2]
            along.append(abs(e@u)); cross.append(abs(e@n2))
        if len(along)<20: continue
        a=np.array(along); c=np.array(cross)
        print(f"{proj:22s} {vname:7s} n={len(a):4d}  along {a.mean()*1000:7.2f} mm   cross {c.mean()*1000:7.2f} mm"
              f"   cross/along {c.mean()/max(a.mean(),1e-12):5.2f}")
print("DONE")
