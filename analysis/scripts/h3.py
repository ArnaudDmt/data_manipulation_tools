"""H3: is the KO's translation error a scale factor on distance travelled?

Over short windows (alignment-invariant), regress the KO's displacement magnitude against the
mocap's. Slope 1 means no scale error; a slope away from 1 is a systematic under/over-travel.
"""
import sys, numpy as np, csv, os
sys.path.insert(0,'scripts')
from scipy.spatial.transform import Rotation as R

for proj in ("KO_TRO2024_RHPS1_3","KO_TRO2024_RHPS1_1","HRP5_MultiContact_1","HRP5_MultiContact_3"):
    p=f"Projects/{proj}/output_data/synchronizedObserversMocapData.csv"
    if not os.path.exists(p): continue
    with open(p) as f:
        rd=csv.reader(f,delimiter=';'); head=next(rd); idx={n:i for i,n in enumerate(head)}
        need=[f"MocapAligner_worldBodyKine_position_{a}" for a in "xyz"]+[f"KO_position_{a}" for a in "xyz"]+\
             [f"MocapAligner_worldBodyKine_ori_{a}" for a in "wxyz"]+[f"KO_orientation_{a}" for a in "wxyz"]
        rows=[]
        for r in rd:
            try: rows.append([float(r[idx[n]]) for n in need])
            except (ValueError,KeyError): rows.append([np.nan]*len(need))
    d=np.array(rows)
    pm=d[:,0:3]; pk=d[:,3:6]
    qm=d[:,6:10][:,[1,2,3,0]]; qk=d[:,10:14][:,[1,2,3,0]]
    ok=np.isfinite(d).all(1)&(np.linalg.norm(qm,axis=1)>1e-6)&(np.linalg.norm(qk,axis=1)>1e-6)
    W=200   # 1 s windows at 200 Hz
    dm,dk=[],[]
    for s in range(0,len(d)-W,W):
        sl=slice(s,s+W)
        if not ok[sl].all(): continue
        # displacement expressed in each estimate's own start frame -> alignment invariant
        Rm=R.from_quat(qm[s]/np.linalg.norm(qm[s])); Rk=R.from_quat(qk[s]/np.linalg.norm(qk[s]))
        vm=Rm.inv().apply(pm[s+W-1]-pm[s]); vk=Rk.inv().apply(pk[s+W-1]-pk[s])
        if np.linalg.norm(vm)<0.02: continue
        dm.append(np.linalg.norm(vm)); dk.append(np.linalg.norm(vk))
    if len(dm)<20: print(f"{proj}: only {len(dm)} windows"); continue
    dm=np.array(dm); dk=np.array(dk)
    slope=float(np.linalg.lstsq(dm[:,None],dk,rcond=None)[0][0])
    r2=1-np.var(dk-dm*slope)/max(np.var(dk),1e-12)
    raw=np.abs(dk-dm); res=np.abs(dk-dm*slope)
    print(f"{proj:22s} windows={len(dm):4d}  slope = {slope:6.4f}"
          f"   |error| median: raw {raw.mean()*1000:6.2f} mm  after scale correction {res.mean()*1000:6.2f} mm"
          f"   -> {100*(1-res.mean()/max(raw.mean(),1e-12)):5.1f}% of the error is the scale")
print("DONE")
