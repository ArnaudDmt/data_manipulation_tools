"""Velocity RMSE per category, KO vs RI-EKF, for two result sets."""
import glob,os,collections,numpy as np
RUNS=[("baseline","W_tuned_a1_posx30"),("wrench calibration","C_wrenchcal")]
def cat(p):
    if "MultiContact" in p: return "MultiContact"
    if "LongWalk" in p: return "LongWalk"
    if "SLIPPAGE" in p.upper(): return "RHPS1 slip"
    return "RHPS1 walk"
def load(pref):
    dirs=glob.glob(f"results/{pref}-*/")
    if not dirs: return None
    acc=collections.defaultdict(lambda: ([],[]))
    for d in dirs:
        for proj in sorted(os.listdir(d)):
            kv=os.path.join(d,proj,"kinetics_velocity.txt")
            if not os.path.isfile(kv): continue
            base=f"Projects/{proj}/output_data/kinetics_eval/reference"
            try:
                ko=np.loadtxt(kv,comments="#",ndmin=2)
                mo=np.loadtxt(f"{base}/mocap_velocity.txt",comments="#",ndmin=2)
                ri=np.loadtxt(f"{base}/riekf_velocity.txt",comments="#",ndmin=2)
            except Exception: continue
            n=min(len(ko),len(mo),len(ri))
            if n<100: continue
            e_ko=ko[:n,1:4]-mo[:n,1:4]; e_ri=ri[:n,1:4]-mo[:n,1:4]
            f=lambda e,c: float(np.sqrt(np.mean(np.sum(e[:,c]**2,axis=1))))
            a,b=acc[cat(proj)]
            a.append((f(e_ko,[0,1]),f(e_ko,[2]))); b.append((f(e_ri,[0,1]),f(e_ri,[2])))
    return {k:(np.mean([x[0] for x in v[0]]),np.mean([x[1] for x in v[0]]),
               np.mean([x[0] for x in v[1]]),np.mean([x[1] for x in v[1]])) for k,v in acc.items() if v[0]}
res={l:load(p) for l,p in RUNS}
cats=["MultiContact","RHPS1 walk","RHPS1 slip"]
for idx,lab,unit in ((0,"vel xy","mm/s"),(1,"vel z","mm/s")):
    print(f"\n--- {lab} RMSE [{unit}]   KO / RI-EKF   (ratio)")
    print(f"{'run':22s}"+"".join(f"{c:>28s}" for c in cats))
    for l,_ in RUNS:
        r=res.get(l)
        if not r: print(f"{l:22s} (no results)"); continue
        row=f"{l:22s}"
        for c in cats:
            v=r.get(c)
            if not v: row+=f"{'-':>28s}"; continue
            ko,ri=v[idx]*1000,v[idx+2]*1000
            row+=f"{ko:10.2f}/{ri:8.2f}({ko/ri:5.3f})"
        print(row)
