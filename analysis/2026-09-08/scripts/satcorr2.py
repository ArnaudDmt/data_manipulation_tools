"""RPE-style windowed translation error vs measured-force saturation.
Each window is yaw-aligned at its start (as RPE does), so no global frame fit is needed.
Saturation is the FRACTION of samples in the window at the friction limit, not the mean ratio."""
import sys, glob, numpy as np; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
P="Observers_MainObserverPipeline_MCKineticsObserver_"; B=P+"MEKF_"
CAL={"LeftFootCenter":np.array([0.047962,0.324858,0.015621]),
     "RightFootCenter":np.array([-0.282761,0.142506,0.001466])}
def rod(rv):
    a=np.linalg.norm(rv); k=rv/a
    K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(a)*K+(1-np.cos(a))*K@K
def vec(log,b,c="xyz"): return np.array([np.asarray(log[f"{b}_{k}"],float) for k in c]).T
def yaw(q):    # q = (qx,qy,qz,qw)
    x,y,z,w=q.T
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
def rot2(a): return np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
THRESH=0.40
for name in ("KO_TRO_2024_RHPS1_SLIPPAGE_1","KO_TRO_2024_RHPS1_SLIPPAGE_2","KO_TRO_2024_RHPS1_SLIPPAGE_3"):
    base=f"Projects/{name}/output_data/kinetics_eval"
    gt=np.loadtxt(f"{base}/reference/mocap.txt",comments="#",ndmin=2)
    ri=np.loadtxt(f"{base}/reference/riekf.txt",comments="#",ndmin=2)
    ko=np.loadtxt(glob.glob(f"results/PW_cal-*/{name}/kinetics.txt")[0],comments="#",ndmin=2)
    t=gt[:,0]
    interp=lambda a: np.column_stack([np.interp(t,a[:,0],a[:,c]) for c in range(1,8)])
    G=gt[:,1:8]; R_=interp(ri); K_=interp(ko)
    log=read_log(f"{base}/logReplay_full.bin"); lt=np.asarray(log["t"],float)
    n=len(lt); sat=np.zeros(n)
    for c in ("LeftFootCenter","RightFootCenter"):
        S=np.array([str(s).strip().lower().startswith("set") for s in log[P+f"debug_contactState_isSet_{c}"]])
        f=np.einsum('jk,nk->nj',rod(CAL[c]),vec(log,B+f"measurements_contacts_force_{c}_measured"))
        good=S&np.isfinite(f).all(1)&(f[:,2]>50)
        r=np.zeros(n); r[good]=np.linalg.norm(f[good][:,:2],axis=1)/f[good][:,2]
        sat=np.maximum(sat,r)
    sat_on_t=np.interp(t,lt,sat)
    dt=float(np.median(np.diff(t))); W=max(2,int(round(1.0/dt)))
    starts=np.arange(0,len(t)-W,W//2)
    errs={}
    for lab,E in (("KO",K_),("RI",R_)):
        e=[]
        for s in starts:
            dgt=rot2(float(yaw(G[s:s+1,3:7])[0]))@(G[s+W-1,:2]-G[s,:2])
            dest=rot2(float(yaw(E[s:s+1,3:7])[0]))@(E[s+W-1,:2]-E[s,:2])
            e.append(np.linalg.norm(dest-dgt)*1000)
        errs[lab]=np.array(e)
    frac=np.array([np.mean(sat_on_t[s:s+W]>THRESH) for s in starts])
    mx=np.array([sat_on_t[s:s+W].max() for s in starts])
    hi=frac>0; lo=frac==0
    print(f"\n=== {name}   {len(starts)} windows of 1 s   dt={dt:.4f}")
    print(f"  window max ratio: p50 {np.median(mx):.3f}  p90 {np.percentile(mx,90):.3f}  max {mx.max():.3f}")
    print(f"  windows with any sample >{THRESH}: {hi.sum()} of {len(starts)}")
    print(f"  overall  KO {np.median(errs['KO']):7.2f} mm   RI {np.median(errs['RI']):7.2f} mm"
          f"   ratio {np.median(errs['KO'])/max(np.median(errs['RI']),1e-9):5.3f}   <-- sanity vs benchmark 0.60")
    # Control: saturation may just track how much the robot moved. Stratify by window
    # displacement and compare saturated vs not INSIDE each stratum.
    disp=np.array([np.linalg.norm(G[s+W-1,:2]-G[s,:2]) for s in starts])
    edges=np.percentile(disp,[0,33,66,100]); print(f"  {'displacement stratum':28s}{'n hi/lo':>12s}{'KO/RI unsat':>14s}{'KO/RI sat':>12s}")
    for i in range(3):
        m=(disp>=edges[i])&(disp<=edges[i+1])
        h=m&hi; l=m&lo
        if h.sum()<4 or l.sum()<4:
            print(f"  {f'{edges[i]:.3f}-{edges[i+1]:.3f} m':28s}{f'{h.sum()}/{l.sum()}':>12s}{'too few':>14s}"); continue
        rl=np.median(errs['KO'][l])/max(np.median(errs['RI'][l]),1e-9)
        rh=np.median(errs['KO'][h])/max(np.median(errs['RI'][h]),1e-9)
        print(f"  {f'{edges[i]:.3f}-{edges[i+1]:.3f} m':28s}{f'{h.sum()}/{l.sum()}':>12s}{rl:14.3f}{rh:12.3f}"
              f"   {'worse' if rh>rl else 'better'}")
    if hi.sum()>=5 and lo.sum()>=5:
        rl=np.median(errs['KO'][lo])/max(np.median(errs['RI'][lo]),1e-9)
        rh=np.median(errs['KO'][hi])/max(np.median(errs['RI'][hi]),1e-9)
        print(f"  no-saturation windows  KO {np.median(errs['KO'][lo]):7.2f}  RI {np.median(errs['RI'][lo]):7.2f}  ratio {rl:5.3f}")
        print(f"  saturated windows      KO {np.median(errs['KO'][hi]):7.2f}  RI {np.median(errs['RI'][hi]):7.2f}  ratio {rh:5.3f}")
        print(f"  --> {'KO LOSES ground when saturated' if rh>rl else 'KO does not lose ground when saturated'}")
    else:
        print(f"  (too few windows on one side: hi={hi.sum()} lo={lo.sum()})")
    del log
