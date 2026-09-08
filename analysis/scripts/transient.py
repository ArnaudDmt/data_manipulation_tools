"""Is the KO's RPE disadvantage a start-up transient? Split the 1 m sub-trajectories by where
they begin in the recording and report the error in each fifth. Absolute values, mm and deg."""
import sys, glob, numpy as np, collections, statistics
sys.path.insert(0,'scripts'); import kinetics_tune as kt
from scipy.spatial.transform import Rotation as Rot
def yaw_of(R):
    m=R.as_matrix(); return np.arctan2(m[1,0],m[0,0])
BINS=5
acc=collections.defaultdict(lambda: collections.defaultdict(lambda: [[] for _ in range(BINS)]))
for p in kt.NOMINAL_RUNS:
    base=f'Projects/{p}/output_data/kinetics_eval/reference'
    kof=glob.glob(f'results/kinetics-lw/best-best/{p}/kinetics.txt')
    if not kof: continue
    gt=np.loadtxt(f'{base}/mocap.txt',comments='#',ndmin=2); t=gt[:,0]
    it=lambda a: np.column_stack([np.interp(t,a[:,0],a[:,c]) for c in range(1,8)])
    G=gt[:,1:8]
    step=np.r_[0,np.cumsum(np.linalg.norm(np.diff(G[:,:3],axis=0),axis=1))]
    ends=np.searchsorted(step,step+1.0)
    for lab,f in (('KO',kof[0]),('RI-EKF',f'{base}/riekf.txt')):
        E=it(np.loadtxt(f,comments='#',ndmin=2))
        for i in range(0,len(t)-1,20):
            j=ends[i]
            if j>=len(t): break
            Rg=Rot.from_quat(G[i,3:7]); Re=Rot.from_quat(E[i,3:7])
            dg=Rg.inv().apply(G[j,:3]-G[i,:3]); de=Re.inv().apply(E[j,:3]-E[i,:3])
            dy=abs(np.degrees(yaw_of((Rg.inv()*Rot.from_quat(G[j,3:7])).inv()
                                     *(Re.inv()*Rot.from_quat(E[j,3:7])))))
            b=min(BINS-1,int(i/len(t)*BINS))
            acc['trans_z'][lab][b].append(abs(de[2]-dg[2])*1000)
            acc['trans_xy'][lab][b].append(np.linalg.norm((de-dg)[:2])*1000)
            acc['yaw'][lab][b].append(dy)
rms=lambda v: float(np.sqrt(np.mean(np.square(v))))
print('  RHPS1 nominal walks, 1 m sub-trajectories, grouped by position in the recording')
for m,u in (('trans_z','mm'),('trans_xy','mm'),('yaw','deg')):
    print(f'\n  {m} [{u}]')
    print(f"    {'portion':12s}{'KO':>10s}{'RI-EKF':>10s}{'difference':>13s}")
    for b in range(BINS):
        k=acc[m]['KO'][b]; r=acc[m]['RI-EKF'][b]
        if not k: continue
        print(f'    {f"{b*20}-{(b+1)*20}%":12s}{rms(k):10.3f}{rms(r):10.3f}{rms(k)-rms(r):+13.3f}')
