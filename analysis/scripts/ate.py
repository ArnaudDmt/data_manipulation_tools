"""ATE with the same alignment the pipeline's RPE config declares: posyaw, all frames.
Estimate is aligned to ground truth by a yaw rotation + translation, then ATE is the RMSE of
the position difference. Reported as full 3D, and split horizontal / vertical."""
import sys, glob, numpy as np, collections, statistics
sys.path.insert(0,'scripts'); import kinetics_tune as kt
from scipy.spatial.transform import Rotation as Rot
def yaw_of(q):
    m=Rot.from_quat(q).as_matrix(); return np.arctan2(m[1,0],m[0,0])
def align_posyaw(est, gt):
    """One yaw + translation fitted over all frames (rpg 'posyaw', align_num_frames -1)."""
    ep,gp=est[:,:3],gt[:,:3]
    ec,gc=ep-ep.mean(0),gp-gp.mean(0)
    # best yaw about z by Procrustes on the horizontal components
    num=np.sum(ec[:,0]*gc[:,1]-ec[:,1]*gc[:,0]); den=np.sum(ec[:,0]*gc[:,0]+ec[:,1]*gc[:,1])
    th=np.arctan2(num,den)
    R=np.array([[np.cos(th),-np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]])
    return (R@ep.T).T + (gp.mean(0)-(R@ep.mean(0)))
acc=collections.defaultdict(lambda: collections.defaultdict(list))
for p in kt.PROJECTS:
    base=f'Projects/{p}/output_data/kinetics_eval/reference'
    ko=glob.glob(f'results/kinetics-lw/best-best/{p}/kinetics.txt')
    if not ko: continue
    gt=np.loadtxt(f'{base}/mocap.txt',comments='#',ndmin=2)
    t=gt[:,0]
    it=lambda a: np.column_stack([np.interp(t,a[:,0],a[:,c]) for c in range(1,8)])
    for lab,f in (('KO',ko[0]),('RI-EKF',f'{base}/riekf.txt')):
        E=it(np.loadtxt(f,comments='#',ndmin=2))
        A=align_posyaw(E,gt[:,1:8])
        d=A-gt[:,1:4]
        acc[kt.category_of(p)][lab].append((
            float(np.sqrt(np.mean(np.sum(d**2,axis=1)))),
            float(np.sqrt(np.mean(np.sum(d[:,:2]**2,axis=1)))),
            float(np.sqrt(np.mean(d[:,2]**2)))))
cats=['MultiContact','LongWalk','RHPS1 walk','RHPS1 slip']
print('  ATE ratio, Kinetics / RI-EKF   (<1 = KO better)')
print(f"  {'component':14s}"+''.join(f'{c:>16s}' for c in cats))
for i,name in ((0,'ATE 3D'),(1,'ATE horizontal'),(2,'ATE vertical')):
    row=f'  {name:14s}'
    for c in cats:
        d=acc[c]
        if 'KO' in d:
            ko=statistics.fmean(v[i] for v in d['KO']); ri=statistics.fmean(v[i] for v in d['RI-EKF'])
            row+=f'{ko/ri:16.3f}'
        else: row+=f'{"-":>16s}'
    print(row)
print()
print('  ATE (posyaw-aligned, RMSE over the whole trajectory) [mm]')
print(f"  {'component':14s}"+''.join(f'{c:>22s}' for c in cats))
for i,name in ((0,'ATE 3D'),(1,'ATE horizontal'),(2,'ATE vertical')):
    row=f'  {name:14s}'
    for c in cats:
        d=acc[c]
        if 'KO' in d:
            row+=f"{statistics.fmean(v[i] for v in d['KO'])*1000:10.1f} /{statistics.fmean(v[i] for v in d['RI-EKF'])*1000:9.1f}"
        else: row+=f'{"-":>22s}'
    print(row)
