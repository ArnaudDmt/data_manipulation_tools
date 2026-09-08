"""Marginal value of a 10% improvement in each metric, from the current best configuration.
This is what the sampler actually feels."""
import sys, glob, copy; sys.path.insert(0,'scripts')
from pathlib import Path
import kinetics_tune as kt
PROJ=["HRP5_MultiContact_1","HRP5_MultiContact_2","HRP5_MultiContact_3","HRP5_MultiContact_4",
 "KO_TRO2024_RHPS1_1","KO_TRO2024_RHPS1_2","KO_TRO2024_RHPS1_3","KO_TRO2024_RHPS1_4","KO_TRO2024_RHPS1_5",
 "KO_TRO_2024_RHPS1_SLIPPAGE_1","KO_TRO_2024_RHPS1_SLIPPAGE_2","KO_TRO_2024_RHPS1_SLIPPAGE_3"]
base=kt.read_ratios(Path(glob.glob("results/PW_cal-*/summary.csv")[0]),PROJ)
VEL={"MultiContact":(1.4854,1.2347),"RHPS1 walk":(1.3441,1.5269),"RHPS1 slip":(1.2180,1.1819)}
for p in PROJ:
    c=kt.category_of(p); base[f"{p}|vel_xy"]=VEL[c][0]; base[f"{p}|vel_z"]=VEL[c][1]
J0=kt.objective(base)
print(f"  baseline objective {J0:+.4f}\n")
print(f"  {'improve by 10%':34s}{'dJ':>9s}{'share of J':>12s}")
scopes=[("trans_xy, RHPS1 only","trans_xy",("RHPS1 walk","RHPS1 slip")),
        ("trans_xy, everywhere","trans_xy",None),
        ("yaw, everywhere","yaw",None),
        ("yaw, RHPS1 walk only","yaw",("RHPS1 walk",)),
        ("vel_xy, everywhere","vel_xy",None),
        ("trans_z, everywhere","trans_z",None),
        ("tilt, everywhere","tilt",None)]
for label,metric,cats in scopes:
    r=dict(base)
    for case in list(r):
        proj,m=case.split("|",1)
        if m==metric and (cats is None or kt.category_of(proj) in cats):
            r[case]*=0.90
    dJ=kt.objective(r)-J0
    print(f"  {label:34s}{dJ:+9.4f}{abs(dJ)/abs(J0)*100:11.1f}%")
# and what clearing one yaw category is worth
r=dict(base)
for case in list(r):
    proj,m=case.split("|",1)
    if m=="yaw" and kt.category_of(proj)=="RHPS1 walk": r[case]=0.94
print(f"\n  {'clear the yaw floor on RHPS1 walk':34s}{kt.objective(r)-J0:+9.4f}")
