import sys; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
log=read_log("Projects/KO_TRO2024_RHPS1_1/output_data/kinetics_eval/logReplay_full.bin")
ks=sorted(log.keys())
import re
for pat in ("extForce","extTorque","contacts?_.*force","contacts?_.*rest","contacts?_.*position","contacts?_.*orientation","ori(entation)?$"):
    sel=[k for k in ks if re.search(pat,k,re.I)]
    print(f"--- {pat}  ({len(sel)})")
    for k in sel[:26]: print("   ",k)
