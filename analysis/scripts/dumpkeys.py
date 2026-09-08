import sys; sys.path.insert(0,'scripts')
from mc_log_ui import read_log
log=read_log("Projects/KO_TRO2024_RHPS1_1/output_data/kinetics_eval/logReplay_full.bin")
open(sys.argv[1],"w").write("\n".join(sorted(log.keys())))
