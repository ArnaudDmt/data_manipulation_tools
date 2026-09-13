"""Keep only the disturbance-wrench columns of a routine log.

The estimator is blind to the left hand in this variant, so the effort still logged at that
sensor is the reference its disturbance-wrench state has to recover.
"""
import sys
import pandas as pd

PREFIX = "Observers_MainObserverPipeline_MCKineticsObserver"
COLUMNS = ["t"] + [
    f"{PREFIX}_{part}_{axis}"
    for part in (f"debug_wrenchesInCentroid_LeftHandForceSensor_{kind}"
                 for kind in ("force", "torque"))
    for axis in "xyz"] + [
    f"{PREFIX}_MEKF_estimatedState_{state}_{axis}"
    for state in ("extForceCentr", "extTorqueCentr") for axis in "xyz"]

source, target = sys.argv[1], sys.argv[2]
header = pd.read_csv(source, delimiter=";", nrows=0)
keep = [c for c in COLUMNS if c in header.columns]
missing = set(COLUMNS) - set(keep)
if missing:
    raise SystemExit(f"colonnes absentes de {source}: {sorted(missing)}")
# Column-selective read: these logs run to 1500 columns and reading one whole has already
# exhausted the machine's memory once.
pd.read_csv(source, delimiter=";", usecols=keep)[keep].to_csv(target, sep=";", index=False)
print(f"{target}: {len(keep)} colonnes")
