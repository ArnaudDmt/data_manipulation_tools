---
name: corrupt-replay-duplicate-messages
description: Some kinetics_eval runs silently produce duplicated trajectory messages that read as filter divergence; detect by row count in kinetics.txt.
metadata: 
  node_type: memory
  type: project
  originSessionId: 287daf9e-ad1c-4931-a07e-923142412c80
  modified: 2026-09-01T07:54:42.369Z
---

Measured 2026-09-01. 10 of 703 runs under `results/` have a `kinetics.txt` with 2-2.7x the
expected row count and interleaved (non-monotonic) timestamps — the same time span published
several times into one output bag, which `extract_ros_trajectories` concatenates.

These read as catastrophic filter divergence but are **artifacts**:
- `bt-1e-9-HRP5P_LongWalk-0` and `-1`: trans 3.55 / 3.61 m, yaw ~108 deg. A third run with the
  byte-identical config (`lwc-1e-9-0-f3fbd7b48e`, same hash `f3fbd7b48e`) gives trans 0.2006.
- `v13-baseline-2.5e-7-HRP5P_LongWalk`: trans 3.607, yaw 107.5. The clean run at that anchor
  (`vf-baseline-2.5e-7-HRP5P_LongWalk-0`) gives 0.2241.
- 3 more inside `tune-screen-0031/0037/0043` (HRP5_MultiContact_2/3), so the Optuna objective
  scored garbage on those trials.

**Detection:** expected rows per project are LongWalk 1474096, MC_1 11151, MC_2 10249,
MC_3 10581, MC_4 10438, RHPS1_1 81792, _2 104897, _3 80813, _4 97257, _5 96021,
SLIPPAGE_1 20800, _2 22600, _3 21200. Any other count, or `np.diff(t) <= 0` anywhere, means
discard the run. Replays that are *not* corrupted are bit-reproducible: identical config hash
gives a byte-identical `kinetics.txt`.

**How to apply:** never conclude "the filter diverged" from a LongWalk run without checking the
row count first. Worth adding the check to `kinetics_eval.py` after extraction so a corrupt run
fails loudly instead of scoring. Related: [[longwalk-sublength-recheck]].
