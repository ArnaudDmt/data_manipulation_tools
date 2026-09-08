---
name: rhps1-force-sensor-tilt
description: RHPS1 foot force sensors read 18 deg off vertical; correcting it buys 4-5% translation but not velocity or slippage
metadata:
  type: project
---

RHPS1's foot force sensors report a load-proportional tangential force of ~0.29x the normal load,
in a direction fixed to each foot. Measured 2026-09-07 on the mc_rtc routine logs
(`logReplay_full.bin`, which is the conversion *input*, not the replay output).

Evidence: at standstill the two feet sum to 147 N of net horizontal force (impossible, 2.5 m/s2 on
a static robot) while the vertical is correct at 568 N vs 572 required. The ratio holds at 0.283-0.328
across both feet, all 5 walks and all 3 slippage runs, at both 286 N and 390 N normal load. Tilt
identified from single support = 18.5-19.5 deg (left) and 17.4-18.7 deg (right), stable to 1 deg.
HRP5-P reads 2-6% and is NOT consistent across its own datasets (MultiContact 2.8/6.2 deg vs
LongWalk 2.4/2.0 deg), so that one is real friction, not calibration -- leave HRP5-P uncorrected.
Confirmed against mocap attitude, not just the estimator's (19.00 vs 19.00 deg), so it is not circular.
The torque does NOT share the defect: CoP is already inside the sole and a rotation moves it <1.5 mm.
Correct the force only.

**Why:** the KO absorbs this with the unmodeled wrench, which parks at 135 N to cancel it. That is
why tightening `unmodeledForceProcessVariance` is catastrophic (RHPS1 trans_xy ratio 0.629 -> 1.601
at x100, -> 2.120 at x10000) -- see [[unmodeled-wrench-is-load-bearing]]. The RI-EKF never reads the
wrench, so this penalises only the KO.

**How to apply:** `wrenchCalibration: {LeftFootCenter: [0.047962, 0.324858, 0.015621],
RightFootCenter: [-0.282761, 0.142506, 0.001466]}` (rotvec, rad) in the mc_rtc rhps1.yaml. Plumbed
through `load_mc_rtc_configuration` -> `wrench_calibration` -> msg `wrench_tilt_correction` -> bridge.
Changing the observer config forces a full `prepare --force` (~24 s/dataset): the contact config is
baked into the bag at conversion time.

Measured effect over 12 datasets. On the RPE metrics it is real but small: trans_xy RHPS1 walk
0.629 -> 0.603, slip 0.626 -> 0.596; MultiContact unchanged; yaw 1.077 -> 1.074; velocity slightly
WORSE (vel_xy walk 1.327 -> 1.344); slippage degradation 0.996 -> 0.988, nowhere near the 0.90 target.

**The real payoff is the perturbation wrench**, which the RPE benchmark never scores and the RI-EKF
cannot estimate at all. Truth is 0 N / 0 N.m (nothing touches the robot); measured at standstill
from the replay bags' `unmodeled_wrench`:

    RHPS1 walk       force 136.9 -> 31.4 N (4.4x)   torque 128.15 -> 16.43 N.m (7.8x)
    RHPS1 slippage   force 150.3 -> 60.6 N (2.5x)   torque 132.94 -> 16.31 N.m (8.2x)
    HRP5P MultiCt.   7.4 N / 3.93 N.m, bit-identical before and after (negative control)

The HRP5P control matches the ~10 N/N.m the paper reports from the hand-removal experiment, which
validates the measurement. Torque improves ~8x although only the FORCE is rotated: the spurious
tangential foot force was generating a spurious moment about the centroid through its lever arm.
Residual torque is suspiciously uniform (15.5-18.5 N.m vs HRP5P's 3.9) -- RHPS1's torque channels
may carry their own offset, not yet investigated.
