---
name: yaw-rpe-depends-on-subtrajectory-length
description: The KO's yaw advantage appears only at long sub-trajectories; RHPS1 is scored at 1 m and LongWalk at 10 m
metadata:
  type: project
---

Yaw RPE ratio (KO/RI-EKF) versus sub-trajectory length, measured 2026-09-07:

    length    RHPS1 walk      LongWalk
    0.5 m       1.148           1.060
    1.0 m       1.070  <-RHPS1  1.009
    2.0 m       0.999           0.918
    4.0 m       0.885           0.775
    8.0 m       0.903           0.558
    10.0 m      0.864           0.490  <-LongWalk

Both robots show the SAME monotonic trend. There is no RHPS1-vs-HRP5P yaw discrepancy -- there is
a 1 m-vs-10 m one. The KO's absolute yaw error is nearly flat with distance (0.403 -> 0.492 over a
20x range) while the RI-EKF's grows (0.351 -> 0.569): bounded error vs accumulating drift.

**Why:** any global yaw figure that looked good was carrying LongWalk's 0.49; drop LongWalk from
the dataset set and the RHPS1 deficit stops being hidden by the average. RHPS1 walk yaw has been
1.074-1.080 in EVERY run of the tuning campaign -- it never improved, it was just averaged over.

**How to apply:** report yaw as a curve over sub-trajectory length for all datasets rather than one
distance per dataset. NOT explained by gyro bias: both estimators converge to the same net bias,
and the KO's actually wanders 3-5x MORE despite 1e-18 vs 1e-10 process variance. The real driver at
short range is [[rhps1-yaw-torque-friction]]. Fairness note: gyroBiasInitVariance matches (1e-8) but
gyroBiasProcessVariance does NOT (KO 1e-18, tuned down from 1e-12; RI-EKF 1e-10, the Hartley
default). Rerunnable via HARTLEY_CONFIG if a reviewer challenges it.
