---
name: longwalk-vertical-drift-double-support
description: "On HRP5P_LongWalk both KO and RI-EKF climb 1.2-1.5 m in height over 1697 s, and 73% of it is injected during double support."
metadata: 
  node_type: memory
  type: project
  originSessionId: 287daf9e-ad1c-4931-a07e-923142412c80
  modified: 2026-09-01T08:21:10.959Z
---

Measured 2026-09-01 on `results/lwc-1e-9-0-f3fbd7b48e` (installed config, anchor xy 1e-9).

Absolute base height at the end of the 1697 s run, against a mocap that stays in 0.728-0.764 m:

| | start z | end z | drift |
|---|---:|---:|---:|
| mocap | 0.7565 | 0.7413 | — |
| Kinetics | 0.7566 | **2.2541** | +1.513 m |
| RI-EKF | 0.7565 | **1.9688** | +1.227 m |
| Control (pure kinematics) | — | — | **+0.010 m** |

Both IMU-fused estimators climb; the controller's own kinematic estimate does not. So it is not the
leg kinematics and not KO tuning — it is how both filters fuse contacts. The anchor covariance does
not touch it (1e-9 and fully pinned give the same +1.51 m).

**Where it is injected:** splitting the drift by gait phase (foot fz thresholded at 15% of stance load):

- double support: **+1.113 m over 302.8 s = +3.68 mm/s** (18% of the time, 73% of the drift)
- single support: +0.399 m over 1393.9 s = +0.29 mm/s

0.816 mm per gait cycle over 1853 cycles. That is the scale of foot-sole deflection under load
(470 N / 3e5 N/m = 1.6 mm), and it lands at the contact *transition*, not during steady stance —
consistent with a new contact being registered at a height that does not account for the sole
compression state of the outgoing contact.

RPE hides this: over a 10 m segment it is only +0.037 m of vertical bias, so `trans_perc` never
exposed it. Check absolute z, not just relative pose, on long sequences.

Related: [[longwalk-force-sensor-zero-drift]], [[longwalk-sublength-recheck]].
