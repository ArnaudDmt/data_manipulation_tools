---
name: rhps1-yaw-torque-friction
description: RHPS1's yaw deficit comes from foot yaw torque being modelled as elastic compliance instead of ground friction
metadata:
  type: project
---

The KO's yaw is worse than the RI-EKF on RHPS1 (ratio 1.070 at 1 m) because the visco-elastic
model treats contact yaw torque as compliance: `tau_z = -K_ang * dyaw`, K_ang = 727 N.m/rad. When
the foot resists a ground FRICTION moment, tau_z is unrelated to twist angle, but the filter
inverts the law and infers a yaw that never happened. Measured 2026-09-07.

Stratified by yaw-torque residual over 40371 sub-trajectories of 1 m (RHPS1 walks):

    Q1 (0.013-0.051 deg)  KO 0.4123  RI 0.4767  ratio 0.865
    Q2 (0.051-0.215 deg)  KO 0.3822  RI 0.4764  ratio 0.802
    Q3 (0.215-0.671 deg)  KO 0.4957  RI 0.3780  ratio 1.311
    Q4 (0.671-1.988 deg)  KO 0.4953  RI 0.3280  ratio 1.510

**In the bottom half of the residual the KO already BEATS the RI-EKF on yaw by 13-20%.** The
overall 1.070 is dragged up entirely by the top half. corr with the KO-RI difference is +0.309
(gyro bias, the rejected explanation, gave +0.113). Control: corr with the RI-EKF's own error is
NEGATIVE (-0.205) -- it never sees torque, and does better in those windows, so they are not
intrinsically hard, only hard for the KO.

Robot-specific in the right way: median |dTz| is the same on both robots (0.4-1.7 vs 0.7-1.1 N.m)
but RHPS1's p90 is 2.7-3.8x HRP5P's (8.5-11.9 vs 3.1-4.4 N.m), and the excess is confined to the
yaw axis -- |dTxy| is identical between robots. Matches the robot split exactly: RHPS1 yaw 1.070,
HRP5P MultiContact 1.00, HRP5P LongWalk 0.49.

**Why:** `contact_wrench_moment_z_scale` was given a search range of 1e-2..1e6 -- the search has
been tuning its way around a model defect by distrusting the measurement, rather than fixing the
constitutive law. See [[yaw-rpe-depends-on-subtrajectory-length]].

**How to apply:** the structural fix is to bound tau_z by what friction can supply (a function of
normal load), not by elastic deflection, and let the contact's rest yaw rotate when it saturates.
Do NOT clamp the visco-elastic FORCE prediction to the friction cone -- that kills the residual
that drives the anchor correction, which is a different and refuted idea.
