---
name: riekf-imu-bias-covariances-inert
description: The RI-EKF's gyro- and accelerometer-bias process covariances do not measurably change its trajectory metrics; do not re-sweep them.
metadata:
  type: project
---

Measured 2026-09-01 on 6 RHPS1 datasets (walk + slippage, 1 m sublength), sweeping
`plugins/HartleyIEKF.yaml`:

- `gyroBiasProcessVariance` 1e-10 (shipped) / 1e-12 / 1e-14 / 1e-18 — saturated by 1e-14.
  Translation moves at most 0.067 mm out of ~30; yaw at most 0.0007 deg out of ~0.27, mostly
  in the worse direction. The filter *does* respond internally (bias range 2.1e-4 vs 6.9e-5
  rad/s) but the trajectory shifts by a median 43 um over a 409 s run.
- `accelerometerBiasVariance` 1e-8 (shipped) / 1e-6 / 1e-10 / 1e-12 — already saturated at the
  shipped value. Tightening changes < 0.006 mm; loosening to 1e-6 degrades translation on 5 of
  6 datasets by up to 0.33 mm.

**Why:** these were checked because the KO's gyroBiasProcessVariance is 1e-18 against the
RI-EKF's 1e-10, which looked like an unfair asymmetry. It is not one at the level of the
reported metrics. It also means the RI-EKF's RHPS1 error is not about IMU bias handling —
consistent with the compliance/scale-bias diagnosis. The remaining untested RI-EKF parameter
is `contactVariance` (1e-4), its whole rigid-contact assumption.

**How to apply:** do not sweep either bias covariance again, and do not change them (the user
decided: "then don't change it"). Changing them to weaken the baseline would be indefensible
anyway.

Sweep harness that avoids re-running the routine: `scratchpad/riekf_sweep.py` — the RI-EKF is
the standalone `~/Documents/HartleyIEKF_WithPlots/bin/InEkfLogParser` reading `HARTLEY_CONFIG`
(+ `HARTLEY_ROBOT`), and its IMU pose maps to the shipped body trajectory by one constant rigid
transform (translation [-0.0478, 0, -0.0300] m, 6 um scatter). Only the 8 RHPS1 projects retain
`output_data/kinetics_eval/HartleyInput.txt`; the 5 HRP5-P ones need a replay first.
RHPS1_5 and SLIPPAGE_3 have clipped shipped trajectories, so the row-for-row transform skips them.
