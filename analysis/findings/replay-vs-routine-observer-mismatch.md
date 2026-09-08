---
name: replay-vs-routine-observer-mismatch
description: "RESOLVED — the kinetics_eval replay vs in-pipeline KO gap was build skew, not a protocol gap; with both sides from one revision they agree to 0.014 mm."
metadata: 
  node_type: memory
  type: project
  originSessionId: 287daf9e-ad1c-4931-a07e-923142412c80
  modified: 2026-09-02T00:00:00.000Z
---

**RESOLVED 2026-09-02: the cause was build skew.** The standalone replay node and the mc_rtc
observer plugin were linked against *different builds* of state-observation. Rebuilding both from a
single revision collapsed the residual on HRP5_MultiContact_4 from **2.47 mm to 0.0137 mm mean /
0.112 mm max** over 10,438 samples, with the rpg metrics agreeing to 0.04%. Independently reproduced
by a second session (Codex): 0.0138 mm mean / 0.1121 mm max on a fresh MC_4 tick.

The behaviour change most likely responsible is the contact-contact coupling in the C Jacobian: the
off-diagonal blocks `A.block<sizeForceTangent,sizeForceTangent>(contactForceIndexTangent(i),
contactForceIndexTangent(j))`, i != j, are zero in the old build and non-zero in the new one
(‖F_i<-F_j‖ = 0.349 vs self 0.361, i.e. 97% of the diagonal).

`kinetics_eval.py:replay_dependencies()` now fingerprints `libstate-observation` and
`MCKineticsObserver.so`, so this skew is detected instead of silently re-appearing. **Rebuild and
reinstall both before trusting any A/B comparison.**

**Superseded hypothesis (do not re-propose):** an earlier version of this note blamed the replay
protocol for not carrying per-contact `worldRestPose` or the running EKF covariance `P`. Those
fields genuinely are absent from the bag, but they are *not* the cause of the observed gap — parity
holds without them, because both feet are already in contact at row 0 and the rest poses agree.

**Eliminated by direct measurement (still valid):** every global input (com, comDot, comDotDot,
angMom, additionalWrench, accel, gyro) and every per-contact input (position, linVel, angVel, force,
torque) are **bit-identical** to `mc_log_ui.read_log`; contact `active` flags 0 mismatches/10438;
orientations 2.2e-16 once the log's inverse-quaternion convention is undone; all covariances, flags,
mass and stiffness identical; no CDR field shift; `with_contact_rest_yaw_deflection` true on both
sides; input row off-by-one falsified by rebuilding the bag (2.475 -> 2.457 mm, no effect).

A contact-ID transposition does exist (mc_rtc Left=0, converter Right=0) but is harmless at 3.5 nm.
