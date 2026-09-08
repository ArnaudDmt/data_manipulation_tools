---
name: longwalk-force-sensor-zero-drift
description: "HRP5P foot force sensors drift ~+25 N per foot in fz over the 28 min LongWalk run; real, but absorbed by the unmodeled-wrench state."
metadata: 
  node_type: memory
  type: project
  originSessionId: 287daf9e-ad1c-4931-a07e-923142412c80
  modified: 2026-09-01T08:21:26.701Z
---

Measured 2026-09-01 from `logReplay_allkeys.bin` (raw `LeftFootForceSensor` / `RightFootForceSensor`,
the only place the unloaded-foot reading survives; `logReplay_full.bin` has only the KO channels).

Swing-phase (unloaded) `fz`, averaged per eighth of the run, foot unloaded when fz < 10% of its own
median stance load:

| eighth | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|
| left fz | 22.1 | 26.7 | 35.4 | 43.6 |
| right fz | 20.7 | 23.8 | 34.9 | 44.4 |

(eighths 1-2 are standing double support, 7-8 have too few swing samples). fx and fy drift only
1.9 -> 4.9 N, so the drift is essentially vertical. Total foot fz reads 937 N at the start against
m*g = 931.7 N (correct to 5 N) and 1059 N at the end — a ramp of 2.77 N/min, tracked exactly by the
KO's `unbiasedExtForce_z` state, which goes -6.7 -> -127.6 N across the same eighths.

**It is not the cause of the height drift.** The unmodeled-wrench state absorbs essentially all of
it: the residual leak is ~1 um/s^2, six orders of magnitude short of the observed +0.81 mm/s. See
[[longwalk-vertical-drift-double-support]] for the real mechanism. Do not re-propose the sensor
drift as the vertical explanation.

**Why it still matters:** by the end of the run a ~45 N zero offset on an *unloaded* foot is about
half the Schmitt release threshold (`schmittTriggerLowerPropThreshold: 0.1` x 931.7 N = 93 N), so it
eats half the contact-detection margin and can delay liftoff detection. A swing-phase re-zero is
cheap — each foot is unloaded ~14% of the time — and is an input fix, not a filter change. The
RI-EKF uses force only for contact detection, so de-biasing is a KO-side gain that is still
physically justified, like the RHPS1 mass fix.
