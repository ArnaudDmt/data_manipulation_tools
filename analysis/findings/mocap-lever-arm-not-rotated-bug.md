---
name: mocap-lever-arm-not-rotated-bug
description: "resampleMocapAndExtractPose.py rotates the mocap rigid-body-to-limb lever arm by a constant instead of the body's world orientation, corrupting the ground-truth reference for every dataset."
metadata: 
  node_type: memory
  type: project
  originSessionId: 287daf9e-ad1c-4931-a07e-923142412c80
  modified: 2026-09-01T10:06:24.346Z
---

**FIXED 2026-09-01.** One line; `world_RigidBody_Ori_R` now used. Verified end-to-end by re-running
the script on HRP5_MultiContact_4: the world-frame offset std went 6e-17 -> 0.002-0.006 m and the
body-frame std 0.003-0.006 -> 2e-14 m, i.e. the lever arm is now rigid to the body rather than to
the world. **Every mocap-derived artefact predating the fix is still stale** — `resampledMocapData.csv`,
`finalDataCSV.csv`, the eval references and every number in `results/` — so a full routine re-run is
required before any of them is quoted.

Originally found in `scripts/resampleMocapAndExtractPose.py`, the two lines that build the mocap
reference pose:

```python
world_MocapLimb_Ori_R = world_RigidBody_Ori_R * rigidBody_MocapLimb_Ori_R          # correct
world_MocapLimb_Pos = world_RigidBody_Pos + rigidBody_MocapLimb_Ori_R.apply(rigidBody_MocapLimb_Pos)
```

The position line rotates the lever arm by `rigidBody_MocapLimb_Ori_R` (the *constant* body->limb
rotation) where it must use `world_RigidBody_Ori_R` (the time-varying world orientation), as the
orientation line above it correctly does. The lever arm therefore never rotates with the robot: the
mocap "limb" trajectory is the mocap rigid-body trajectory plus a fixed world-frame vector.

**Proof (alignment-invariant, no fitting):** in HRP5P_LongWalk,
`world_MocapLimb_Pos - RigidBody001_t` is constant to machine precision — std 1.9e-16 m, spread
3e-15 m over 487548 samples — while the rigid body yaws through 13330 deg (1175 deg range). A
correct rigid transform would sweep that offset by up to 2*|p| = 2.06 m.

Here |p| = 1.028 m (`rigidBody_MocapLimb_Pos` = [-0.106, +0.188, -1.005] m), so the induced
reference error is roughly |p_horizontal| * dyaw per segment, and it is *worst on the datasets with
the most turning*. LongWalk accumulates 11588 deg of yaw, which is why it shows up there first.

**Impact on the KO-vs-RI-EKF conclusion (LongWalk, installed config, reference rebuilt offline):**

| sublength | KO/RI as shipped | KO/RI with lever arm rotating |
|---|---:|---:|
| 1 m | 1.148 | **1.016** |
| 2 m | 1.175 | **1.023** |
| 5 m | 1.218 | **1.038** |
| 10 m | 1.215 | **1.037** |

The bug flatters the RI-EKF and penalises the KO: fixing it takes the KO's LongWalk translation
deficit from 21.5% to 3.7%. So **the LongWalk translation gap is mostly a ground-truth artefact**,
not an observer deficiency.

**Caveat, not yet closed:** the corrected numbers come from rebuilding the reference offline
(applying a rotating lever arm to the shipped mocap), not from re-running the pipeline. A
cross-check against the controller's own kinematics was ambiguous (1 m favoured the shipped
reference, 2 m the corrected one), so the *magnitude* needs a real re-run of
`resampleMocapAndExtractPose.py` + downstream before it goes in the thesis. The *bug itself* is
certain.

Related: [[longwalk-sublength-recheck]] — the ratio's sublength dependence (1.148 at 1 m vs 1.215 at
10 m) largely disappears once the reference is fixed.
