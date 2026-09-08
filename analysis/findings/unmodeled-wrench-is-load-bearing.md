---
name: unmodeled-wrench-is-load-bearing
description: Never tighten unmodeledForceProcessVariance -- it cancels a real force-sensor artifact
metadata:
  type: project
---

The KO's unmodeled wrench state is not slack to be tuned away. On RHPS1 it sits at 135 N horizontal
and cancels the force-sensor artifact in [[rhps1-force-sensor-tilt]], leaving the total centroid
force at correct physics (565.8 N up at standstill vs 572 required).

**Why:** measured 2026-09-07. Tightening the force terms of `unmodeled_wrench_process` from 0.09
degrades everything monotonically -- RHPS1 walk trans_xy ratio 0.629 -> 1.601 (x100 tighter) ->
2.120 (x10000); slippage 0.626 -> 2.583; yaw on the walk up 4.5x.

**How to apply:** do not propose clamping this state, and do not read a large unmodeled force as
evidence of a bad dynamic model. It is also NOT motion-dependent: 135 N standing still vs 153 N
walking, so it is not a dynamics error that grows with motion.
