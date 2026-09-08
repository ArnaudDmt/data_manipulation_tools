---
name: coulomb-limit-incompatible-with-ekf
description: A friction limit with return mapping on the contact rest pose breaks the EKF; do not retry it as a drop-in
metadata:
  type: project
---

Implemented and measured 2026-09-07, behind `contactFrictionCoefficient` (0 = off, bit-exact,
both observer test suites pass). In `stateDynamics`, when the visco-elastic tangential force
exceeds mu*f_n it is capped AND the contact rest position is advanced so the remaining deflection
reproduces the capped force -- the return mapping of an elastoplastic friction element.

**It does not work.** 12 datasets, trans_xy KO/RI (MultiContact / walk / slip / slip-degradation):

    mu off    1.026 / 0.556 / 0.558 / 1.005
    mu 0.8    1.220 / 0.604 / 0.648 / 1.074
    mu 0.5    1.614 / 0.821 / 0.924 / 1.126

Monotonically worse as the limit binds, and the slippage degradation moves the WRONG way. Yaw is
untouched (only the tangential force was capped). With `withFiniteDifferences: true` it diverges
outright: walk trans_xy 819.4, walk yaw 3.165.

**Why:** the return mapping moves the rest pose discontinuously while the EKF propagates its
covariance as if the dynamics were smooth -- the state jumps and the covariance does not know.
Finite differences cannot rescue it either, since a 1e-6 probe straddling the saturation boundary
returns a near-infinite derivative. This is a structural mismatch between an elastoplastic reset
map and an EKF, not an implementation slip.

**How to apply:** do not retry this as a drop-in flag. The physics argument still stands -- friction
is not compliance, see [[rhps1-yaw-torque-friction]] -- but exploiting it requires modelling the
reset as a jump with its own Jacobian on the covariance propagation, which is a serious piece of
work. Capping the force WITHOUT the return mapping is worse than useless: it makes the prediction
agree with the saturated measurement and removes the innovation that corrects the rest pose.
