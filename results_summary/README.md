# Distilled evaluation results

`results/` holds 7 GB of trajectory dumps and per-sub-trajectory error caches and is not tracked.
This directory is what those caches reduce to, produced by `scripts/paper_pipeline/distill.py`.

## What is here

- `relative_errors.json` — per variant, per dataset, per sub-trajectory length, per metric:
  `count`, `sum_abs`, `sum_sq`. The `riekf` variant is the RI-EKF baseline, which is one offline
  parse shared by every run.
- `velocities.json` — the same three numbers for the local linear velocity error, split into the
  `xy` norm and `z`.
- `macros/` — the LaTeX macros the paper includes.
- `overlays/`, `MCKineticsObserver.yaml` — the tuning each variant was run with.

## Why three numbers per trial

The paper reports, per category, the mean of the absolute errors pooled across that category's
trials and their population standard deviation. Both follow from the three:

    mean = sum(sum_abs) / sum(count)
    std  = sqrt(sum(sum_sq) / sum(count) - mean^2)

This is exact, not an approximation — checked against pooling the raw arrays, agreeing to 1.6e-15
relative over 128 statistics. What is lost is the distribution within a trial: histograms,
quantiles, the shape of the tail. Recovering those means rerunning the evaluation.

## Rebuilding it

    scripts/paper_pipeline/distill.py

after an evaluation, or `run.sh` for the whole rebuild.
