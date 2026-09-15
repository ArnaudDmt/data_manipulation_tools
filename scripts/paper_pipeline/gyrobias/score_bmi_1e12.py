#!/usr/bin/env python3
"""Biais BMI088 injecte, depart a froid rampe rebasee : avant (init 1e-8) contre apres (init 1e-12).

Memes erreurs que score_variants.py (sous-trajectoires de 10 m, moyennes), plus la translation
horizontale et verticale separees, comme dans le papier.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model
from score_coldstart import SCRATCH, errors, from_csv, from_ko, geometry  # noqa: E402

SERIES = [
    ('KO, init 1e-8', ('kocold2-ba73545d4c', 'ko_bias_cold2.npz')),
    ('KO, init 1e-12', ('bmi-ko-1e12', 'ko_bias_bmi-ko-1e12.npz')),
    ('KO lineaire, init 1e-8', ('ko-noang', 'ko_bias_ko-noang.npz')),
    ('KO lineaire, init 1e-12', ('bmi-kolin-1e12', 'ko_bias_bmi-kolin-1e12.npz')),
    ('RI-EKF base, init 1e-8', SCRATCH / 'cold2-papier.csv'),
    ('RI-EKF base, init 1e-12', SCRATCH / 'cold3-papier.csv'),
    ('RI-EKF adapte, init 1e-8', SCRATCH / 'cold2-adapte.csv'),
    ('RI-EKF adapte, init 1e-12', SCRATCH / 'cold3-adapte.csv'),
    ('RI-EKF biais gele', SCRATCH / 'cold2-gele.csv'),
]

t, p_t, q_t, first, last = geometry()
truth = model.at(t)[:, 2]
print(f"\n{'estimateur':27s} {'Transxy m':>9s} {'Transz m':>8s} {'tilt':>7s} {'lacet':>7s} "
      f"{'|b-b^|':>8s} {'rattrape':>9s}   lacet par quart")
for name, source in SERIES:
    try:
        p_e, q_e, bias = (from_ko(*source) if isinstance(source, tuple) else from_csv(source, t))
    except (FileNotFoundError, OSError):
        print(f'{name:27s}   -- absent')
        continue
    _, yaw, tilt = errors(p_e, q_e, p_t, q_t, first, last)
    # Meme decomposition que errors(), separee en plan et vertical.
    d_t = q_t[first].inv().apply(p_t[last] - p_t[first])
    d_e = q_e[first].inv().apply(p_e[last] - p_e[first])
    xy = np.linalg.norm((d_e - d_t)[:, :2], axis=1)
    z = np.abs((d_e - d_t)[:, 2])
    error = np.degrees(np.abs(truth - bias)[first].mean())
    recovered = 100.0 * np.mean(bias[first]) / np.mean(truth[first])
    quarters = [q.mean() for q in np.array_split(yaw, 4)]
    print(f'{name:27s} {xy.mean():9.4f} {z.mean():8.4f} {tilt.mean():7.4f} {yaw.mean():7.4f} '
          f'{error:8.5f} {recovered:8.1f}%   ' + ' '.join(f'{v:.3f}' for v in quarters))
