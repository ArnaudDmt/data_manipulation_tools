#!/usr/bin/env python3
"""Tableau des variantes du KO sur l'experience a froid rebasee, plus les RI-EKF en reference."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model
import score_coldstart as score

# (nom affiche, source) ; un tuple (run, npz) designe un replay du KO, un chemin un CSV du parseur.
SERIES = [
    ('KO complet', ('kocold2-ba73545d4c', 'ko_bias_cold2.npz')),
    ('KO, raideur ang. = 0', ('ko-noangstiff', 'ko_bias_ko-noangstiff.npz')),
    ('KO, raideur + amort. ang. = 0', ('ko-noang', 'ko_bias_ko-noang.npz')),
    ('KO-PC + biais', ('kopc-bias', 'ko_bias_pc.npz')),
    ('RI-EKF, process papier', score.SCRATCH / 'cold2-papier.csv'),
    ('RI-EKF, process adapte', score.SCRATCH / 'cold2-adapte.csv'),
    ('RI-EKF, biais gele', score.SCRATCH / 'cold2-gele.csv'),
]

t, p_t, q_t, first, last = score.geometry()
truth = model.at(t)[:, 2]
print(f"\n{'estimateur':31s} {'transl mm':>9s} {'lacet':>7s} {'tilt':>7s} {'|b-b^|':>8s} "
      f"{'rattrape':>9s}   lacet par quart")
for name, source in SERIES:
    try:
        p_e, q_e, bias = (score.from_ko(*source) if isinstance(source, tuple)
                          else score.from_csv(source, t))
    except (FileNotFoundError, OSError):
        print(f'{name:31s}   -- pas encore disponible')
        continue
    translation, yaw, tilt = score.errors(p_e, q_e, p_t, q_t, first, last)
    error = np.degrees(np.abs(truth - bias)[first].mean())
    recovered = 100.0 * np.mean(bias[first]) / np.mean(truth[first])
    quarters = [q.mean() for q in np.array_split(yaw, 4)]
    print(f'{name:31s} {translation.mean()*1000:9.1f} {yaw.mean():7.4f} {tilt.mean():7.4f} '
          f'{error:8.5f} {recovered:8.1f}%   ' + ' '.join(f'{v:.3f}' for v in quarters)
          + f'  (x{quarters[3]/quarters[0]:.2f})')
