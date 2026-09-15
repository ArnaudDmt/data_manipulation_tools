#!/usr/bin/env python3
"""Le seul scenario propre : filtres ET derive demarrant au premier instant de la mocap.

Aucune des courbes ci-dessous n'utilise une seule donnee anterieure a la mocap. Les trois filtres
repartent a t=0 du meme etat froid -- pose a cet instant, vitesse nulle, biais nul, torseur non
modelise nul -- et la rampe thermique du BMI088 part de zero au meme instant, comme le ferait une
centrale mise a zero robot a l'arret puis chauffant pendant la marche.

En gris, une variante de controle : le meme RI-EKF avec son etat de biais gele. Il n'estime aucun
biais, donc son erreur de biais est la rampe elle-meme ; il sert de repere pour dire ce que
l'estimation de biais rapporte reellement.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model
import score_coldstart as score

SERIES = [
    ('Kinetics Observer', ('kocold2-ba73545d4c', 'ko_bias_cold2.npz'), '#2471a3', '-', 2.1),
    ('RI-EKF, process du papier', score.SCRATCH / 'cold2-papier.csv', '#c0392b', '-', 1.8),
    ('RI-EKF, process adapte', score.SCRATCH / 'cold2-adapte.csv', '#e59866', ':', 1.8),
    ('RI-EKF, biais gele (controle)', score.SCRATCH / 'cold2-gele.csv', '#95a5a6', '--', 1.3),
]

t, p_t, q_t, first, last = score.geometry()
starts = t[first]
truth = model.at(t)[:, 2]                       # rampe rebasee sur le debut de la fenetre
window = float(np.median(t[last] - t[first]))


def smooth(values, seconds=150.0):
    n = max(1, int(seconds / np.median(np.diff(starts))))
    return np.convolve(values, np.ones(n) / n, mode='same')


figure, panels = plt.subplots(2, 1, figsize=(10.0, 8.0), sharex=True,
                              gridspec_kw={'height_ratios': [1, 1.35]})
panels[0].plot(starts, np.degrees(truth[first]), color='black', lw=2.2, ls='--',
               label='biais z injecte (rampe thermique BMI088)', zorder=4)
report = []
for name, source, colour, style, width in SERIES:
    p_e, q_e, bias = (score.from_ko(*source) if isinstance(source, tuple)
                      else score.from_csv(source, t))
    translation, yaw, tilt = score.errors(p_e, q_e, p_t, q_t, first, last)
    recovered = 100.0 * np.mean(bias[first]) / np.mean(truth[first])
    panels[0].plot(starts, np.degrees(bias[first]), color=colour, lw=width, ls=style,
                   label=f'{name}   ({recovered:+.0f} % rattrapes)')
    panels[1].plot(starts, smooth(yaw), color=colour, lw=width, ls=style,
                   label=f'{name}   (moyenne {yaw.mean():.3f} deg)')
    quarters = [q.mean() for q in np.array_split(yaw, 4)]
    report.append((name, translation.mean() * 1000, yaw.mean(), tilt.mean(),
                   np.degrees(np.abs(truth - bias)[first].mean()), recovered, quarters))

panels[0].set_ylabel('biais gyrometrique en z  (deg/s)')
panels[0].legend(frameon=False, fontsize=9, loc='upper left')
panels[0].grid(alpha=0.3)
panels[0].set_title("Demarrage a froid au premier instant de la mocap, derive partant de zero\n"
                    f"LongWalk -- sous-trajectoires de 10 m = {window:.0f} s, {len(first)} fenetres",
                    fontsize=11)
panels[1].set_ylabel(f'erreur de lacet par sous-trajectoire de 10 m  (deg)\n'
                     'moyenne glissante sur 150 s')
panels[1].set_xlabel('Temps depuis le debut de la fenetre evaluee  (s)')
panels[1].legend(frameon=False, fontsize=9, loc='upper left')
panels[1].grid(alpha=0.3)
figure.tight_layout()

destination = score.ROOT / 'results/paper-rebuild/figures/coldramp_longwalk.pdf'
figure.savefig(destination)
figure.savefig(destination.with_suffix('.png'), dpi=150)
print(f'ecrit {destination}\n')

print(f"{'estimateur':31s} {'transl mm':>9s} {'lacet':>7s} {'tilt':>6s} "
      f"{'|b-b^|':>8s} {'rattrape':>9s}   lacet par quart")
for name, tr, yaw, tilt, err, rec, quarters in report:
    print(f'{name:31s} {tr:9.1f} {yaw:7.4f} {tilt:6.4f} {err:8.5f} {rec:8.1f}%   '
          + ' '.join(f'{q:.3f}' for q in quarters)
          + f'  (x{quarters[3]/quarters[0]:.2f})')
