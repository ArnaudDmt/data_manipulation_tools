#!/usr/bin/env python3
"""Erreur relative de lacet au fil du temps sur LongWalk, sans biais injecte.

La RPE du papier est une moyenne sur toutes les sous-trajectoires ; elle dit ou on arrive, pas
comment on y arrive. Ici chaque sous-trajectoire de 10 m est portee a l'instant de son debut, ce
qui montre si l'erreur est stationnaire, si elle derive, ou si elle est portee par quelques
episodes.

Les trajectoires decimees a 10 Hz suffisent : l'angle d'une sous-trajectoire de 10 m ne depend pas
du contenu haute frequence. La translation, elle, en dependrait -- la longueur de chemin mesuree
raccourcit a la decimation -- donc on ne trace que le lacet.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
PROJECT = 'HRP5P_LongWalk'
SUBLENGTH = 10.0            # la longueur de sous-trajectoire de la categorie Longwalk
SMOOTH = 60.0               # s, fenetre de la moyenne glissante


def yaw_errors(estimator):
    data = np.load(ROOT / f'results/paper-rebuild/runs/clean/{PROJECT}/{estimator}_traj10.npz')
    estimate, truth = data['estimate'].astype(float), data['truth'].astype(float)
    time = estimate[:, 0]
    p_t, q_t = truth[:, 1:4], R.from_quat(truth[:, 4:8])
    q_e = R.from_quat(estimate[:, 4:8])
    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p_t, axis=0), axis=1))])
    ends = np.searchsorted(distance, distance + SUBLENGTH)
    starts, values = [], []
    for i in range(len(distance)):
        j = ends[i]
        if j >= len(distance):
            break
        delta_e = q_e[i].inv() * q_e[j]
        delta_t = q_t[i].inv() * q_t[j]
        m = (delta_t.inv() * delta_e).as_matrix()
        starts.append(time[i])
        values.append(abs(np.degrees(np.arctan2(m[1, 0], m[0, 0]))))
    return np.array(starts), np.array(values)


def smooth(t, v, window=SMOOTH):
    step = np.median(np.diff(t))
    n = max(1, int(window / step))
    kernel = np.ones(n) / n
    return t, np.convolve(v, kernel, mode='same')


figure, panels = plt.subplots(2, 1, sharex=True, figsize=(9.5, 6.5),
                              gridspec_kw={'height_ratios': [2, 1]})
summary = {}
for estimator, label, colour in (('Hartley', 'RI-EKF', '#c0392b'),
                                 ('KO', 'Kinetics Observer', '#2471a3')):
    t, v = yaw_errors(estimator)
    summary[label] = v
    panels[0].plot(t, v, color=colour, lw=0.5, alpha=0.25)
    panels[0].plot(*smooth(t, v), color=colour, lw=1.8, label=f'{label}  (moyenne {v.mean():.3f} deg)')
    # Moyenne cumulee : ou en serait la RPE si le log s'arretait la.
    panels[1].plot(t, np.cumsum(v) / np.arange(1, len(v) + 1), color=colour, lw=1.6, label=label)

panels[0].set_ylabel('erreur de lacet par sous-trajectoire de 10 m  (deg)')
panels[0].legend(frameon=False)
panels[0].grid(alpha=0.3)
panels[0].set_title("Erreur relative de lacet au fil du temps -- LongWalk, sans biais injecte\n"
                    "trait fin : chaque sous-trajectoire ; trait epais : moyenne glissante sur 60 s")
panels[1].set_ylabel('moyenne cumulee  (deg)')
panels[1].set_xlabel("Temps depuis le debut de la fenetre evaluee  (s)")
panels[1].grid(alpha=0.3)
panels[1].legend(frameon=False)
figure.tight_layout()

destination = ROOT / 'results/paper-rebuild/figures/yawRPE_over_time_longwalk.pdf'
destination.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(destination)
figure.savefig(destination.with_suffix('.png'), dpi=150)
print(f'ecrit {destination}\n')

for label, v in summary.items():
    quarters = np.array_split(v, 4)
    print(f"{label:20s} moyenne {v.mean():.4f} deg   par quart : "
          + '  '.join(f'{q.mean():.4f}' for q in quarters))
print(f"\nrapport KO / RI-EKF : {summary['Kinetics Observer'].mean() / summary['RI-EKF'].mean():.3f}")
