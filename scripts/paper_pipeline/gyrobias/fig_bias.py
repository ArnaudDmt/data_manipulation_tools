#!/usr/bin/env python3
"""Biais gyrometrique injecte contre biais estime, et l'erreur de lacet qui en resulte.

La verite est connue exactement -- c'est nous qui l'avons injectee -- ce qui est le seul cas ou l'on
voit ce que chaque filtre rattrape reellement. Sur les donnees telles quelles HRP5-P porte un gyro a
fibre optique qui ne derive pas (0.0003 deg/s mesure), donc il n'y a rien a rattraper et la
difference entre les deux estimateurs reste invisible.

Trois estimateurs sont traces :
  - RI-EKF, process du papier        gyroBiasProcessVariance 1.4e-16 continu
  - RI-EKF, process adapte           1.4e-10, soit 700 000 fois plus lache
  - Kinetics Observer                config de reference, non adaptee

Le lacet est note avec la meme mesure pour tous : la moyenne, sur toutes les sous-trajectoires de
10 m, de l'erreur de rotation autour de la verticale. Un desalignement constant du monde s'annule
dans cette difference relative, ce qui permet de partir du CSV brut du parseur -- verifie : l'ecart
entre ce CSV et la trajectoire alignee du pipeline est un lacet monde constant de -0.257 deg.

Matplotlib plutot que plotly : l'export kaleido reclame un Chrome qui n'est pas installe.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
SCRATCH = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/'
               '287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/bias')
PROJECT = 'HRP5P_LongWalk'
AXES = ('x', 'y', 'z')
EVAL_START = 1002.28        # la fenetre notee commence la ; avant, le filtre construit son etat
SUBLENGTH = 10.0            # la longueur de sous-trajectoire de la categorie Longwalk

BIAS_COLUMNS = [f'IMU_GyroBias_{a}' for a in AXES]
QUATERNION = [f'IMU_Orientation_{a}' for a in 'xyzw']

SERIES = [
    ('RI-EKF, process du papier', SCRATCH / f'{PROJECT}-biased.csv', '#c0392b', '-'),
    ('RI-EKF, process adapte', SCRATCH / f'{PROJECT}-biased-adapted.csv', '#e59866', ':'),
    ('Kinetics Observer', SCRATCH / 'ko_bias_bmi.npz', '#2471a3', '-'),
]


def geometry():
    """Instants, verite de rotation, et les bornes de chaque sous-trajectoire de 10 m."""
    data = np.load(ROOT / f'results/paper-rebuild/runs/clean/{PROJECT}/Hartley_traj10.npz')
    estimate, truth = data['estimate'].astype(float), data['truth'].astype(float)
    t, p_t = estimate[:, 0], truth[:, 1:4]
    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p_t, axis=0), axis=1))])
    ends = np.searchsorted(distance, distance + SUBLENGTH)
    keep = ends < len(distance)
    return t, R.from_quat(truth[:, 4:8]), np.nonzero(keep)[0], ends[keep]


def yaw_error(q_e, q_t, first, last):
    m = ((q_t[first].inv() * q_t[last]).inv() * (q_e[first].inv() * q_e[last])).as_matrix()
    return np.abs(np.degrees(np.arctan2(m[:, 1, 0], m[:, 0, 0])))


def load(path):
    """(instants, biais 3 axes, orientation ou None) pour un CSV de parseur ou un .npz de bag."""
    if path.suffix == '.npz':
        d = np.load(path)
        return d['t'], d['b'], None
    d = pd.read_csv(path, sep=';', usecols=['t'] + BIAS_COLUMNS + QUATERNION)
    return (d['t'].to_numpy(), d[BIAS_COLUMNS].to_numpy(),
            d[QUATERNION].to_numpy())


def thin(t, values, target=4000):
    step = max(1, len(t) // target)
    return t[::step], values[::step]


t_eval, q_t, first, last = geometry()
figure, panels = plt.subplots(3, 1, sharex=True, figsize=(9.5, 8.5))
report = []

for label, path, colour, style in SERIES:
    t, b, quaternions = load(path)
    t_thin, b_thin = thin(t, np.degrees(b))
    for row in range(3):
        panels[row].plot(t_thin, b_thin[:, row], color=colour, lw=1.3, ls=style,
                         label=label, zorder=2)
    index = np.searchsorted(t, t_eval + EVAL_START).clip(0, len(t) - 1)
    truth = model.at(t_eval + EVAL_START)
    recovered = 100.0 * np.mean(b[index][:, 2]) / np.mean(truth[:, 2])
    yaw = (yaw_error(R.from_quat(quaternions[index]), q_t, first, last)
           if quaternions is not None else None)
    report.append((label, np.degrees(b[-1]), recovered, yaw))

truth_thin = np.linspace(0.0, float(t[-1]), 4000)
truth_curve = np.degrees(model.at(truth_thin))
for row, axis in enumerate(AXES):
    panels[row].axvspan(0.0, EVAL_START, color='0.92', zorder=0)
    panels[row].plot(truth_thin, truth_curve[:, row], color='black', lw=2.0, ls='--',
                     label='Biais injecte', zorder=3)
    panels[row].set_ylabel(f'biais {axis}  (deg/s)')
    panels[row].grid(alpha=0.3)
handles, labels = panels[0].get_legend_handles_labels()
order = [labels.index('Biais injecte')] + [i for i, l in enumerate(labels) if l != 'Biais injecte']
panels[0].legend([handles[i] for i in order], [labels[i] for i in order],
                 ncol=2, loc='upper left', frameon=False, fontsize=9)
panels[0].set_title("Biais gyrometrique injecte et estime -- LongWalk, derive d'une centrale BMI088")
panels[-1].set_xlabel('Temps (s)   (la zone grisee precede la fenetre evaluee)')
figure.tight_layout()

destination = ROOT / 'results/paper-rebuild/figures/injectedGyroBias_longwalk.pdf'
destination.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(destination)
figure.savefig(destination.with_suffix('.png'), dpi=150)
print(f'ecrit {destination}\n')

end_truth = np.degrees(model.at(np.array([float(t[-1])]))[0])
print(f"biais injecte en fin de log : " + '  '.join(f'{a}={v:+.5f}' for a, v in zip(AXES, end_truth))
      + '  deg/s\n')
print(f"{'estimateur':28s} {'bx':>9s} {'by':>9s} {'bz':>9s} {'bz rattrape':>12s} {'lacet 10 m':>11s}")
for label, final, recovered, yaw in report:
    tail = f'{yaw.mean():11.4f}' if yaw is not None else f'{"(rpg)":>11s}'
    print(f'{label:28s} {final[0]:+9.5f} {final[1]:+9.5f} {final[2]:+9.5f} '
          f'{recovered:11.1f}% {tail}')
