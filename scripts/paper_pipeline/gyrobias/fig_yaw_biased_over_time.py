#!/usr/bin/env python3
"""Erreur relative de lacet au fil du temps, avec et sans le biais injecte.

Sans biais, la RPE de lacet est stationnaire : chaque sous-trajectoire est realignee en position et
en lacet, donc seule l'erreur accumulee A L'INTERIEUR de la fenetre compte, et le vrai biais de
HRP5-P (gyro a fibre optique) est nul et constant.

Avec un biais qui CROIT et une estimation qui ne suit pas, l'erreur de biais croit elle aussi, et
l'erreur de lacet par fenetre devrait croitre avec elle. La prediction est directe :

    erreur de lacet(t)  ~  |b(t) - b_estime(t)|  x  duree d'une fenetre de 10 m

C'est ce que cette figure superpose a la mesure.
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
SUBLENGTH = 10.0
OFFSET = 1002.28            # la fenetre evaluee commence la dans le log brut
SMOOTH = 120.0              # s, moyenne glissante


def geometry():
    """Instants, verite de rotation, et les paires (debut, fin) de chaque sous-trajectoire."""
    data = np.load(ROOT / f'results/paper-rebuild/runs/clean/{PROJECT}/Hartley_traj10.npz')
    estimate, truth = data['estimate'].astype(float), data['truth'].astype(float)
    t = estimate[:, 0]
    p_t, q_t = truth[:, 1:4], R.from_quat(truth[:, 4:8])
    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p_t, axis=0), axis=1))])
    ends = np.searchsorted(distance, distance + SUBLENGTH)
    keep = ends < len(distance)
    return t, q_t, np.nonzero(keep)[0], ends[keep], R.from_quat(estimate[:, 4:8])


def yaw_of(q_e, q_t, first, last):
    m = ((q_t[first].inv() * q_t[last]).inv() * (q_e[first].inv() * q_e[last])).as_matrix()
    return np.abs(np.degrees(np.arctan2(m[:, 1, 0], m[:, 0, 0])))


def from_parser(path, t):
    """Oriente et biais du parser, echantillonnes aux instants de la trajectoire decimee."""
    d = pd.read_csv(path, sep=';',
                    usecols=['t', 'IMU_Orientation_x', 'IMU_Orientation_y',
                             'IMU_Orientation_z', 'IMU_Orientation_w', 'IMU_GyroBias_z'])
    index = np.searchsorted(d['t'].to_numpy(), t + OFFSET).clip(0, len(d) - 1)
    q = R.from_quat(d[['IMU_Orientation_x', 'IMU_Orientation_y',
                       'IMU_Orientation_z', 'IMU_Orientation_w']].to_numpy()[index])
    return q, d['IMU_GyroBias_z'].to_numpy()[index]


def smooth(t, v, window=SMOOTH):
    n = max(1, int(window / np.median(np.diff(t))))
    return t, np.convolve(v, np.ones(n) / n, mode='same')


t, q_t, first, last, q_reference = geometry()
starts = t[first]
window_seconds = float(np.median(t[last] - t[first]))

figure, panels = plt.subplots(2, 1, sharex=True, figsize=(9.5, 7.2),
                              gridspec_kw={'height_ratios': [2, 1]})

# Controle : la trajectoire de reference, celle qui donne la RPE publiee.
control = yaw_of(q_reference, q_t, first, last)
panels[0].plot(*smooth(starts, control), color='0.55', lw=1.6, ls=':',
               label=f'RI-EKF sans biais  ({control.mean():.3f} deg)')

q_e, b_estimated = from_parser(SCRATCH / f'{PROJECT}-biased.csv', t)
biased = yaw_of(q_e, q_t, first, last)
panels[0].plot(starts, biased, color='#c0392b', lw=0.4, alpha=0.2)
panels[0].plot(*smooth(starts, biased), color='#c0392b', lw=2.0,
               label=f'RI-EKF avec biais BMI088  ({biased.mean():.3f} deg)')

b_true = model.at(t + OFFSET)[:, 2]
error = np.abs(b_true - b_estimated)
predicted = np.degrees(error[first]) * window_seconds
panels[0].plot(starts, predicted, color='black', lw=1.8, ls='--',
               label=r"prediction  $|b-\hat b|\times$ duree de fenetre")

panels[0].set_ylabel('erreur de lacet par sous-trajectoire de 10 m  (deg)')
panels[0].legend(frameon=False, fontsize=9)
panels[0].grid(alpha=0.3)
panels[0].set_title("Erreur relative de lacet au fil du temps -- LongWalk\n"
                    f"sous-trajectoire de {SUBLENGTH:.0f} m = {window_seconds:.0f} s ; "
                    "moyenne glissante sur 120 s")

panels[1].plot(starts, np.degrees(b_true[first]), color='black', lw=1.8, ls='--',
               label='biais z injecte')
panels[1].plot(starts, np.degrees(b_estimated[first]), color='#c0392b', lw=1.6,
               label='biais z estime par le RI-EKF')
panels[1].set_ylabel('biais z  (deg/s)')
panels[1].set_xlabel('Temps depuis le debut de la fenetre evaluee  (s)')
panels[1].legend(frameon=False, fontsize=9)
panels[1].grid(alpha=0.3)
figure.tight_layout()

destination = ROOT / 'results/paper-rebuild/figures/yawRPE_biased_over_time.pdf'
destination.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(destination)
figure.savefig(destination.with_suffix('.png'), dpi=150)
print(f'ecrit {destination}\n')

print(f"{'serie':34s} {'moyenne':>9s}   par quart")
for label, v in (('RI-EKF sans biais', control), ('RI-EKF avec biais', biased),
                 ('prediction |b-b_est| x fenetre', predicted)):
    print(f'{label:34s} {v.mean():9.4f}   '
          + '  '.join(f'{q.mean():.4f}' for q in np.array_split(v, 4)))
print(f"\ncroissance du 1er au 4e quart : "
      f"sans biais x{np.array_split(control,4)[3].mean()/np.array_split(control,4)[0].mean():.2f}, "
      f"avec biais x{np.array_split(biased,4)[3].mean()/np.array_split(biased,4)[0].mean():.2f}")
