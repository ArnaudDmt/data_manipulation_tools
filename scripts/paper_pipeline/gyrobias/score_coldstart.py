#!/usr/bin/env python3
"""Erreurs des trois estimateurs, demarrage chaud contre demarrage a froid au premier instant note.

Une seule mesure pour tous, appliquee aux sorties brutes de chaque estimateur : la moyenne, sur
toutes les sous-trajectoires de 10 m, de l'erreur de translation, de lacet et de tilt. Chaque
sous-trajectoire est comparee en RELATIF, donc un desalignement global du monde s'annule -- ce qui
est verifie : entre le CSV brut du parseur et la trajectoire alignee du pipeline il n'y a qu'un
lacet monde constant de -0.257 deg.
"""
import sys
from pathlib import Path

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
OFFSET = 1002.276
QUATERNION = [f'IMU_Orientation_{a}' for a in 'xyzw']
POSITION = [f'IMU_Position_{a}' for a in 'xyz']


def geometry():
    data = np.load(ROOT / f'results/paper-rebuild/runs/clean/{PROJECT}/Hartley_traj10.npz')
    truth = data['truth'].astype(float)
    t = data['estimate'].astype(float)[:, 0]
    p_t, q_t = truth[:, 1:4], R.from_quat(truth[:, 4:8])
    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p_t, axis=0), axis=1))])
    ends = np.searchsorted(distance, distance + SUBLENGTH)
    keep = ends < len(distance)
    return t, p_t, q_t, np.nonzero(keep)[0], ends[keep]


def errors(p_e, q_e, p_t, q_t, first, last):
    """Erreurs relatives de chaque sous-trajectoire, exprimees dans le repere de son debut."""
    delta_t_p = q_t[first].inv().apply(p_t[last] - p_t[first])
    delta_e_p = q_e[first].inv().apply(p_e[last] - p_e[first])
    translation = np.linalg.norm(delta_e_p - delta_t_p, axis=1)
    m = ((q_t[first].inv() * q_t[last]).inv() * (q_e[first].inv() * q_e[last])).as_matrix()
    yaw = np.abs(np.degrees(np.arctan2(m[:, 1, 0], m[:, 0, 0])))
    tilt = np.abs(np.degrees(np.arccos(np.clip(m[:, 2, 2], -1.0, 1.0))))
    return translation, yaw, tilt


def from_csv(path, t):
    frame = pd.read_csv(path, sep=';', usecols=['t'] + QUATERNION + POSITION + ['IMU_GyroBias_z'])
    index = np.searchsorted(frame['t'].to_numpy(), t + OFFSET).clip(0, len(frame) - 1)
    return (frame[POSITION].to_numpy()[index], R.from_quat(frame[QUATERNION].to_numpy()[index]),
            frame['IMU_GyroBias_z'].to_numpy()[index])


def from_ko(run, npz):
    """Le kinetics.txt du replay porte deja l'horloge de la fenetre, sans decalage a ajouter."""
    k = np.loadtxt(ROOT / f'results/{run}/{PROJECT}/kinetics.txt', skiprows=1)
    index = np.searchsorted(k[:, 0], t).clip(0, len(k) - 1)
    z = np.load(SCRATCH / npz)
    j = np.searchsorted(z['t'], t + OFFSET).clip(0, len(z['t']) - 1)
    return k[index, 1:4], R.from_quat(k[index, 4:8]), z['b'][j, 2]


t, p_t, q_t, first, last = geometry()
window = float(np.median(t[last] - t[first]))

# Deux realisations de la MEME derive : calee sur le debut du log (les filtres chauds l'ont vue
# monter pendant 17 min, les filtres froids de la premiere version encaissaient un echelon de
# 0.0154 deg/s a leur premier instant), ou REBASEE sur le premier instant note, ou elle part de
# zero comme le ferait une centrale mise a zero a l'arret.
BIAS = {'log': model.at(t + OFFSET)[:, 2], 'rebase': model.at(t)[:, 2]}

SERIES = [
    ('RI-EKF papier', 'chaud', 'log', SCRATCH / 'HRP5P_LongWalk-biased.csv'),
    ('RI-EKF papier', 'froid, echelon', 'log', SCRATCH / 'cold-papier.csv'),
    ('RI-EKF papier', 'froid, rampe', 'rebase', SCRATCH / 'cold2-papier.csv'),
    ('RI-EKF adapte', 'chaud', 'log', SCRATCH / 'HRP5P_LongWalk-biased-adapted.csv'),
    ('RI-EKF adapte', 'froid, echelon', 'log', SCRATCH / 'cold-adapte.csv'),
    ('RI-EKF adapte', 'froid, rampe', 'rebase', SCRATCH / 'cold2-adapte.csv'),
    ('Kinetics Observer', 'chaud', 'log', ('kobmi-ba73545d4c', 'ko_bias_bmi.npz')),
    ('Kinetics Observer', 'froid, echelon', 'log', ('kocold-ba73545d4c', 'ko_bias_cold.npz')),
    ('Kinetics Observer', 'froid, rampe', 'rebase', ('kocold2-ba73545d4c', 'ko_bias_cold2.npz')),
]

print(f'sous-trajectoires de {SUBLENGTH:.0f} m = {window:.0f} s, {len(first)} fenetres\n')
print(f"{'estimateur':20s} {'depart':16s} {'transl mm':>10s} {'lacet deg':>10s} "
      f"{'tilt deg':>9s} {'|b-b^| deg/s':>13s} {'bz rattrape':>12s}")
table = {}
for name, start, calage, source in SERIES:
    truth_bias = BIAS[calage]
    try:
        p_e, q_e, bias = (from_ko(*source) if isinstance(source, tuple)
                          else from_csv(source, t))
    except (FileNotFoundError, OSError) as problem:
        print(f'{name:20s} {start:16s}   -- absent ({type(problem).__name__})')
        continue
    translation, yaw, tilt = errors(p_e, q_e, p_t, q_t, first, last)
    error = np.abs(truth_bias - bias)[first]
    table[(name, start)] = (translation.mean() * 1000, yaw.mean(), tilt.mean())
    print(f'{name:20s} {start:16s} {translation.mean()*1000:10.2f} {yaw.mean():10.4f} '
          f'{tilt.mean():9.4f} {np.degrees(error.mean()):13.5f} '
          f'{100*np.mean(bias[first])/np.mean(truth_bias[first]):11.1f}%')

print()
for name in ('RI-EKF papier', 'RI-EKF adapte', 'Kinetics Observer'):
    for start in ('froid, echelon', 'froid, rampe'):
        if (name, 'chaud') in table and (name, start) in table:
            warm, cold = table[(name, 'chaud')], table[(name, start)]
            print(f'{name:20s} {start:16s}/chaud  transl x{cold[0]/warm[0]:.3f}  '
                  f'lacet x{cold[1]/warm[1]:.3f}  tilt x{cold[2]/warm[2]:.3f}')
