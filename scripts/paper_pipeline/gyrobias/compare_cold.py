#!/usr/bin/env python3
"""Course normale contre depart a froid au premier instant de la mocap, KO et RI-EKF.

Moyennes des valeurs absolues rpg ; lacet aussi par quart de la fenetre (les sous-trajectoires sont
rangees par instant de depart). Normal KO : chk-offsetfix ; froid KO : cold-ko ; RI-EKF : evaluations
ecrites par score_riekf_raw.py sous les noms `normal` et `froid`.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
OUT = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/'
           '287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/riekfcheck')
PICKLE = 'saved_results/traj_est/cached/cached_rel_err.pickle'


def load(path):
    data = pickle.load(open(path, 'rb'))
    (distance, values), = data.items()
    return distance, {k: np.abs(np.asarray(values[k], dtype=float)) for k in ('rel_trans_perc', 'rel_yaw', 'rel_tilt')}


def main():
    cold_run = sorted(ROOT.glob('results/cold-ko-*'))[-1]
    for project in ('HRP5P_LongWalk', 'KO_TRO2024_RHPS1_5'):
        rows = [('KO', 'normal', ROOT / f'results/chk-offsetfix-0f9072d48f/{project}/eval/{PICKLE}'),
                ('KO', 'froid', cold_run / project / f'eval/{PICKLE}'),
                ('RI-EKF', 'normal', OUT / f'eval-normal-{project}/{PICKLE}'),
                ('RI-EKF', 'froid', OUT / f'eval-froid-{project}/{PICKLE}')]
        print(f'\n== {project}')
        print(f"{'':8s} {'depart':7s} {'dist':>5s} {'trans %':>8s} {'lacet':>7s} {'tilt':>7s}   lacet par quart")
        table = {}
        for estimator, start, path in rows:
            if not path.exists():
                print(f'{estimator:8s} {start:7s}   absent : {path}')
                continue
            distance, v = load(path)
            quarters = [q.mean() for q in np.array_split(v['rel_yaw'], 4)]
            table[(estimator, start)] = (v['rel_trans_perc'].mean(), v['rel_yaw'].mean(), v['rel_tilt'].mean())
            print(f"{estimator:8s} {start:7s} {distance:5g} {table[(estimator, start)][0]:8.4f} "
                  f"{table[(estimator, start)][1]:7.4f} {table[(estimator, start)][2]:7.4f}   "
                  + ' '.join(f'{q:.3f}' for q in quarters))
        for estimator in ('KO', 'RI-EKF'):
            if (estimator, 'normal') in table and (estimator, 'froid') in table:
                n, f = table[(estimator, 'normal')], table[(estimator, 'froid')]
                print(f'{estimator:8s} froid/normal : trans x{f[0]/n[0]:.3f}  lacet x{f[1]/n[1]:.3f}  tilt x{f[2]/n[2]:.3f}')


if __name__ == '__main__':
    main()
