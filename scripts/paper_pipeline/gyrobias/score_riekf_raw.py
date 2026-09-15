#!/usr/bin/env python3
"""Note rpg du RI-EKF depuis la sortie BRUTE du parseur, sans passer par la routine.

Usage : score_riekf_raw.py <projet> [--decalage=<s>] <nom>=<csv> [<nom>=<csv> ...]

Pourquoi pas la routine : elle n'a pas utilise le parse archive (Projects/<p>/output_data/
HartleyOutputCSV.csv du 14/09 23:38 differe de results/paper-rebuild/hartley/ du 15/09 00:07), et
les courses a froid n'y passent pas du tout. Normal et froid sont donc notes ICI par la meme methode.

Methode, validee sur RHPS1_5 contre formatted_Hartley_Traj.txt (0.01 mm sur la partie commune) :
  t_fenetre = t_log - decalage + iterations sautees ; base = p_imu + R_imu.apply(-pos_fb_imu), rotation identite
  (imuFbKine_position = -pos_fb_imu, imuFbKine_ori = identite) ; puis kinetics_eval.analyze(),
  c'est-a-dire rpg posyaw avec les sous-longueurs de la categorie. Moyennes, pas medianes.
"""
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
sys.path.insert(0, str(ROOT / 'scripts'))
import kinetics_eval as ke  # noqa: E402

OUT = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/'
           '287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/riekfcheck')
Q = [f'IMU_Orientation_{a}' for a in 'xyzw']
P = [f'IMU_Position_{a}' for a in 'xyz']
KEYS = ('rel_trans_x_y_norm', 'rel_trans_z', 'rel_tilt', 'rel_yaw', 'rel_trans_perc')


def means(pickle_path):
    data = pickle.load(open(pickle_path, 'rb'))
    return {d: {k: float(np.mean(np.abs(v[k]))) for k in KEYS if k in v} for d, v in data.items()}


def score(project, name, csv, correction=0.0):
    cache = ROOT / f'Projects/{project}/output_data/kinetics_eval'
    offset_info = json.loads((cache / 'time_offset.json').read_text())
    assert np.allclose(offset_info['r_imu_fb'], [0, 0, 0, 1], atol=1e-9), 'rotation IMU-base non identite'
    lever = -np.array(offset_info['pos_fb_imu'])
    frame = pd.read_csv(csv, sep=';', usecols=['t'] + Q + P)
    rotation = R.from_quat(frame[Q].to_numpy())
    position = frame[P].to_numpy() + rotation.apply(lever)
    # correction : ce qu'il faut AJOUTER au decalage du replay pour tomber sur la fenetre de la
    # routine (+0.004 s LongWalk, +0.050 s RHPS1_5, mesure sur la trajectoire du KO).
    stamps = frame['t'].to_numpy() - (offset_info['offset'] + correction)
    # Meme correction des iterations sautees que kinetics_eval.py:1137 et la routine : une ligne du
    # log vaut plusieurs periodes quand perf_GlobalRun depasse le pas du controleur.
    project_dir = ROOT / f'Projects/{project}'
    step = ke.project_timestep(project_dir)
    shift = ke.skipped_iteration_shift(project_dir, step)
    if shift is not None and len(shift):
        rows = np.rint(frame['t'].to_numpy() / step).astype(int).clip(0, len(shift) - 1)
        stamps = stamps + shift[rows]
        print(f'  {project}: correction des iterations sautees, {shift[-1]*1000:.1f} ms en fin de log', flush=True)
    trajectory = OUT / f'{name}-{project}.txt'
    with trajectory.open('w') as stream:
        stream.write('# timestamp tx ty tz qx qy qz qw\n')
        for k in np.nonzero(stamps >= 0.0)[0]:
            stream.write(' '.join(f'{v:.17g}' for v in (stamps[k], *position[k], *rotation[k].as_quat())) + '\n')
    evaluation = OUT / f'eval-{name}-{project}'
    shutil.rmtree(evaluation, ignore_errors=True)
    ke.analyze(trajectory, cache / 'reference/mocap.txt', evaluation, ke.sublengths(ROOT / f'Projects/{project}'))
    return means(evaluation / 'saved_results/traj_est/cached/cached_rel_err.pickle')


def show(label, result):
    for distance, values in result.items():
        print(f'  {label:26s} {distance:5g} m  ' + '  '.join(f'{k.removeprefix("rel_")} {v:.4f}' for k, v in values.items()))


if __name__ == '__main__':
    project = sys.argv[1]
    correction = 0.0
    for item in sys.argv[2:]:
        if item.startswith('--decalage='):
            correction = float(item.split('=', 1)[1])
            continue
        name, csv = item.split('=', 1)
        show(f'RI-EKF {name}', score(project, name, Path(csv), correction))
    paper = ROOT / f'results/paper-rebuild/runs/clean/{project}'
    show('RI-EKF, pickle du papier', means(paper / 'riekf_rel_err.pickle'))
    show('KO, pickle du papier', means(paper / 'cached_rel_err.pickle'))
