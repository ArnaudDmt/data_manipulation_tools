#!/usr/bin/env python3
"""KO rejoue, fenetre recalee sur la routine : retrouve-t-on les chiffres du papier ?

Usage : verify_offset_ko.py <run> <projet> <correction en s> [<correction> ...]
On ne rejoue rien : les horodatages de kinetics.txt valent t_log - decalage + iterations sautees,
donc augmenter le decalage de c revient a retrancher c a chaque horodatage. Une correction nulle
doit redonner le replay tel quel -- c'est le controle.
"""
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
sys.path.insert(0, str(ROOT / 'scripts'))
import kinetics_eval as ke  # noqa: E402

OUT = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/'
           '287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/offsetcheck')
KEYS = ('rel_trans_perc', 'rel_yaw', 'rel_tilt')


def means(path):
    data = pickle.load(open(path, 'rb'))
    return {d: [float(np.mean(np.abs(v[k]))) for k in KEYS] for d, v in data.items()}


def show(label, result):
    for distance, values in result.items():
        print(f'  {label:34s} {distance:5g} m   trans % {values[0]:.4f}   lacet {values[1]:.4f}   tilt {values[2]:.4f}', flush=True)


def main():
    run, project = sys.argv[1], sys.argv[2]
    source = ROOT / f'results/{run}/{project}/kinetics.txt'
    reference = ROOT / f'Projects/{project}/output_data/kinetics_eval/reference/mocap.txt'
    lengths = ke.sublengths(ROOT / f'Projects/{project}')
    data = np.loadtxt(source, comments='#')
    OUT.mkdir(parents=True, exist_ok=True)
    print(f'== {project}', flush=True)
    for correction in map(float, sys.argv[3:]):
        shifted = data.copy()
        shifted[:, 0] -= correction
        trajectory = OUT / f'{project}-ko-{correction:+.3f}.txt'
        np.savetxt(trajectory, shifted, fmt='%.17g', header='timestamp tx ty tz qx qy qz qw')
        evaluation = OUT / f'eval-{project}-ko-{correction:+.3f}'
        shutil.rmtree(evaluation, ignore_errors=True)
        ke.analyze(trajectory, reference, evaluation, lengths)
        show(f'KO replay, correction {correction*1000:+.0f} ms',
             means(evaluation / 'saved_results/traj_est/cached/cached_rel_err.pickle'))
    show('KO pickle du papier', means(ROOT / f'results/paper-rebuild/runs/clean/{project}/cached_rel_err.pickle'))
    routine = ROOT / f'Projects/{project}/output_data/evals/KO/saved_results/traj_est/cached/cached_rel_err.pickle'
    if routine.exists():
        show('KO pickle de la routine actuelle', means(routine))


if __name__ == '__main__':
    main()
