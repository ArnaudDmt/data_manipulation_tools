#!/usr/bin/env python3
"""Entree du RI-EKF : tronquee au premier instant de la mocap, avec la rampe de biais REBASEE.

La premiere version de l'experience a froid gardait la derive calee sur le debut du log : au moment
ou les filtres redemarraient, le biais valait deja 0.0154 deg/s, soit 67 % de son asymptote. Ils
encaissaient donc un ECHELON qu'aucune centrale ne produit, et qui penalise mecaniquement celui
qui ne sait pas sauter.

Ici la rampe thermique commence au meme instant que les filtres : b(t) = model.at(t - START). C'est
le scenario physique -- l'IMU est mise a zero a l'arret, puis chauffe pendant la marche.

La ligne InitState reprend la meme pose que la premiere version, pour que les deux experiences
soient comparables ; a cet instant le biais vaut zero, donc la pose de la course non biaisee et
celle de la course biaisee ne different que par ce que le biais avait deja fait deriver, et le
repere monde de depart est de toute facon sans effet sur une erreur relative.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
SCRATCH = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/'
               '287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/bias')
START = 1002.276
SOURCE = ROOT / 'Projects/HRP5P_LongWalk/output_data/kinetics_eval/HartleyInput.txt'
DESTINATION = SCRATCH / 'HartleyInput-coldstart-rebased.txt'


def initial_pose():
    columns = [f'IMU_Orientation_{a}' for a in 'xyzw'] + [f'IMU_Position_{a}' for a in 'xyz']
    frame = pd.read_csv(SCRATCH / 'HRP5P_LongWalk-biased.csv', sep=';', usecols=['t'] + columns)
    return [frame.iloc[(frame['t'] - START).abs().idxmin()][c] for c in columns]


def main():
    pose = initial_pose()
    grid_t, grid_b = model.trajectory()
    kept = dropped = 0
    with SOURCE.open() as src, DESTINATION.open('w') as dst:
        dst.write('InitState ' + ' '.join(f'{v:.8f}' for v in pose) + '\n')
        for line in src:
            if line.startswith('InitState'):
                continue
            fields = line.rstrip('\n').split(' ')
            if len(fields) < 2:
                continue
            t = float(fields[1])
            if t < START:
                dropped += 1
                continue
            if fields[0] == 'IMU':
                for axis in range(3):
                    drift = float(np.interp(t - START, grid_t, grid_b[:, axis]))
                    fields[2 + axis] = f'{float(fields[2 + axis]) + drift:.8f}'
            dst.write(' '.join(fields) + '\n')
            kept += 1
    print(f'{kept} lignes gardees, {dropped} jetees ({100 * dropped / (kept + dropped):.1f}%)')
    check = np.degrees(model.at(np.array([0.0, 500.0, 1945.9]))[:, 2])
    print(f'biais z rebase : {check[0]:+.5f} au depart, {check[1]:+.5f} a 500 s, '
          f'{check[2]:+.5f} deg/s a la fin')
    print(f'ecrit {DESTINATION} ({DESTINATION.stat().st_size / 1e9:.2f} Go)')


if __name__ == '__main__':
    main()
