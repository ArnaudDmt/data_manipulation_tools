#!/usr/bin/env python3
"""Redemarre le RI-EKF au premier instant de la mocap, a froid.

Les deux estimateurs tournent aujourd'hui sur les 2948 s du log alors que seules les 1945.9
dernieres sont notees : ils entrent dans la fenetre avec 17 minutes de station debout derriere eux.
Ce script coupe cet avantage en tronquant l'entree et en reecrivant la ligne InitState.

L'etat de depart est FROID et symetrique de celui du Kinetics Observer :
  - pose : celle de l'IMU au premier instant note, prise dans la course chaude (elle coincide avec
    la mocap a 1.5 mm et 0.05 deg pres, le pipeline alignant deja les trois a cet instant) ;
  - vitesse, biais gyro et biais accelerometre : zero -- c'est ce que kinematics.cpp:185-187 pose
    de toute facon, le parseur ne lit que l'orientation et la position de la ligne InitState ;
  - covariance : celle de la configuration, inchangee.

Rien n'est ecrit hors du scratchpad et du repertoire de travail du parseur.
"""
import sys
from pathlib import Path

import pandas as pd

SCRATCH = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/'
               '287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/bias')
ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
START = 1002.276        # premier instant de la mocap, dans l'horloge du log brut


def pose_at(csv, instant):
    """Pose de l'IMU a l'instant demande, lue dans la sortie de la course chaude."""
    columns = ['t'] + [f'IMU_Orientation_{a}' for a in 'xyzw'] + [f'IMU_Position_{a}' for a in 'xyz']
    frame = pd.read_csv(csv, sep=';', usecols=columns)
    row = frame.iloc[(frame['t'] - instant).abs().idxmin()]
    return row['t'], [row[c] for c in columns[1:]]


def truncate(source, destination, instant, pose):
    kept = dropped = 0
    with source.open() as src, destination.open('w') as dst:
        dst.write('InitState ' + ' '.join(f'{v:.8f}' for v in pose) + '\n')
        for line in src:
            if line.startswith('InitState'):
                continue
            fields = line.split(' ', 2)
            if len(fields) < 2:
                continue
            if float(fields[1]) < instant:
                dropped += 1
                continue
            dst.write(line)
            kept += 1
    return kept, dropped


if __name__ == '__main__':
    source = SCRATCH / 'HartleyInput-biased-full.txt'
    if not source.exists():
        sys.exit(f'ABANDON: {source} absent -- il faut la copie complete et biaisee de HartleyInput')
    found, pose = pose_at(SCRATCH / 'HRP5P_LongWalk-biased.csv', START)
    print(f'pose de depart prise a t={found:.6f} s (cible {START})')
    print('  quaternion ' + ' '.join(f'{v:+.6f}' for v in pose[:4]))
    print('  position   ' + ' '.join(f'{v:+.6f}' for v in pose[4:]))
    destination = SCRATCH / 'HartleyInput-coldstart.txt'
    kept, dropped = truncate(source, destination, START, pose)
    print(f'{kept} lignes gardees, {dropped} jetees ({100*dropped/(kept+dropped):.1f}%)')
    print(f'ecrit {destination} ({destination.stat().st_size/1e9:.2f} Go)')
