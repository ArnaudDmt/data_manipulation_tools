#!/usr/bin/env python3
"""RI-EKF redemarre au premier instant de la mocap, SANS derive, a partir de l'entree du cache.

Usage : coldstart_riekf_unbiased.py <projet>
Coupe et pose initiale sur l'horloge CORRIGEE (temps brut + iterations sautees), comme le decalage de
time_offset.json depuis le commit d05499f. Vitesse et biais nuls sont imposes par le parseur
(kinematics.cpp:185-187) ; seule la ligne InitState porte la pose. Sortie riekfcheck/<projet>-cold.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kinetics_eval as ke  # noqa: E402
from riekf_check_and_cold import HARTLEY, OUT, POSE, parse  # noqa: E402


def main(project):
    directory = ROOT / f'Projects/{project}'
    cache = directory / 'output_data/kinetics_eval'
    offset = json.loads((cache / 'time_offset.json').read_text())['offset']
    step = ke.project_timestep(directory)
    skipped = ke.skipped_iteration_shift(directory, step)

    def repaired(raw):
        if skipped is None:
            return raw
        return raw + skipped[np.clip(np.rint(np.asarray(raw) / step).astype(int), 0, len(skipped) - 1)]

    warm = pd.read_csv(OUT / f'{project}-warm.csv', sep=';', usecols=['t'] + POSE)
    row = warm.iloc[int(np.abs(repaired(warm['t'].to_numpy()) - offset).argmin())]
    destination = OUT / f'{project}-HartleyInput-cold.txt'
    kept = dropped = 0
    with (cache / 'HartleyInput.txt').open() as src, destination.open('w') as dst:
        dst.write('InitState ' + ' '.join(f'{row[c]:.8f}' for c in POSE) + '\n')
        for line in src:
            if line.startswith('InitState'):
                continue
            fields = line.split(' ', 2)
            try:
                stamp = float(fields[1])
            except (IndexError, ValueError):
                dst.write(line)
                continue
            if float(repaired(stamp)) < offset:
                dropped += 1
                continue
            dst.write(line)
            kept += 1
    print(f'{project}: decalage {offset:.3f} s, pose prise a t_brut={row["t"]:.3f}, '
          f'{dropped} lignes jetees, {kept} gardees', flush=True)
    parse(project, destination, OUT / f'{project}-cold.csv')
    link = HARTLEY / 'data/HartleyInput.txt'
    if link.is_symlink():
        link.unlink()
    print(f'{project}: RI-EKF froid ecrit', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
