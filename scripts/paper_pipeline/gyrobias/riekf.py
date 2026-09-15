#!/usr/bin/env python3
"""Injecte la derive de biais dans HartleyInput.txt et relance le parseur autonome.

Le fichier de LongWalk fait 2 Go, donc il est traite en flux : une ligne lue, une ligne ecrite.
Seules les colonnes wx wy wz des lignes IMU sont modifiees ; l'accelerometre n'est pas touche.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
HARTLEY = Path.home() / 'Documents/HartleyIEKF_WithPlots'
HERE = Path(__file__).resolve().parent


def inject(project, destination):
    source = ROOT / f'Projects/{project}/output_data/kinetics_eval/HartleyInput.txt'
    grid_t, grid_b = model.trajectory()
    n = 0
    first = last = None
    with source.open() as src, destination.open('w') as dst:
        for line in src:
            if line.startswith('IMU '):
                f = line.rstrip('\n').split(' ')
                t = float(f[1])
                for axis in range(3):
                    f[2 + axis] = f'{float(f[2 + axis]) + np.interp(t, grid_t, grid_b[:, axis]):.8f}'
                dst.write(' '.join(f) + '\n')
                n += 1
                if first is None:
                    first = t
                last = t
            else:
                dst.write(line)
    return n, first, last


def main():
    project = sys.argv[1] if len(sys.argv) > 1 else 'HRP5P_LongWalk'
    robot = 'hrp5_p' if project.startswith('HRP5') else 'rhps1'
    out = HERE / f'{project}-biased.csv'
    if out.exists():
        print(f'{out.name} deja produit'); return
    print(f'injection dans {project} ...', flush=True)
    n, first, last = inject(project, HARTLEY / 'data/HartleyInput.txt')
    print(f'  {n} lignes IMU, t de {first:.1f} a {last:.1f} s', flush=True)
    (HARTLEY / 'data/HartleyOutput.csv').unlink(missing_ok=True)
    code = subprocess.run(['./InEkfLogParser'], cwd=HARTLEY / 'bin',
                          env={**os.environ, 'HARTLEY_ROBOT': robot}, capture_output=True).returncode
    produced = HARTLEY / 'data/HartleyOutput.csv'
    if code or not produced.exists() or produced.stat().st_size == 0:
        sys.exit('ABANDON: le parseur a echoue')
    produced.replace(out)
    print(f'  ecrit {out}', flush=True)


if __name__ == '__main__':
    main()
