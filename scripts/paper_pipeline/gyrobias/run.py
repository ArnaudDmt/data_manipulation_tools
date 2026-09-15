#!/usr/bin/env python3
"""Bruit UNIFORME sur le gyro du RI-EKF, sans re-tick.

Le parseur autonome lit `HartleyInput.txt`, dont les lignes IMU sont "IMU t wx wy wz ax ay az".
On y ajoute le bruit directement, puis on relance `InEkfLogParser` : 0.4 s par passe, et l'
accelerometre n'est pas touche.

Le bruit uniforme U(-a, a) est cale a la MEME VARIANCE que la gaussienne du plugin NoisySensors
(sigma = 1.2e-3 rad/s), donc a = sigma*sqrt(3) = 2.078e-3. Les deux realisations partagent la
meme graine, donc la comparaison porte sur la FORME de la loi, pas sur le tirage.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
HARTLEY = Path.home() / 'Documents/HartleyIEKF_WithPlots'
HERE = Path(__file__).resolve().parent
SIGMA = 1.2e-3          # rad/s, l'ecart-type du plugin
SEED = 12345


def noisy(source, destination, kind, sigma=SIGMA, seed=SEED):
    rng = np.random.default_rng(seed)
    lines = source.read_text().splitlines()
    imu = [i for i, l in enumerate(lines) if l.startswith('IMU ')]
    if kind == 'gauss':
        draw = rng.normal(0.0, sigma, (len(imu), 3))
    elif kind == 'uniform':
        half = sigma * np.sqrt(3.0)          # meme variance que la gaussienne
        draw = rng.uniform(-half, half, (len(imu), 3))
    elif kind == 'clean':
        draw = np.zeros((len(imu), 3))
    else:
        sys.exit(f'bruit inconnu : {kind}')
    for k, i in enumerate(imu):
        f = lines[i].split(' ')
        for axis in range(3):
            f[2 + axis] = f'{float(f[2 + axis]) + draw[k, axis]:.6f}'
        lines[i] = ' '.join(f)
    destination.write_text('\n'.join(lines) + '\n')
    return len(imu), draw.std(), np.abs(draw).max()


def parse(project, robot, tag):
    out = HERE / f'{project}-{tag}.csv'
    if out.exists():
        return out
    source = ROOT / f'Projects/{project}/output_data/kinetics_eval/HartleyInput.txt'
    n, std, peak = noisy(source, HARTLEY / 'data/HartleyInput.txt', tag)
    print(f'  {tag:8s} {n} lignes IMU, bruit ecart-type {std:.3e} pic {peak:.3e} rad/s', flush=True)
    (HARTLEY / 'data/HartleyOutput.csv').unlink(missing_ok=True)
    code = subprocess.run(['./InEkfLogParser'], cwd=HARTLEY / 'bin',
                          env={**__import__('os').environ, 'HARTLEY_ROBOT': robot},
                          capture_output=True).returncode
    produced = HARTLEY / 'data/HartleyOutput.csv'
    if code or not produced.exists() or produced.stat().st_size == 0:
        print(f'  ABANDON {tag}', flush=True)
        return None
    produced.replace(out)
    return out


if __name__ == '__main__':
    project = sys.argv[1] if len(sys.argv) > 1 else 'HRP5_MultiContact_1'
    robot = 'hrp5_p' if project.startswith('HRP5') else 'rhps1'
    print(f'=== {project} ({robot})')
    for tag in ('clean', 'gauss', 'uniform'):
        parse(project, robot, tag)
