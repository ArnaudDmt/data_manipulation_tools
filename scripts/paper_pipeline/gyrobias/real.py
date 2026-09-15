#!/usr/bin/env python3
"""Modele realiste d'un gyro MEMS de milieu de gamme, injecte dans HartleyInput.txt.

Classe ADIS16445 / STIM-lite, +-250 deg/s sur 16 bits. Quatre composantes, dans l'ordre ou le
capteur les produit :

    1. ARW      bruit blanc GAUSSIEN, sigma = 1.2e-3 rad/s par echantillon a 200 Hz
                (0.3 deg/sqrt(h)) -- somme de tres nombreuses contributions, donc gaussien par
                le theoreme central limite. C'est la composante dominante.
    2. RW       marche aleatoire du biais, 3e-7 rad/s par pas (instabilite ~5 deg/h)
    3. biais    residu d'etalonnage. HRP5-P mesure 0.0003 deg/s sur l'axe de lacet, donc il est
                effectivement etalonne ; on prend 0.005 deg/s, un residu prudent.
    4. quant    QUANTIFICATION, uniforme par construction : LSB = 500/65536 deg/s = 1.33e-4 rad/s,
                appliquee en dernier sur la mesure totale. Ecart-type LSB/sqrt(12) = 3.8e-5, soit
                31 fois moins que l'ARW -- d'ou l'interet de la tester seule.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
HARTLEY = Path.home() / 'Documents/HartleyIEKF_WithPlots'
HERE = Path(__file__).resolve().parent
ARW = 1.2e-3                     # rad/s, ecart-type par echantillon
RW = 3.0e-7                      # rad/s par pas
BIAS = np.radians(0.005)         # residu d'etalonnage, rad/s
LSB = np.radians(500.0 / 65536)  # rad/s

MODELS = {
    'clean': (),
    'arw': ('arw',),
    'quant': ('quant',),
    'arw_quant': ('arw', 'quant'),
    'realiste': ('arw', 'rw', 'bias', 'quant'),
}


def build(project, tag, seed=12345):
    parts = MODELS[tag]
    source = ROOT / f'Projects/{project}/output_data/kinetics_eval/HartleyInput.txt'
    lines = source.read_text().splitlines()
    imu = [i for i, l in enumerate(lines) if l.startswith('IMU ')]
    n = len(imu)
    rng = np.random.default_rng(seed)
    add = np.zeros((n, 3))
    if 'arw' in parts:
        add += rng.normal(0.0, ARW, (n, 3))
    if 'rw' in parts:
        add += np.cumsum(rng.normal(0.0, RW, (n, 3)), axis=0)
    if 'bias' in parts:
        add += BIAS * np.array([1.0, -0.8, 1.1])
    raw = np.array([[float(x) for x in lines[i].split(' ')[2:5]] for i in imu]) + add
    if 'quant' in parts:
        raw = np.round(raw / LSB) * LSB
    for k, i in enumerate(imu):
        f = lines[i].split(' ')
        f[2:5] = [f'{v:.8f}' for v in raw[k]]
        lines[i] = ' '.join(f)
    (HARTLEY / 'data/HartleyInput.txt').write_text('\n'.join(lines) + '\n')
    return n, add.std(), np.abs(add).max()


def parse(project, robot, tag):
    out = HERE / f'{project}-{tag}.csv'
    if out.exists():
        return out
    n, std, peak = build(project, tag)
    print(f'  {tag:10s} ecart-type ajoute {std:.3e}  pic {peak:.3e} rad/s', flush=True)
    (HARTLEY / 'data/HartleyOutput.csv').unlink(missing_ok=True)
    code = subprocess.run(['./InEkfLogParser'], cwd=HARTLEY / 'bin',
                          env={**os.environ, 'HARTLEY_ROBOT': robot}, capture_output=True).returncode
    produced = HARTLEY / 'data/HartleyOutput.csv'
    if code or not produced.exists() or produced.stat().st_size == 0:
        print(f'  ABANDON {tag}', flush=True)
        return None
    produced.replace(out)
    return out


if __name__ == '__main__':
    project = sys.argv[1] if len(sys.argv) > 1 else 'HRP5_MultiContact_1'
    robot = 'hrp5_p' if project.startswith('HRP5') else 'rhps1'
    print(f'=== {project} ({robot})  LSB = {LSB:.3e} rad/s, quantification sigma = {LSB/np.sqrt(12):.3e}')
    for tag in MODELS:
        parse(project, robot, tag)
