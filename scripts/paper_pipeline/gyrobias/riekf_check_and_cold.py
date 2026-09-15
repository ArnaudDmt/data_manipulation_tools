#!/usr/bin/env python3
"""RI-EKF : (1) relance sur les 13 datasets et comparaison aux sorties du papier,
(2) redemarrage a froid au premier instant de la mocap sur les deux datasets dont la fenetre notee
ne commence pas au debut du log (LongWalk +1002 s, RHPS1_5 +207 s).

Aucune derive injectee. Configuration par defaut du parseur, comme rebuild_hartley.sh.
Le fichier d'entree du parseur est un LIEN vers l'entree du cache, et il est RETIRE a la fin :
rebuild_hartley.sh fait `cp /tmp/HartleyInput.txt data/HartleyInput.txt`, qui ecrirait a travers
un lien laisse en place et ecraserait le fichier du projet.
"""
import filecmp
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
HARTLEY = Path.home() / 'Documents/HartleyIEKF_WithPlots'
OUT = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/'
           '287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/riekfcheck')
PROJECTS = ['HRP5_MultiContact_1', 'HRP5_MultiContact_2', 'HRP5_MultiContact_3', 'HRP5_MultiContact_4',
            'HRP5P_LongWalk', 'KO_TRO2024_RHPS1_1', 'KO_TRO2024_RHPS1_2', 'KO_TRO2024_RHPS1_3',
            'KO_TRO2024_RHPS1_4', 'KO_TRO2024_RHPS1_5', 'KO_TRO_2024_RHPS1_SLIPPAGE_1',
            'KO_TRO_2024_RHPS1_SLIPPAGE_2', 'KO_TRO_2024_RHPS1_SLIPPAGE_3']
COLD = ['HRP5P_LongWalk', 'KO_TRO2024_RHPS1_5']
POSE = [f'IMU_Orientation_{a}' for a in 'xyzw'] + [f'IMU_Position_{a}' for a in 'xyz']


def parse(project, source, destination):
    link = HARTLEY / 'data/HartleyInput.txt'
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(source)
    produced = HARTLEY / 'data/HartleyOutput.csv'
    produced.unlink(missing_ok=True)
    env = {**os.environ, 'HARTLEY_ROBOT': 'hrp5_p' if project.startswith('HRP5') else 'rhps1'}
    env.pop('HARTLEY_CONFIG', None)
    result = subprocess.run(['./InEkfLogParser'], cwd=HARTLEY / 'bin', env=env, capture_output=True)
    if result.returncode or not produced.exists() or produced.stat().st_size == 0:
        print(f'{project}: ECHEC du parseur (code {result.returncode})', flush=True)
        return None
    shutil.move(produced, destination)
    return destination


def compare(project, produced, reference):
    if filecmp.cmp(produced, reference, shallow=False):
        return 'IDENTIQUE octet a octet'
    a = pd.read_csv(produced, sep=';')
    b = pd.read_csv(reference, sep=';')
    if len(a) != len(b):
        return f'DIFFERENT : {len(a)} lignes contre {len(b)}'
    diff = (a.select_dtypes('number') - b.select_dtypes('number')).abs().max()
    return 'DIFFERENT : ecart max ' + ', '.join(f'{k}={v:.3g}' for k, v in diff.nlargest(3).items())


def cold_input(project):
    cache = ROOT / f'Projects/{project}/output_data/kinetics_eval'
    offset = json.loads((cache / 'time_offset.json').read_text())['offset']
    warm = pd.read_csv(ROOT / f'results/paper-rebuild/hartley/{project}-HartleyOutput.csv', sep=';',
                       usecols=['t'] + POSE)
    row = warm.iloc[(warm['t'] - offset).abs().idxmin()]
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
            if stamp < offset:
                dropped += 1
                continue
            dst.write(line)
            kept += 1
    print(f'{project}: froid a t_log={offset:.3f} (pose prise a {row["t"]:.3f}), '
          f'{dropped} lignes jetees, {kept} gardees', flush=True)
    return destination


def main():
    print('=== RI-EKF : relance et comparaison aux sorties du papier ===', flush=True)
    for project in PROJECTS:
        produced = parse(project, ROOT / f'Projects/{project}/output_data/kinetics_eval/HartleyInput.txt',
                         OUT / f'{project}-warm.csv')
        if produced:
            print(f'{project:30s} {compare(project, produced, ROOT / f"results/paper-rebuild/hartley/{project}-HartleyOutput.csv")}',
                  flush=True)
    print('=== RI-EKF : redemarrage a froid ===', flush=True)
    for project in COLD:
        parse(project, cold_input(project), OUT / f'{project}-cold.csv')
    link = HARTLEY / 'data/HartleyInput.txt'
    if link.is_symlink():
        link.unlink()
    print('RIEKF_DONE', flush=True)


if __name__ == '__main__':
    main()
