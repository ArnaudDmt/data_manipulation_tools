#!/usr/bin/env python3
"""RI-EKF, variance initiale du biais gyro resserree (1e-12), sur les entrees du cache ; puis notation rpg
normale et resserree par la meme methode (score_riekf_raw.score). LongWalk exclu a la demande d'Arnaud."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from riekf_check_and_cold import HARTLEY, OUT, ROOT  # noqa: E402
import score_riekf_raw as scorer  # noqa: E402

CONFIG = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/linear/hartley-bvar.yaml')
PROJECTS = ['HRP5_MultiContact_1', 'HRP5_MultiContact_2', 'HRP5_MultiContact_3', 'HRP5_MultiContact_4',
            'KO_TRO2024_RHPS1_1', 'KO_TRO2024_RHPS1_2', 'KO_TRO2024_RHPS1_3', 'KO_TRO2024_RHPS1_4',
            'KO_TRO2024_RHPS1_5', 'KO_TRO_2024_RHPS1_SLIPPAGE_1', 'KO_TRO_2024_RHPS1_SLIPPAGE_2',
            'KO_TRO_2024_RHPS1_SLIPPAGE_3']


def parse_with_config(project, destination):
    link = HARTLEY / 'data/HartleyInput.txt'
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(ROOT / f'Projects/{project}/output_data/kinetics_eval/HartleyInput.txt')
    produced = HARTLEY / 'data/HartleyOutput.csv'
    produced.unlink(missing_ok=True)
    env = {**os.environ, 'HARTLEY_ROBOT': 'hrp5_p' if project.startswith('HRP5') else 'rhps1',
           'HARTLEY_CONFIG': str(CONFIG)}
    code = subprocess.run(['./InEkfLogParser'], cwd=HARTLEY / 'bin', env=env, capture_output=True).returncode
    if code or not produced.exists():
        sys.exit(f'ABANDON: parseur en echec sur {project}')
    shutil.move(produced, destination)


def main():
    for project in PROJECTS:
        parse_with_config(project, OUT / f'{project}-bvar.csv')
        print(f'{project}: RI-EKF resserre ecrit', flush=True)
    link = HARTLEY / 'data/HartleyInput.txt'
    if link.is_symlink():
        link.unlink()
    for project in PROJECTS:
        scorer.score(project, 'normal', OUT / f'{project}-warm.csv')
        scorer.score(project, 'bvar', OUT / f'{project}-bvar.csv')
        print(f'{project}: note', flush=True)


if __name__ == '__main__':
    main()
