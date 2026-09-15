#!/usr/bin/env python3
"""Test, AVANT modification de kinetics_eval.reference_time_offset, du recalage sur l'horloge corrigee.

Pour chacun des 13 datasets, calcule le debut de fenetre de deux facons :
  ancien  : intercorrelation accelerometre, log sur son horloge BRUTE (le code actuel) ;
  nouveau : meme chose, log remis sur l'horloge CORRIGEE de la routine (t + iterations sautees,
            repair_mc_rtc_skipped_iters.py:48-61).
Controles attendus :
  - ancien == time_offset.json du cache (la reimplementation est fidele) ;
  - nouveau == cible mesuree sur la trajectoire du KO : 0 pour les 11 dont la fenetre couvre le log,
    +4 ms LongWalk, +50 ms RHPS1_5 -- sans decalage d'un echantillon sur les 11.
Colonnes lues avec pandas : t et l'accelerometre seulement (le logReplay.csv de LongWalk fait 3,9 Go).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
sys.path.insert(0, str(ROOT / 'scripts'))
import kinetics_eval as ke  # noqa: E402

PROJECTS = ['HRP5_MultiContact_1', 'HRP5_MultiContact_2', 'HRP5_MultiContact_3', 'HRP5_MultiContact_4',
            'HRP5P_LongWalk', 'KO_TRO2024_RHPS1_1', 'KO_TRO2024_RHPS1_2', 'KO_TRO2024_RHPS1_3',
            'KO_TRO2024_RHPS1_4', 'KO_TRO2024_RHPS1_5', 'KO_TRO_2024_RHPS1_SLIPPAGE_1',
            'KO_TRO_2024_RHPS1_SLIPPAGE_2', 'KO_TRO_2024_RHPS1_SLIPPAGE_3']
MEASURED_LAG = {'HRP5P_LongWalk': 0.004, 'KO_TRO2024_RHPS1_5': 0.050}
ACC = [f'Accelerometer_linearAcceleration_{a}' for a in 'xyz']


def locate(log_time, log, reference_time, reference):
    """Coeur de kinetics_eval.reference_time_offset, sans les garde-fous d'affichage."""
    step = max(ke.sampling_step(log_time), ke.sampling_step(reference_time))
    log_grid, log = ke.resample_uniform(log_time, log, step)
    _, reference = ke.resample_uniform(reference_time, reference, step)
    if len(reference) > len(log):
        reference = reference[:len(log)]
    log = log - log.mean(0)
    reference = reference - reference.mean(0)
    scores = sum(correlate(log[:, k], reference[:, k], mode='valid') for k in range(3))
    shift = int(np.argmax(scores))
    return float(log_grid[shift] - reference_time[0]), len(scores), step


def main():
    print(f"{'dataset':30s} {'cache':>10s} {'ancien':>10s} {'nouveau':>10s} {'cible':>10s} "
          f"{'ecart nouveau-cible':>20s} {'positions':>9s}")
    for project in PROJECTS:
        directory = ROOT / f'Projects/{project}'
        output = directory / 'output_data'
        cached = json.loads((output / 'kinetics_eval/time_offset.json').read_text())['offset']
        log = pd.read_csv(output / 'logReplay.csv', sep=';', usecols=['t'] + ACC)
        reference = pd.read_csv(output / 'finalDataCSV.csv', sep=';', usecols=['t'] + ACC)
        log_time, log_acc = log['t'].to_numpy(), log[ACC].to_numpy()
        reference_time, reference_acc = reference['t'].to_numpy(), reference[ACC].to_numpy()
        skipped = ke.skipped_iteration_shift(directory, ke.project_timestep(directory))
        if skipped is None or len(skipped) != len(log_time):
            print(f'{project:30s} ABANDON : {0 if skipped is None else len(skipped)} lignes de perf '
                  f'pour {len(log_time)} lignes de log')
            continue
        old, _, _ = locate(log_time, log_acc, reference_time, reference_acc)
        new, positions, step = locate(log_time + skipped, log_acc, reference_time, reference_acc)
        target = cached + MEASURED_LAG.get(project, 0.0)
        print(f'{project:30s} {cached:10.4f} {old:10.4f} {new:10.4f} {target:10.4f} '
              f'{(new - target) * 1000:+17.1f} ms {positions:9d}', flush=True)


if __name__ == '__main__':
    main()
