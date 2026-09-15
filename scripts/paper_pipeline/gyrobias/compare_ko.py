#!/usr/bin/env python3
"""KO rejoue contre ses references : replay de reference du 14/09 et pickle du papier (routine).

Usage : compare_ko.py <motif du dossier de run> [projets...]
Moyennes des valeurs absolues, par sous-longueur. Replay et routine sont connus pour differer un peu
(89 mm cumules sur LongWalk, fenetre de RHPS1_5 50 ms plus tot cote replay) : l'egalite attendue est
d'abord avec var-clean-ref, le replay precedent.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
KEYS = ('rel_trans_perc', 'rel_yaw', 'rel_tilt')


def means(path):
    if not path.exists():
        return None
    data = pickle.load(open(path, 'rb'))
    return {d: [float(np.mean(np.abs(v[k]))) for k in KEYS] for d, v in data.items()}


def main():
    run = sorted(ROOT.glob(f'results/{sys.argv[1]}'))[-1]
    projects = sys.argv[2:] or sorted(p.name for p in run.iterdir() if p.is_dir())
    print(f'run : {run.name}\n')
    print(f"{'projet':30s} {'source':16s} {'dist':>5s} {'trans %':>9s} {'lacet':>8s} {'tilt':>8s}")
    for project in projects:
        sources = [('rejoue', run / project / 'eval/saved_results/traj_est/cached/cached_rel_err.pickle'),
                   ('var-clean-ref', ROOT / f'results/var-clean-ref-f29e92bec0/{project}/eval/saved_results/traj_est/cached/cached_rel_err.pickle'),
                   ('papier (routine)', ROOT / f'results/paper-rebuild/runs/clean/{project}/cached_rel_err.pickle')]
        table = {name: means(path) for name, path in sources}
        for name, result in table.items():
            if result is None:
                print(f'{project:30s} {name:16s}   absent')
                continue
            for distance, values in result.items():
                print(f'{project:30s} {name:16s} {distance:5g} ' + ' '.join(f'{v:9.4f}' if i == 0 else f'{v:8.4f}' for i, v in enumerate(values)))
        new, ref = table['rejoue'], table['var-clean-ref']
        if new and ref:
            gaps = [abs(a / b - 1) * 100 for d in new for a, b in zip(new[d], ref.get(d, new[d]))]
            print(f'{"":30s} {"ecart max rejoue / var-clean-ref":>40s} {max(gaps):.4f} %')
        print()


if __name__ == '__main__':
    main()
