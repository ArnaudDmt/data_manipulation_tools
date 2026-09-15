#!/usr/bin/env python3
"""Bilan : variance initiale du biais gyro 1e-8 (normal) contre 1e-12, KO et RI-EKF, par dataset et par categorie."""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools/results')
X = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/riekfcheck')
PK = 'saved_results/traj_est/cached/cached_rel_err.pickle'
CAT = {'Multicontact': ['HRP5_MultiContact_1', 'HRP5_MultiContact_2', 'HRP5_MultiContact_3', 'HRP5_MultiContact_4'],
       'Marche a plat': ['KO_TRO2024_RHPS1_1', 'KO_TRO2024_RHPS1_2', 'KO_TRO2024_RHPS1_3', 'KO_TRO2024_RHPS1_4', 'KO_TRO2024_RHPS1_5'],
       'Glissement': ['KO_TRO_2024_RHPS1_SLIPPAGE_1', 'KO_TRO_2024_RHPS1_SLIPPAGE_2', 'KO_TRO_2024_RHPS1_SLIPPAGE_3'],
       'LongWalk': ['HRP5P_LongWalk']}
K = [('rel_trans_x_y_norm', 'Transxy'), ('rel_trans_z', 'Transz'), ('rel_tilt', 'tilt'), ('rel_yaw', 'lacet')]


def run(glob):
    return sorted(ROOT.glob(glob))[-1]


def ko_normal(p):
    if p in ('HRP5P_LongWalk', 'KO_TRO2024_RHPS1_5'):
        return run('chk-offsetfix-*') / p / 'eval' / PK
    if p.startswith('KO_TRO2024_RHPS1_'):
        return run('chk-clean-rhps-*') / p / 'eval' / PK
    return run('chk-clean-rest-*') / p / 'eval' / PK


def ko_bvar(p):
    if p == 'HRP5P_LongWalk':
        return run('lw-warm-bvar-*') / p / 'eval' / PK
    if p.startswith('HRP5_MultiContact'):
        return run('all13-bvar-2e5a3683cf') / p / 'eval' / PK
    return run('all13-bvar-b-*') / p / 'eval' / PK


def ri(name, p):
    return X / f'eval-{name}-{p}' / PK


def arrays(path):
    (d, v), = pickle.load(open(path, 'rb')).items()
    return {k: np.abs(np.asarray(v[k], float)) for k, _ in K}


def show(label, a, b):
    ma = {k: a[k].mean() for k, _ in K}; mb = {k: b[k].mean() for k, _ in K}
    print(f'  {label:26s} ' + '  '.join(f'{n} {ma[k]:.4f}->{mb[k]:.4f} ({100*(mb[k]/ma[k]-1):+.1f}%)' for k, n in K))


def main():
    for estimator, normal, tight in (('KO', ko_normal, ko_bvar), ('RI-EKF', lambda p: ri('normal', p), lambda p: ri('bvar', p))):
        print(f'\n===== {estimator} : variance initiale du biais 1e-8 -> 1e-12')
        for cat, projects in CAT.items():
            pooled_n = {k: [] for k, _ in K}; pooled_t = {k: [] for k, _ in K}
            for p in projects:
                pn, pt = normal(p), tight(p)
                if not (pn.exists() and pt.exists()):
                    print(f'  {p:26s} absent ({"normal" if not pn.exists() else "resserre"})'); continue
                a, b = arrays(pn), arrays(pt)
                show(p, a, b)
                for k, _ in K:
                    pooled_n[k].append(a[k]); pooled_t[k].append(b[k])
            if pooled_n['rel_yaw']:
                show(f'>> {cat}', {k: np.concatenate(v) for k, v in pooled_n.items()}, {k: np.concatenate(v) for k, v in pooled_t.items()})


if __name__ == '__main__':
    main()
