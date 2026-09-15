#!/usr/bin/env python3
"""Score les meilleurs points de la recherche avec la metrique du papier, pas le J du tuner.

Le J agrege 42 comparaisons ponderees avec des barrieres ; le papier rapporte la moyenne des
valeurs absolues poolee par categorie. Les deux ne classent pas pareil, donc un point ne vaut rien
tant qu'il n'a pas ete passe par la pipeline comme les candidates de l'etape 3.

Reconstruit l'overlay a partir des parametres de l'essai, RABATTUS sur les barreaux (le tuner
stocke la valeur continue proposee, pas celle qu'il a evaluee).
"""
import importlib.util
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import yaml

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
G = ROOT / 'results/paper-rebuild/grid-scripts'
OUT = ROOT / 'results/paper-rebuild/runs/promoted'
SLIP = [f'KO_TRO_2024_RHPS1_SLIPPAGE_{i}' for i in (1, 2, 3)]
MULTI = [f'HRP5_MultiContact_{i}' for i in (1, 2, 3, 4)]
WALK = [f'KO_TRO2024_RHPS1_{i}' for i in (1, 2, 3, 4, 5)]
M = ("rel_trans_x_y_norm", "rel_trans_z", "rel_tilt", "rel_yaw")
TOP = 6

optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.argv = ['ys']
spec = importlib.util.spec_from_file_location('ys', G / 'yawsearch.py')
ys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ys)
ys.kt.ACTIVE_GRID = ys.kt.GRID
sys.path.insert(0, str(ROOT / 'scripts/paper_pipeline'))
from make_overlays import retained_covariances

PLACE = {name: (field, indices) for name, field, indices, *_ in ys.kt.SPACE}


def overlay_for(trial):
    cov = retained_covariances()
    changed = {}
    for name, exponent in trial.params.items():
        if name not in PLACE:
            continue
        field, indices = PLACE[name]
        value = ys.kt.snap(name, 10.0 ** exponent)
        for index in indices:
            cov[field][index] = value
        changed[name] = value
    return {'covariances': {k: [float(x) for x in v] for k, v in sorted(cov.items())}}, changed


def pooled(directory, projects):
    out = {}
    for metric in M:
        values = []
        for project in projects:
            cache = directory / project / 'cached_rel_err.pickle'
            if not cache.exists():
                return None
            data = pickle.load(cache.open('rb'))
            values.append(np.abs(np.asarray(data[sorted(data)[0]][metric], float)))
        out[metric] = float(np.concatenate(values).mean())
    return out


def score(tag, overlay, projects):
    store = OUT / tag
    if all((store / p / 'cached_rel_err.pickle').exists() for p in projects):
        return pooled(store, projects)
    path = OUT / f'{tag}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(overlay, sort_keys=True))
    label = f'prom{tag}t{int(time.time())}'
    code = subprocess.run(
        [str(ROOT / '.venv/bin/python'), 'scripts/kinetics_eval.py', '--projects', ','.join(projects),
         '--covariance-overlay', str(path), 'run', '--label', label,
         '--no-plots', '--no-latest', '--no-open'],
        cwd=ROOT, capture_output=True).returncode
    produced = sorted(ROOT.glob(f'results/{label}-*'))
    if code or not produced:
        print(f'  ABANDON {tag}', flush=True)
        return None
    for project in projects:
        cache = produced[0] / project / 'eval/saved_results/traj_est/cached/cached_rel_err.pickle'
        (store / project).mkdir(parents=True, exist_ok=True)
        if cache.exists():
            shutil.copy(cache, store / project / 'cached_rel_err.pickle')
    shutil.rmtree(produced[0], ignore_errors=True)
    return pooled(store, projects)


def main():
    study = optuna.load_study(study_name='screen',
                              storage=f"sqlite:///{ROOT}/results/kinetics-retuning/study.db")
    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value < 49]
    top = sorted(done, key=lambda t: t.value)[:TOP]
    runs = ROOT / 'results/paper-rebuild/runs/clean'

    for group, projects in (('GLISSEMENTS', SLIP), ('MULTICONTACTS', MULTI), ('MARCHES RHPS1', WALK)):
        reference = pooled(runs, projects)
        print(f'\n=== {group} : {" ".join(f"{m.replace("rel_","")} {reference[m]:.5f}" for m in M)}',
              flush=True)
        for trial in top:
            overlay, changed = overlay_for(trial)
            got = score(f'{trial.number}', overlay, projects)
            if got is None:
                continue
            print(f'  essai {trial.number:4d} J={trial.value:+7.3f}  ' +
                  '  '.join(f'{m.replace("rel_",""):14s} {100*(got[m]/reference[m]-1):+7.2f}%' for m in M),
                  flush=True)
    print('\n=== ce que change le meilleur, par rapport au reglage publie', flush=True)
    _, changed = overlay_for(top[0])
    reference_cov = retained_covariances()
    for name in sorted(changed):
        field, indices = PLACE[name]
        was = reference_cov[field][indices[0]]
        if abs(changed[name] / max(was, 1e-300) - 1) > 1e-9:
            print(f'  {name:34s} {was:<10.4g} -> {changed[name]:.4g}', flush=True)


if __name__ == '__main__':
    main()
