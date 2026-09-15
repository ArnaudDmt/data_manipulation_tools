"""Etape 2 : criblage des nouveaux reglages autour du meilleur de l'etape 1, puis croisement cible.

Un croisement complet des sept facteurs vaut 6075 combinaisons, soit 76 h. On procede donc en deux
temps : on fait varier chaque nouveau reglage SEUL autour du meilleur point de l'etape 1 (11
essais, ~9 min), puis on ne croise que ceux qui ont bouge quelque chose, sous plafond de budget.
"""
import itertools, pickle, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

SP = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/grid')
ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
OUT = ROOT / 'results/paper-rebuild/runs/grid2'
PROJECTS = [f'KO_TRO_2024_RHPS1_SLIPPAGE_{i}' for i in (1, 2, 3)]
M = ("rel_trans_x_y_norm", "rel_trans_z", "rel_tilt", "rel_yaw")
REF = dict(av='1e-10', oy='0.0001', px='1e-05', oi='0.0002', gb='1e-18', cf='900', fz='900', ct='225', cz='225',
           uf='0.09', un='0.09', ut='0.09', uz='0.09')
# Nouveaux niveaux demandes. oy descend plus bas pour voir si le serrage sature.
SCREEN = {'oy': ['1e-8', '1e-10', '1e-12'], 'gb': ['1e-16', '1e-17', '1e-19', '1e-20'],
          # Force de contact : x,y (tangentielle, le frottement) et z (normale) separement.
          'cf': ['400', '1600'], 'fz': ['400', '1600'],
          # Couple de contact : x,y ensemble, et z (le couple de LACET) separement --
          # c'est le canal qui porte le deficit de lacet de RHPS1.
          'ct': ['100', '900'], 'cz': ['100', '900'],
          # Force et couple du wrench non modelise, independamment. Le couple a deja ete
          # mesure cette nuit a 1e-2 et 1e-4 : degradation monotone. La force ne l'a jamais ete.
          # Wrench non modelise, quatre blocs independants et uniquement vers le bas :
          # force tangentielle, force normale, couple roulis-tangage, couple lacet.
          'uf': ['9e-3', '9e-4'], 'un': ['9e-3', '9e-4'],
          'ut': ['9e-3', '9e-4'], 'uz': ['9e-3', '9e-4']}
BUDGET = 220          # cellules max pour le croisement, ~2 h 40 a 44 s piece
MOVED = 0.003         # 0.3 % sur une metrique suffit a declarer un facteur actif


def pooled(directory):
    out = {}
    for metric in M:
        values = []
        for project in PROJECTS:
            cache = directory / project / 'cached_rel_err.pickle'
            if not cache.exists():
                return None
            values.append(np.abs(np.asarray(pickle.load(cache.open('rb'))[1.0][metric], float)))
        out[metric] = float(np.concatenate(values).mean())
    return out


def tag_of(settings):
    return '_'.join(f'{k}{settings[k]}' for k in sorted(settings))


def run(settings, counter=[0]):
    """Une cellule : overlay, replay sur les trois glissements, recopie du pickle."""
    tag = tag_of(settings)
    store = OUT / tag
    if all((store / p / 'cached_rel_err.pickle').exists() for p in PROJECTS):
        return pooled(store)
    counter[0] += 1
    label = f'g2c{counter[0]}_{int(time.time())}'
    overlay = SP / 'overlays2' / f'{tag}.yaml'
    overlay.parent.mkdir(parents=True, exist_ok=True)
    args = [str(overlay)] + [f'{k}={v}' for k, v in settings.items() if v != REF[k]]
    if subprocess.run([str(ROOT / '.venv/bin/python'), str(SP / 'overlay2.py')] + args,
                      capture_output=True).returncode:
        print(f'  ABANDON overlay {tag}', flush=True)
        return None
    log = SP / 'logs2' / f'{tag}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        code = subprocess.run(
            [str(ROOT / '.venv/bin/python'), 'scripts/kinetics_eval.py', '--projects', ','.join(PROJECTS),
             '--covariance-overlay', str(overlay), 'run', '--label', label,
             '--no-plots', '--no-latest', '--no-open'],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
    if code:
        print(f'  ABANDON run {tag}, voir {log}', flush=True)
        return None
    produced = sorted(ROOT.glob(f'results/{label}-*'))
    if not produced:
        print(f'  ABANDON: pas de sortie pour {tag}', flush=True)
        return None
    for project in PROJECTS:
        cache = produced[0] / project / 'eval/saved_results/traj_est/cached/cached_rel_err.pickle'
        (store / project).mkdir(parents=True, exist_ok=True)
        if cache.exists():
            shutil.copy(cache, store / project / 'cached_rel_err.pickle')
        else:
            print(f'  MANQUE {project} pour {tag}', flush=True)
    shutil.rmtree(produced[0], ignore_errors=True)
    log.unlink(missing_ok=True)
    return pooled(store)


def best_of_stage1():
    """Meilleur point de l'etape 1 : la translation la plus basse parmi ceux qui ne regressent
    sur aucune metrique de plus de 0.5 %. A defaut, la translation la plus basse tout court."""
    grid = ROOT / 'results/paper-rebuild/runs/grid'
    rows = {}
    for directory in sorted(grid.glob('av*')):
        m = re.fullmatch(r'av(.+?)_oy(.+?)_px(.+?)_oi(.+)', directory.name)
        value = pooled(directory)
        if m and value:
            rows[dict(zip(('av', 'oy', 'px', 'oi'), m.groups()))['av'], m.group(2), m.group(3), m.group(4)] = value
    if not rows:
        return dict(REF), None
    reference = rows.get((REF['av'], REF['oy'], REF['px'], REF['oi']))
    safe = [k for k, o in rows.items()
            if reference and all(o[x] <= reference[x] * 1.005 for x in M)]
    # Le lacet prime : c'est la seule case du tableau ou le RI-EKF nous devance sur les glissements
    # (1.036), alors que la translation est deja a 0.611. On prend donc le meilleur lacet, puis, a
    # 0.3 % pres de ce meilleur lacet, celui qui gagne aussi le plus en translation.
    pool = safe or list(rows)
    best_yaw = min(rows[k]['rel_yaw'] for k in pool)
    band = [k for k in pool if rows[k]['rel_yaw'] <= best_yaw * 1.003]
    pick = min(band, key=lambda k: rows[k]['rel_trans_x_y_norm'])
    settings = dict(REF)
    settings.update(dict(zip(('av', 'oy', 'px', 'oi'), pick)))
    print(f'etape 1 : {len(rows)} cellules, meilleur point {pick}, '
          f'{"sous contrainte de non-regression" if safe else "SANS contrainte (aucun candidat sur)"}', flush=True)
    return settings, rows[pick]


def main():
    base, base_metrics = best_of_stage1()
    print(f'point de depart : {base}', flush=True)
    if base_metrics is None:
        base_metrics = run(base)
    if base_metrics is None:
        sys.exit('ABANDON: le point de depart ne se calcule pas')

    print('\n=== criblage, un reglage a la fois ===', flush=True)
    active, effect = {}, {}
    for key, levels in SCREEN.items():
        for level in levels:
            settings = dict(base); settings[key] = level
            o = run(settings)
            if o is None:
                continue
            delta = {m: o[m] / base_metrics[m] - 1 for m in M}
            moved = max(abs(d) for d in delta.values())
            print(f'  {key}={level:<8s} ' + '  '.join(f'{m.replace("rel_",""):14s} {100*delta[m]:+6.2f}%' for m in M)
                  + ('   ACTIF' if moved > MOVED else ''), flush=True)
            if moved > MOVED:
                active.setdefault(key, []).append(level)
                effect[key] = max(effect.get(key, 0.0), moved)

    if not active:
        print('\naucun nouveau reglage n a d effet au-dela de 0.3 % : pas de croisement.', flush=True)
        return
    print(f'\n=== croisement des facteurs actifs : '
          f'{ {k: v for k, v in active.items()} } ===', flush=True)
    # Tronquer le produit cartesien a ses N premiers elements n'explorerait qu'un coin de l'espace :
    # itertools.product ne fait varier que les derniers axes. On prefere croiser MOINS de facteurs,
    # mais a pleine resolution, en gardant les plus influents d'abord.
    ordered = sorted(active, key=lambda k: -effect[k])
    axes, dropped = {}, []
    for key in ordered:
        candidate = dict(axes); candidate[key] = [REF[key]] + active[key]
        size = 1
        for levels in candidate.values():
            size *= len(levels)
        if size <= BUDGET:
            axes = candidate
        else:
            dropped.append(key)
    if dropped:
        print(f'budget {BUDGET} : croisement limite a {sorted(axes)} '
              f'(effets {", ".join(f"{k} {100*effect[k]:.1f}%" for k in sorted(axes))}) ; '
              f'ecartes faute de place : {dropped}', flush=True)
    if not axes:
        print('meme un seul facteur ne tient pas dans le budget, ce qui est impossible.', flush=True)
        return
    combos = [dict(zip(axes, values)) for values in itertools.product(*axes.values())]
    for i, delta in enumerate(combos, 1):
        settings = dict(base); settings.update(delta)
        if i % 10 == 1:
            print(f'  [{i}/{len(combos)}] {tag_of(settings)}', flush=True)
        run(settings)
    print('ETAPE 2 TERMINEE', flush=True)


if __name__ == '__main__':
    main()
