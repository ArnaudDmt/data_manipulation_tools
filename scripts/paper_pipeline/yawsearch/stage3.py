"""Etape 3 : les candidats des etapes 1-2 sont d'abord cribles sur les QUATRE multicontacts,
puis, seulement s'ils survivent, evalues sur les neuf datasets restants.

Regle du 2026-09-15 : un reglage trouve sur une categorie n'est pas un resultat tant qu'il n'a pas
ete evalue sur les treize. Mais les multicontacts servent de crible, parce que RHPS1 et HRP-5P ont
reagi en sens INVERSE a chaque parametre de contact mesure cette nuit : un gain sur les glissements
seuls ne prouve rien.
"""
import pickle, re, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

SP = Path('/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/grid')
ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
OUT = ROOT / 'results/paper-rebuild/runs/grid3'
SLIP = [f'KO_TRO_2024_RHPS1_SLIPPAGE_{i}' for i in (1, 2, 3)]
MULTI = [f'HRP5_MultiContact_{i}' for i in (1, 2, 3, 4)]
# LongWalk est volontairement EXCLU de l'enchainement automatique : 1.47 M de lignes, soit 130
# fois un multicontact, et chacune de ses passes coute plusieurs minutes. Il reste a lancer a la
# main sur les survivantes avant de valider quoi que ce soit -- c'est le dataset ou le KO gagne
# le plus en lacet, donc celui ou une regression serait la plus couteuse a manquer.
REST = [f'KO_TRO2024_RHPS1_{i}' for i in (1, 2, 3, 4, 5)]
EXCLUDED = ['HRP5P_LongWalk']
M = ("rel_trans_x_y_norm", "rel_trans_z", "rel_tilt", "rel_yaw")
# Ordre de priorite pour la selection en tour de role : le lacet d'abord. La translation est
# deja a 0.611 face au RI-EKF sur les glissements, le lacet a 1.036 -- c'est la seule case
# perdue, donc celle ou un gain vaut le plus.
PRIORITY = ("rel_yaw", "rel_trans_x_y_norm", "rel_tilt", "rel_trans_z")
REF = dict(av='1e-10', oy='0.0001', px='1e-05', oi='0.0002', gb='1e-18', cf='900', fz='900', ct='225', cz='225',
           uf='0.09', un='0.09', ut='0.09', uz='0.09')
# --- regles de rejet, du plus imperatif au plus souple -----------------------------------------
#
# Le LACET ne doit pas se degrader. C'est la seule case du tableau ou le RI-EKF nous devance sur
# les glissements (1.036) alors que la translation est deja a 0.611 : un gain de translation ne
# change aucune phrase du papier, une perte de lacet, si.
#
# 1. degrade des DEUX cotes (glissements ET multicontact) -> jetee, sans appel.
# 2. lacet multicontact au-dela de YAW_HARD -> jetee, sauf tolerance ci-dessous.
# 3. tolerance : jusqu'a YAW_TOLERATED de lacet, mais seulement pour un TRES gros gain de
#    translation en multicontact (>= BIG_MULTI).
# 4. sur les autres metriques, le seuil ordinaire DROP, avec la voie d'arbitrage.
YAW_EPS = 0.001        # en deca de 0.1 %, on considere que le lacet n'a pas bouge
YAW_HARD = 0.005       # au-dela de 0.5 % de lacet en multicontact, c'est non
YAW_TOLERATED = 0.01   # ... tolere jusqu'a 1 % si et seulement si
BIG_MULTI = 0.05       # ... le multicontact gagne au moins 5 % en translation
DROP = 0.02            # autres metriques : au-dela de 2 %, on jette
WORTH_TRANS = 0.05     # arbitrage : 5 % de gain en translation sur les glissements
WORTH_YAW = 0.02       # ou 2 % en lacet
# ... et le gain doit DEPASSER la degradation. Une configuration qui rend ce qu'elle coute n'est
# pas un progres, c'est un deplacement : elle est jetee comme les autres.
# Le crible multicontact coute ~30 s par candidate, donc on peut en passer beaucoup. Ce qui
# coute, ce sont les cinq marches RHPS1 ensuite : on y limite les survivantes.
KEEP = 14            # candidates cribles sur les multicontacts
KEEP_FULL = 8        # survivantes qui gagnent les cinq marches RHPS1


def pooled(directory, projects):
    """Moyenne des valeurs absolues poolee, a la longueur de sous-trajectoire du cache."""
    out = {}
    for metric in M:
        values = []
        for project in projects:
            cache = directory / project / 'cached_rel_err.pickle'
            if not cache.exists():
                return None
            data = pickle.load(cache.open('rb'))
            length = sorted(data)[0]
            if metric not in data[length]:
                return None
            values.append(np.abs(np.asarray(data[length][metric], float)))
        out[metric] = float(np.concatenate(values).mean())
    return out


def tag_of(settings):
    return '_'.join(f'{k}{settings[k]}' for k in sorted(settings) if settings[k] != REF[k]) or 'reference'


def run(settings, projects, counter=[0]):
    tag = tag_of(settings)
    store = OUT / tag
    if all((store / p / 'cached_rel_err.pickle').exists() for p in projects):
        return pooled(store, projects)
    counter[0] += 1
    label = f'g3c{counter[0]}t{int(time.time())}'
    overlay = SP / 'overlays3' / f'{tag}.yaml'
    overlay.parent.mkdir(parents=True, exist_ok=True)
    args = [str(overlay)] + [f'{k}={v}' for k, v in settings.items() if v != REF[k]]
    if subprocess.run([str(ROOT / '.venv/bin/python'), str(SP / 'overlay2.py')] + args,
                      capture_output=True).returncode:
        print(f'  ABANDON overlay {tag}', flush=True); return None
    log = SP / 'logs3' / f'{tag}.log'; log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        code = subprocess.run(
            [str(ROOT / '.venv/bin/python'), 'scripts/kinetics_eval.py', '--projects', ','.join(projects),
             '--covariance-overlay', str(overlay), 'run', '--label', label,
             '--no-plots', '--no-latest', '--no-open'],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
    if code:
        print(f'  ABANDON run {tag}, voir {log}', flush=True); return None
    produced = sorted(ROOT.glob(f'results/{label}-*'))
    if not produced:
        print(f'  ABANDON: pas de sortie pour {tag}', flush=True); return None
    for project in projects:
        cache = produced[0] / project / 'eval/saved_results/traj_est/cached/cached_rel_err.pickle'
        (store / project).mkdir(parents=True, exist_ok=True)
        if cache.exists():
            shutil.copy(cache, store / project / 'cached_rel_err.pickle')
        else:
            print(f'  MANQUE {project} pour {tag}', flush=True)
    shutil.rmtree(produced[0], ignore_errors=True)
    log.unlink(missing_ok=True)
    return pooled(store, projects)


def candidates():
    """Les configurations des etapes 1-2 qui ne regressent nulle part sur les glissements,
    triees par translation. La reference est incluse comme temoin."""
    runs = ROOT / 'results/paper-rebuild/runs'
    rows = {}
    for d in sorted((runs / 'grid').glob('av*')):
        m = re.fullmatch(r'av(.+?)_oy(.+?)_px(.+?)_oi(.+)', d.name)
        o = pooled(d, SLIP)
        if m and o:
            s = dict(REF); s.update(dict(zip(('av', 'oy', 'px', 'oi'), m.groups())))
            rows[tuple(sorted(s.items()))] = o
    for d in sorted((runs / 'grid2').glob('*')):
        o = pooled(d, SLIP)
        if not o:
            continue
        s = dict(REF)
        for key, value in re.findall(r'(av|oy|px|oi|gb|cf|fz|ct|cz|uf|un|ut|uz)([^_]+)', d.name):
            s[key] = value
        rows[tuple(sorted(s.items()))] = o
    ref_key = tuple(sorted(REF.items()))
    base = rows.get(ref_key)
    if base is None:
        sys.exit('ABANDON: la cellule de reference manque')
    # Le lacet ne beneficie pas de la tolerance de 0.5 % : sur les glissements il doit etre au
    # moins aussi bon que la reference, a 0.1 % pres.
    safe = [k for k in rows if k != ref_key
            and all(rows[k][m] <= base[m] * 1.005 for m in M)
            and rows[k]['rel_yaw'] <= base['rel_yaw'] * (1 + YAW_EPS)]
    # Trier par la seule translation retiendrait N quasi-jumelles -- toutes celles qui contiennent
    # le meilleur reglage de translation -- et laisserait passer une candidate qui gagne en lacet.
    # On prend donc les meilleures de CHAQUE metrique a tour de role, sans doublon.
    picks, seen = [], set()
    ranked = {m: sorted(safe, key=lambda k: rows[k][m]) for m in PRIORITY}
    while len(picks) < KEEP:
        added = False
        for metric in PRIORITY:
            for k in ranked[metric]:
                if k not in seen:
                    seen.add(k); picks.append(k); added = True
                    break
            if len(picks) >= KEEP:
                break
        if not added:
            break
    print(f'{len(safe)} configurations sans regression sur les glissements ; '
          f'{len(picks)} retenues (meilleures de chaque metrique, en alternance)', flush=True)
    return base, rows, picks


def main():
    base_slip, rows, picks = candidates()
    print(f'{len(rows)} configurations, {len(picks)} candidates retenues pour le criblage\n', flush=True)
    if not picks:
        print('aucune candidate : rien a valider.', flush=True); return

    print('=== preparation des multicontacts', flush=True)
    subprocess.run([str(ROOT / '.venv/bin/python'), 'scripts/kinetics_eval.py',
                    '--projects', ','.join(MULTI), 'prepare'], cwd=ROOT, capture_output=True)
    ref_multi = run(dict(REF), MULTI)
    if ref_multi is None:
        sys.exit('ABANDON: la reference ne se calcule pas sur les multicontacts')

    print('\n=== crible sur les quatre multicontacts (rejet au-dela de 2 % sur une metrique)', flush=True)
    survivors, arbitrate = [], []
    for key in picks:
        settings = dict(key)
        o = run(settings, MULTI)
        if o is None:
            continue
        delta = {m: o[m] / ref_multi[m] - 1 for m in M}
        dy_multi = delta['rel_yaw']
        dy_slip = rows[key]['rel_yaw'] / base_slip['rel_yaw'] - 1
        multi_trans_gain = -delta['rel_trans_x_y_norm']
        worst = max(delta[m] for m in M if m != 'rel_yaw')
        gain_trans = 1 - rows[key]['rel_trans_x_y_norm'] / base_slip['rel_trans_x_y_norm']
        gain_yaw = -dy_slip
        best_gain = max(gain_trans, gain_yaw)
        worth = ((gain_trans >= WORTH_TRANS or gain_yaw >= WORTH_YAW)
                 and best_gain > max(worst, dy_multi))

        if dy_slip > YAW_EPS and dy_multi > YAW_EPS:
            verdict = (f'REJETE: lacet degrade des deux cotes '
                       f'(glissement {100*dy_slip:+.2f}%, multicontact {100*dy_multi:+.2f}%)')
            print(f'  {tag_of(settings):<44s} ' +
                  '  '.join(f'{m.replace("rel_",""):14s} {100*delta[m]:+6.2f}%' for m in M) +
                  f'   {verdict}', flush=True)
            continue
        if dy_multi > YAW_TOLERATED or (dy_multi > YAW_HARD and multi_trans_gain < BIG_MULTI):
            verdict = (f'REJETE: lacet multicontact {100*dy_multi:+.2f}% '
                       f'(gain translation multicontact seulement {100*multi_trans_gain:+.1f}%)')
            print(f'  {tag_of(settings):<44s} ' +
                  '  '.join(f'{m.replace("rel_",""):14s} {100*delta[m]:+6.2f}%' for m in M) +
                  f'   {verdict}', flush=True)
            continue

        if worst <= DROP:
            verdict = 'garde' + (f' (lacet {100*dy_multi:+.2f}% tolere pour {100*multi_trans_gain:+.1f}%'
                                 f' de translation)' if dy_multi > YAW_HARD else '')
            survivors.append(settings)
        elif worth:
            verdict = (f'A ARBITRER (glissement: transl {100*gain_trans:+.1f}%, lacet {100*gain_yaw:+.1f}% '
                       f'> degradation {100*worst:.1f}%)')
            arbitrate.append(settings)
        else:
            verdict = ('REJETE (gain glissement au plus '
                       f'{100*max(gain_trans, gain_yaw):+.1f}% pour {100*worst:.1f}% de degradation)')
        print(f'  {tag_of(settings):<44s} ' +
              '  '.join(f'{m.replace("rel_",""):14s} {100*delta[m]:+6.2f}%' for m in M) +
              f'   {verdict}', flush=True)

    full = survivors + arbitrate
    if not full:
        print('\naucune candidate ne survit au crible multicontact, et aucune ne merite un arbitrage.',
              flush=True)
        return
    if len(full) > KEEP_FULL:
        # Budget : on garde les meilleures de chaque metrique, meme regle que pour le crible.
        by = {m: sorted(full, key=lambda s: rows[tuple(sorted(s.items()))][m]) for m in PRIORITY}
        kept, seen_tags = [], set()
        while len(kept) < KEEP_FULL:
            grew = False
            for metric in PRIORITY:
                for s in by[metric]:
                    if tag_of(s) not in seen_tags:
                        seen_tags.add(tag_of(s)); kept.append(s); grew = True
                        break
                if len(kept) >= KEEP_FULL:
                    break
            if not grew:
                break
        print(f'\n{len(full)} candidates passent le crible, budget limite a {KEEP_FULL}', flush=True)
        full = kept
        survivors = [s for s in survivors if s in full]
        arbitrate = [s for s in arbitrate if s in full]
    print(f'\n=== {len(survivors)} survivante(s) et {len(arbitrate)} a arbitrer, '
          f'evaluation sur les cinq marches RHPS1', flush=True)
    subprocess.run([str(ROOT / '.venv/bin/python'), 'scripts/kinetics_eval.py',
                    '--projects', ','.join(REST), 'prepare'], cwd=ROOT, capture_output=True)
    ref_rest = run(dict(REF), REST)
    for settings in full:
        o = run(settings, REST)
        mark = '  [A ARBITRER]' if settings in arbitrate else ''
        if o is None:
            print(f'  {tag_of(settings):<44s} ECHEC{mark}', flush=True); continue
        cells = '  '.join(f'{m.replace("rel_",""):14s} {o[m]:.5f} '
                          f'{100*(o[m]/ref_rest[m]-1):+6.2f}%' for m in M) if ref_rest else \
                '  '.join(f'{m.replace("rel_",""):14s} {o[m]:.5f}' for m in M)
        print(f'  {tag_of(settings):<44s} {cells}{mark}', flush=True)
    if arbitrate:
        print('\n=== A ARBITRER : ces configurations degradent le multicontact de plus de 2 %,', flush=True)
        print('    mais leur gain sur les glissements DEPASSE cette degradation, donc la decision', flush=True)
        print('    te revient. Celles qui rendaient seulement ce qu elles coutaient sont jetees.', flush=True)
        for settings in arbitrate:
            key = tuple(sorted(settings.items()))
            print(f'  {tag_of(settings):<44s} glissements : ' +
                  '  '.join(f'{m.replace("rel_",""):14s} {100*(rows[key][m]/base_slip[m]-1):+6.2f}%' for m in M),
                  flush=True)
    print(f'\nNON EVALUE : {EXCLUDED}. A lancer a la main sur les survivantes avant de valider :',
          flush=True)
    print('  .venv/bin/python scripts/kinetics_eval.py --projects HRP5P_LongWalk \\', flush=True)
    print('    --covariance-overlay <overlay> run --label lw --no-plots --no-latest --no-open', flush=True)
    print('  (les overlays des survivantes sont dans ' + str(SP / 'overlays3') + ')', flush=True)
    print('ETAPE 3 TERMINEE', flush=True)


if __name__ == '__main__':
    main()
