"""Depouillement des deux etapes : metriques rpg poolees sur les trois glissements."""
import pickle, re
import numpy as np
from pathlib import Path

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
RUNS = ROOT / 'results/paper-rebuild/runs'
P = [f'KO_TRO_2024_RHPS1_SLIPPAGE_{i}' for i in (1, 2, 3)]
M = ("rel_trans_x_y_norm", "rel_trans_z", "rel_tilt", "rel_yaw")
REF = dict(av='1e-10', oy='0.0001', px='1e-05', oi='0.0002', gb='1e-18', cf='900', fz='900', ct='225', cz='225',
           uf='0.09', un='0.09', ut='0.09', uz='0.09')


def pooled(d, name='cached_rel_err.pickle'):
    out = {}
    for metric in M:
        values = []
        for project in P:
            cache = d / project / name
            if not cache.exists():
                return None
            values.append(np.abs(np.asarray(pickle.load(cache.open('rb'))[1.0][metric], float)))
        out[metric] = float(np.concatenate(values).mean())
    return out


rows = {}
for d in sorted((RUNS / 'grid').glob('av*')):
    m = re.fullmatch(r'av(.+?)_oy(.+?)_px(.+?)_oi(.+)', d.name)
    o = pooled(d)
    if m and o:
        s = dict(REF); s.update(dict(zip(('av', 'oy', 'px', 'oi'), m.groups())))
        rows[tuple(sorted(s.items()))] = o
for d in sorted((RUNS / 'grid2').glob('*')):
    o = pooled(d)
    if not o:
        continue
    s = dict(REF)
    for part in re.findall(r'(av|oy|px|oi|gb|cf|fz|ct|cz|uf|un|ut|uz)([^_]+)', d.name):
        s[part[0]] = part[1]
    rows[tuple(sorted(s.items()))] = o

clean = pooled(RUNS / 'clean')
riekf = pooled(RUNS / 'clean', 'riekf_rel_err.pickle')
key_ref = tuple(sorted(REF.items()))
base = rows.get(key_ref) or clean
print(f"{len(rows)} configurations evaluees\n")

if key_ref in rows and clean:
    print("--- controle : la cellule de reference reproduit runs/clean")
    for metric in M:
        a, b = clean[metric], rows[key_ref][metric]
        print(f"    {metric.replace('rel_',''):16s} {a:.6f}  vs  {b:.6f}   "
              f"{'OK' if abs(b/a-1) < 1e-6 else f'ECART {100*(b/a-1):+.3f}%'}")
    print()


def show(k):
    o = rows[k]
    d = dict(k)
    name = ' '.join(f'{x}={d[x]}' for x in ('av', 'oy', 'px', 'oi', 'gb', 'cf', 'fz', 'ct', 'cz', 'uf', 'un', 'ut', 'uz') if d[x] != REF[x]) or 'reference'
    cells = '  '.join(f"{o[m]:.5f} {100*(o[m]/base[m]-1):+6.2f}%" for m in M)
    return f"  {name:<52s} {cells}"


hdr = '  '.join(f"{m.replace('rel_',''):>17s}" for m in M)
print(f"{'':54s}{hdr}")
print("--- sans regression au-dela de 0.5 %, triees par LACET (la metrique prioritaire)")
safe = [k for k in rows if all(rows[k][m] <= base[m] * 1.005 for m in M)]
for k in sorted(safe, key=lambda k: rows[k]['rel_yaw'])[:15]:
    print(show(k))
if not safe:
    print("  aucune")
print("\n--- 10 meilleures en translation, quel qu en soit le cout")
for k in sorted(rows, key=lambda k: rows[k]['rel_trans_x_y_norm'])[:10]:
    print(show(k))
print("\n--- 10 meilleures en lacet")
for k in sorted(rows, key=lambda k: rows[k]['rel_yaw'])[:10]:
    print(show(k))

print("\n--- effet marginal de chaque reglage (moyenne sur toutes les autres configurations)")
for axis in ('av', 'oy', 'px', 'oi', 'gb', 'cf', 'fz', 'ct', 'cz', 'uf', 'un', 'ut', 'uz'):
    levels = sorted({dict(k)[axis] for k in rows}, key=float)
    if len(levels) < 2:
        continue
    print(f"  {axis}")
    for v in levels:
        sel = [rows[k] for k in rows if dict(k)[axis] == v]
        cells = '  '.join(f"{np.mean([s[m] for s in sel]):.5f} "
                          f"{100*(np.mean([s[m] for s in sel])/base[m]-1):+6.2f}%" for m in M)
        print(f"    {v:<8s} n={len(sel):<4d} {cells}{' (ref)' if v == REF[axis] else ''}")

if riekf:
    print("\n--- rapport KO/RI-EKF")
    print(f"  {'reference publiee':<52s}" + '  '.join(f"{clean[m]/riekf[m]:>17.3f}" for m in M))
    for k in sorted(safe or rows, key=lambda k: rows[k]['rel_yaw'])[:8]:
        print(show(k).split('  ')[0] + f"{'':2s}" +
              '  '.join(f"{rows[k][m]/riekf[m]:>17.3f}" for m in M))
