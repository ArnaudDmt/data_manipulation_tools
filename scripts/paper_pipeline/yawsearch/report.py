"""Depouille le balayage croise : metriques rpg poolees sur les trois glissements."""
import pickle, re, sys
import numpy as np
from pathlib import Path

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
GRID = ROOT / 'results/paper-rebuild/runs/grid'
RUNS = ROOT / 'results/paper-rebuild/runs'
P = [f'KO_TRO_2024_RHPS1_SLIPPAGE_{i}' for i in (1, 2, 3)]
M = ("rel_trans_x_y_norm", "rel_trans_z", "rel_tilt", "rel_yaw")
REF = dict(av='1e-10', oy='0.0001', px='1e-05', oi='0.0002')


def pooled(directory, name='cached_rel_err.pickle'):
    """Moyenne des valeurs absolues, poolee sur les trois essais -- la methode du papier."""
    out = {}
    for metric in M:
        values = []
        for project in P:
            cache = directory / project / name
            if not cache.exists():
                return None
            values.append(np.abs(np.asarray(pickle.load(cache.open('rb'))[1.0][metric], float)))
        out[metric] = float(np.concatenate(values).mean())
    return out


rows = {}
for d in sorted(GRID.glob('av*')):
    m = re.fullmatch(r'av(.+?)_oy(.+?)_px(.+?)_oi(.+)', d.name)
    if not m:
        continue
    o = pooled(d)
    if o is not None:
        rows[m.groups()] = o

clean = pooled(RUNS / 'clean')
riekf = pooled(RUNS / 'clean', 'riekf_rel_err.pickle')
key_ref = (REF['av'], REF['oy'], REF['px'], REF['oi'])
print(f"{len(rows)} combinaisons depouillees sur 135\n")

if key_ref in rows and clean:
    print("--- controle : la cellule de reference doit reproduire runs/clean")
    for metric in M:
        a, b = clean[metric], rows[key_ref][metric]
        flag = 'OK' if abs(b / a - 1) < 1e-6 else f'ECART {100*(b/a-1):+.3f}%'
        print(f"    {metric.replace('rel_',''):16s} {a:.6f}  vs  {b:.6f}   {flag}")
    print()

base = rows.get(key_ref) or clean
order = sorted(rows, key=lambda k: rows[k]['rel_trans_x_y_norm'])


def line(k):
    o = rows[k]
    cells = '  '.join(f"{o[m]:.5f} {100*(o[m]/base[m]-1):+6.1f}%" for m in M)
    star = ' <- reference' if k == key_ref else ''
    return f"av={k[0]:<6s} oy={k[1]:<7s} px={k[2]:<6s} oi={k[3]:<7s} {cells}{star}"


hdr = '  '.join(f"{m.replace('rel_',''):>17s}" for m in M)
print(f"{'':44s}{hdr}")
print("--- 15 meilleures en translation xy")
for k in order[:15]:
    print('  ' + line(k))
print("\n--- 5 pires en translation xy")
for k in order[-5:]:
    print('  ' + line(k))

print("\n--- 10 meilleures en lacet")
for k in sorted(rows, key=lambda k: rows[k]['rel_yaw'])[:10]:
    print('  ' + line(k))

print("\n--- effet marginal de chaque reglage (moyenne sur toutes les autres combinaisons)")
for axis, idx in (('stateAngVel', 0), ('contactOriProc z', 1), ('contactPosProc xy', 2), ('contactOriInit', 3)):
    vals = sorted({k[idx] for k in rows}, key=lambda v: float(v))
    print(f"  {axis}")
    for v in vals:
        sel = [rows[k] for k in rows if k[idx] == v]
        cells = '  '.join(f"{np.mean([s[m] for s in sel]):.5f} {100*(np.mean([s[m] for s in sel])/base[m]-1):+6.1f}%" for m in M)
        tag = ' (ref)' if v == REF[('av', 'oy', 'px', 'oi')[idx]] else ''
        print(f"    {v:<8s} n={len(sel):<4d} {cells}{tag}")

if riekf:
    print("\n--- rapport KO/RI-EKF, 10 meilleures en translation xy")
    for k in order[:10]:
        print('  ' + f"av={k[0]:<6s} oy={k[1]:<7s} px={k[2]:<6s} oi={k[3]:<7s} " +
              '  '.join(f"{rows[k][m]/riekf[m]:>17.3f}" for m in M))
    print('  ' + f"{'reference publiee':<44s}" + '  '.join(f"{clean[m]/riekf[m]:>17.3f}" for m in M))
