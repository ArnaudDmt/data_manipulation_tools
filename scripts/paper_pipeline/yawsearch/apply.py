"""Ecrit les quatre reglages balayes dans la config du KO, et VERIFIE chaque ecriture.

Sort en erreur si un motif n'est pas trouve exactement une fois, ou si la relecture ne
retrouve pas la valeur demandee : une variante mal installee produirait silencieusement
un doublon de la ligne de reference.
"""
import re, sys
from pathlib import Path

CFG = Path.home() / ".config/mc_rtc/observers/MCKineticsObserver.yaml"
angvel, oriyaw, posxy, oriinit = sys.argv[1:5]

EDITS = [
    # (cle, remplacement complet de la liste)
    ("stateAngVelProcessVariance",        f"[{angvel}, {angvel}, {angvel}]"),
    ("contactOrientationProcessVariance", f"[1e-08, 1e-08, {oriyaw}]"),
    ("contactPositionProcessVariance",    f"[{posxy}, {posxy}, 1e-09]"),
    ("contactOriInitVarianceNewContacts", f"[{oriinit}, {oriinit}, {oriinit}]"),
]

text = CFG.read_text()
for key, value in EDITS:
    pattern = r"(?m)^(\s*" + key + r":\s*)\[[^\]]*\]"
    text, n = re.subn(pattern, lambda m, v=value: m.group(1) + v, text)
    if n != 1:
        sys.exit(f"ABANDON: {key} trouve {n} fois au lieu de 1")
CFG.write_text(text)

# Relecture independante : on ne fait pas confiance au remplacement.
after = CFG.read_text()
for key, value in EDITS:
    m = re.search(r"(?m)^\s*" + key + r":\s*(\[[^\]]*\])", after)
    if m is None or m.group(1) != value:
        sys.exit(f"ABANDON: apres ecriture, {key} = {m.group(1) if m else 'absent'}, attendu {value}")
print(f"ok  angvel={angvel}  oriyaw={oriyaw}  posxy={posxy}  oriinit={oriinit}")
