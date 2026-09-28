#!/usr/bin/env python3
"""Biais gyrometrique injecte contre estime : KO, KO sans canal angulaire, RI-EKF. LongWalk, chaud.

La verite est connue exactement, puisque nous l'avons injectee (modele BMI088, graine fixe). Les
trois axes sont traces parce que le resultat est justement anisotrope : la gravite rend les deux
axes HORIZONTAUX observables par le tilt, tandis que l'axe de LACET ne l'est que par les contacts.
Ne montrer que z cacherait la moitie du resultat, et ne montrer que x,y cacherait l'essentiel du
desaccord entre les deux estimateurs.

Les series du KO sont lues dans le states.npz du run lui-meme plutot que dans une extraction
intermediaire : l'origine est ainsi prouvee par le chemin. cold_variant.py ecrit les cles
`t, bias, force, torque` ; d'anciennes extractions utilisaient `t, b`, donc les deux sont acceptees.

MODE D'EMPLOI
-------------
Tout se regle dans les deux listes en tete du fichier, rien d'autre n'a besoin d'etre touche.

  SERIES        une ligne par courbe : (libelle, chemin, couleur, style). Le chemin peut etre un
                .npz de run du Kinetics Observer ou un .csv du parseur RI-EKF -- load() reconnait
                les deux a l'extension. Ajouter une courbe = ajouter une ligne ; la retirer =
                supprimer ou commenter la sienne. Une serie dont le fichier est absent est
                simplement ignoree, avec un message, donc rien ne casse.
  HORS_ECHELLE  les libelles exclus du calcul des bornes verticales. Une courbe qui diverge y est
                mise pour qu'elle sorte du cadre au lieu d'ecraser les autres ; elle reste tracee.

Le repertoire des CSV du RI-EKF se donne par la variable d'environnement KO_BIAS_DATA. Sans elle,
c'est le scratchpad de la session ou ces essais ont ete faits -- qui ne survit pas a la session.

Sortie : results/paper-rebuild/figures/injectedGyroBias_noangclean.{pdf,png}, plus un tableau du
biais final et du taux de rattrapage par axe imprime sur la sortie standard.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

ROOT = Path('/home/arnaud/devel/src/data_manipulation_tools')
sys.path.insert(0, str(ROOT / 'scripts/paper_pipeline/gyrobias'))
import model                             # noqa: E402

import os

# Repertoire des CSV produits par le parseur RI-EKF. Surchargeable par KO_BIAS_DATA : le defaut est
# le scratchpad de la session du 2026-09-17, qui ne survivra pas: deplacer les CSV et pointer
# KO_BIAS_DATA dessus pour rejouer la figure plus tard.
S = Path(os.environ.get(
    'KO_BIAS_DATA',
    '/tmp/claude-1000/-home-arnaud-devel-src-data-manipulation-tools/'
    '287daf9e-ad1c-4931-a07e-923142412c80/scratchpad/bias'))
AXES = ('x', 'y', 'z')
EVAL_START = 1002.28                     # la fenetre evaluee de LongWalk commence ici
BIAS_COLUMNS = [f'IMU_GyroBias_{a}' for a in AXES]

SERIES = [
    ('KO complet',              ROOT / 'results/bmi-b8/HRP5P_LongWalk/states.npz',        '#1a5276', '-'),
    # KO-Lin dans sa definition definitive : raideur et amortissement angulaires nuls, mesure de
    # couple de contact neutralisee, ET process du couple de contact nul.
    # NE PAS repointer sur `noangclean-bias` (process du couple encore a 225) ni sur `kolin-bias2` :
    # ce dernier a ete produit avec un garde-fou trop large qui faisait sauter, en plus du calcul
    # angulaire, le retrait de l'enfoncement LINEAIRE a la creation d'un contact. Son lacet de 0.5436
    # est meilleur pour cette raison-la, pas parce qu'un defaut aurait ete corrige.
    ('KO-Lin',                  ROOT / 'results/kolin-bias/HRP5P_LongWalk/states.npz',      '#5dade2', '--'),
    ('RI-EKF, process 2e-16',   S / 'HRP5P_LongWalk-biased-1e8.csv',                      '#922b21', '-'),
    ('RI-EKF, process 1.4e-11', S / 'HRP5P_LongWalk-biased-p14e11.csv',                   '#e59866', '--'),
    # Les deux leviers qu'on pouvait encore tenter, appliques ENSEMBLE : les covariances de process
    # de contact contraintes du Kinetics Observer (M = I - S.W', ponderees par la force mesuree)
    # portees au RI-EKF, et la variance initiale du biais ouverte sur le seul axe de lacet (1e-6).
    # Cette serie remplace les deux essais separes, qu'elle resume : chacun pris isolement donnait
    # deja son verdict -- la contrainte seule ne change rien (lacet 0.9345 -> 0.9329), l'ouverture
    # seule fait decrocher (23.35) -- et leur combinaison ne fait pas mieux : lacet 23.3255 contre
    # 23.3505 sans contrainte, biais final +0.1188 contre +0.1182. Un dixieme de pour cent.
    # La lecon est dans ce non-effet : la contrainte est ce qui, dans le KO, empeche la correction
    # des contacts de faire deriver le robot ; elle ne retient pas le biais de lacet parce que le
    # probleme n'est pas une mauvaise repartition des incoherences, mais l'absence de toute mesure
    # contraignant cet axe. On ne retient pas un etat que rien n'observe.
    # Le biais suit correctement la verite pendant la station debout (+0.0119 a l'entree de la
    # fenetre evaluee contre +0.0154 injectes) puis decroche des que la marche commence.
    ('RI-EKF, contrainte + init z=1e-6', S / 'HRP5P_LongWalk-biased-contrainte-iz1e6.csv', '#7b241c', '-.'),
]


def load(path):
    if path.suffix == '.npz':
        data = np.load(path)
        key = 'bias' if 'bias' in data else 'b'
        return data['t'], data[key]
    frame = pd.read_csv(path, sep=';', usecols=['t'] + BIAS_COLUMNS)
    return frame['t'].to_numpy(), frame[BIAS_COLUMNS].to_numpy()


def envelope(t, values, bins=1100):
    """Reduit la serie par min/max et moyenne dans chaque colonne de pixels.

    Une decimation naive (un echantillon sur 368 ici) ALIASE les bouffees d'oscillation du biais :
    elle rate les extremes et relie les survivants par des segments rectilignes, ce qui donne des
    pics isoles et des droites la ou le signal tremble. Mesure sur la fenetre 700-800 s de la
    variante sans canal angulaire : etendue reelle 0.004107 deg/s, etendue vue apres decimation
    0.003258, soit un quart de l'amplitude perdu, avec 135 points conserves sur 50 000.
    L'enveloppe montre la bande reellement occupee, et la moyenne la tendance.
    """
    index = np.clip(((t - t[0]) / (t[-1] - t[0]) * bins).astype(int), 0, bins - 1)
    centres = np.array([t[index == b].mean() for b in range(bins) if np.any(index == b)])
    low, high, mean = [], [], []
    for b in range(bins):
        selection = index == b
        if not np.any(selection):
            continue
        block = values[selection]
        low.append(block.min(axis=0))
        high.append(block.max(axis=0))
        mean.append(block.mean(axis=0))
    return centres, np.array(low), np.array(high), np.array(mean)


# Series tracees mais EXCLUES du calcul des bornes verticales. Celle-ci finit a +0.116 deg/s la ou
# toutes les autres tiennent sous 0.026 : la laisser fixer l'echelle tasserait les cinq courbes
# saines dans le bas du cadre, et c'est justement entre elles que la comparaison se joue. Elle sort
# donc du cadre, ce qui se lit tres bien comme une divergence.
HORS_ECHELLE = {'RI-EKF, contrainte + init z=1e-6'}

figure, panels = plt.subplots(3, 1, sharex=True, figsize=(9.5, 8.5))
report, span = [], 0.0
bornes = np.array([[np.inf, -np.inf]] * 3)   # min/max par axe, series retenues seulement

for label, path, colour, style in SERIES:
    if not path.exists():
        print(f'{label}: {path.name} absent, serie ignoree')
        continue
    t, bias = load(path)
    span = max(span, float(t[-1]))
    centres, low, high, mean = envelope(t, np.degrees(bias))
    if label not in HORS_ECHELLE:
        bornes[:, 0] = np.minimum(bornes[:, 0], low.min(axis=0))
        bornes[:, 1] = np.maximum(bornes[:, 1], high.max(axis=0))
    for row in range(3):
        panels[row].fill_between(centres, low[:, row], high[:, row], color=colour, alpha=0.25,
                                 lw=0, zorder=1)
        panels[row].plot(centres, mean[:, row], color=colour, lw=1.3, ls=style, label=label,
                         zorder=2)
    # Rattrapage par axe : moyenne du biais estime rapportee a la moyenne du biais injecte, sur la
    # seule fenetre evaluee -- avant elle le filtre construit encore son etat.
    window = t >= EVAL_START
    truth = model.at(t[window])
    report.append((label, np.degrees(bias[-1]),
                   [100.0 * np.mean(bias[window][:, k]) / np.mean(truth[:, k]) for k in range(3)]))

grid = np.linspace(0.0, span, 4000)
truth_curve = np.degrees(model.at(grid))
for row, axis in enumerate(AXES):
    panels[row].axvspan(0.0, EVAL_START, color='0.92', zorder=0)
    panels[row].plot(grid, truth_curve[:, row], color='black', lw=2.0, ls=':',
                     label='biais injecte', zorder=3)
    panels[row].set_ylabel(f'biais {axis}  (deg/s)')
    panels[row].grid(alpha=0.3)
    # Le biais injecte entre dans le calcul des bornes : c'est la reference, elle doit rester
    # visible. Les series de HORS_ECHELLE, elles, debordent volontairement du cadre.
    bas = min(bornes[row, 0], truth_curve[:, row].min())
    haut = max(bornes[row, 1], truth_curve[:, row].max())
    marge = 0.06 * (haut - bas) if haut > bas else 1e-3
    panels[row].set_ylim(bas - marge, haut + marge)

handles, labels = panels[0].get_legend_handles_labels()
order = [labels.index('biais injecte')] + [i for i, l in enumerate(labels) if l != 'biais injecte']
panels[0].legend([handles[i] for i in order], [labels[i] for i in order], ncol=2, loc='upper left',
                 frameon=False, fontsize=9)
panels[0].set_title('Biais gyrometrique injecte (BMI088) et estime -- LongWalk, demarrage chaud')
panels[-1].set_xlabel('Temps (s)   (la zone grisee precede la fenetre evaluee)')
figure.tight_layout()

destination = ROOT / 'results/paper-rebuild/figures/injectedGyroBias_noangclean.pdf'
destination.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(destination)
figure.savefig(destination.with_suffix('.png'), dpi=150)
print(f'ecrit {destination}\n')

end_truth = np.degrees(model.at(np.array([span]))[0])
print('biais injecte en fin de log : '
      + '  '.join(f'{a}={v:+.5f}' for a, v in zip(AXES, end_truth)) + '  deg/s\n')
print(f"{'estimateur':26s} {'bx':>9s} {'by':>9s} {'bz':>9s}   {'rattrape x':>10s} {'y':>8s} {'z':>8s}")
for label, final, recovered in report:
    print(f'{label:26s} {final[0]:+9.5f} {final[1]:+9.5f} {final[2]:+9.5f}   '
          + '  '.join(f'{v:7.1f}%' for v in recovered))
