#!/usr/bin/env python3
"""Ce que la phase non notee apporte, et ce que le calage de la derive change.

LongWalk dure 2948 s mais seules les 1945.9 dernieres sont notees : avant, le robot est DEBOUT
IMMOBILE (0.56 m parcourus contre 309 m ensuite), parce que la mocap n'a ete lancee qu'au depart de
la marche. C'est un regime ou un biais gyrometrique s'observe tres bien -- la vitesse angulaire
vraie est nulle -- mais ou le lacet n'est pas excite.

Trois colonnes :

  chaud           les filtres tournent depuis le debut du log, la derive aussi ;
  froid, echelon  les filtres redemarrent au premier instant note mais la derive reste calee sur le
                  debut du log : ils encaissent un ECHELON de 0.0154 deg/s, ce qu'aucune centrale ne
                  fait -- montre ici parce que c'est le piege dans lequel la premiere version est
                  tombee, et parce que l'ecart avec la colonne suivante mesure exactement ce que
                  l'echelon fabriquait ;
  froid, rampe    filtres ET derive repartent ensemble : la centrale est mise a zero a l'arret puis
                  chauffe pendant la marche. C'est le scenario physique.

Dans les trois cas les filtres partent du meme etat froid (pose a cet instant, zero ailleurs) et
l'entree est tronquee de la meme fraction, 34.0 %.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model
import score_coldstart as score

SCRATCH, ROOT = score.SCRATCH, score.ROOT
COLOURS = {'RI-EKF papier': '#c0392b', 'RI-EKF adapte': '#e59866', 'Kinetics Observer': '#2471a3'}
STYLES = {'RI-EKF papier': '-', 'RI-EKF adapte': ':', 'Kinetics Observer': '-'}
COLUMNS = [
    ('chaud', 'log', "Depart chaud\n17 min de station debout avant la fenetre"),
    ('froid, echelon', 'log', "Depart froid, derive calee sur le log\nechelon de 0.015 deg/s a t=0"),
    ('froid, rampe', 'rebase', "Depart froid, derive rebasee\nIMU mise a zero puis chauffe"),
]
SOURCES = {
    ('RI-EKF papier', 'chaud'): SCRATCH / 'HRP5P_LongWalk-biased.csv',
    ('RI-EKF papier', 'froid, echelon'): SCRATCH / 'cold-papier.csv',
    ('RI-EKF papier', 'froid, rampe'): SCRATCH / 'cold2-papier.csv',
    ('RI-EKF adapte', 'chaud'): SCRATCH / 'HRP5P_LongWalk-biased-adapted.csv',
    ('RI-EKF adapte', 'froid, echelon'): SCRATCH / 'cold-adapte.csv',
    ('RI-EKF adapte', 'froid, rampe'): SCRATCH / 'cold2-adapte.csv',
    ('Kinetics Observer', 'chaud'): ('kobmi-ba73545d4c', 'ko_bias_bmi.npz'),
    ('Kinetics Observer', 'froid, echelon'): ('kocold-ba73545d4c', 'ko_bias_cold.npz'),
    ('Kinetics Observer', 'froid, rampe'): ('kocold2-ba73545d4c', 'ko_bias_cold2.npz'),
}

t, p_t, q_t, first, last = score.geometry()
starts = t[first]
BIAS = {'log': model.at(t + score.OFFSET)[:, 2], 'rebase': model.at(t)[:, 2]}


def smooth(values, seconds=150.0):
    n = max(1, int(seconds / np.median(np.diff(starts))))
    return np.convolve(values, np.ones(n) / n, mode='same')


figure, panels = plt.subplots(2, 3, figsize=(15.5, 7.8), sharex=True, sharey='row',
                              gridspec_kw={'height_ratios': [1, 1.25]})
for column, (start, calage, title) in enumerate(COLUMNS):
    top, bottom = panels[0][column], panels[1][column]
    truth = BIAS[calage]
    top.plot(starts, np.degrees(truth[first]), color='black', lw=2.0, ls='--',
             label='biais z injecte', zorder=3)
    for name in COLOURS:
        source = SOURCES[(name, start)]
        p_e, q_e, bias = (score.from_ko(*source) if isinstance(source, tuple)
                          else score.from_csv(source, t))
        _, yaw, _ = score.errors(p_e, q_e, p_t, q_t, first, last)
        recovered = 100.0 * np.mean(bias[first]) / np.mean(truth[first])
        top.plot(starts, np.degrees(bias[first]), color=COLOURS[name], lw=1.5,
                 ls=STYLES[name], label=f'{name}  ({recovered:.0f} %)')
        bottom.plot(starts, smooth(yaw), color=COLOURS[name], lw=1.9, ls=STYLES[name],
                    label=f'{name}  ({yaw.mean():.3f} deg)')
    top.set_title(title, fontsize=10)
    top.grid(alpha=0.3)
    top.legend(frameon=False, fontsize=8, loc='upper left')
    bottom.grid(alpha=0.3)
    bottom.legend(frameon=False, fontsize=8.5, loc='upper left')
    bottom.set_xlabel('Temps depuis le debut de la fenetre evaluee  (s)')
panels[0][0].set_ylabel('biais z  (deg/s)\n(% = part rattrapee)')
panels[1][0].set_ylabel('erreur de lacet par fenetre de 10 m  (deg)\nmoyenne glissante sur 150 s')
figure.suptitle('Demarrage a froid au premier instant de la mocap -- LongWalk, derive BMI088')
figure.tight_layout()

destination = ROOT / 'results/paper-rebuild/figures/coldstart_longwalk.pdf'
figure.savefig(destination)
figure.savefig(destination.with_suffix('.png'), dpi=150)
print(f'ecrit {destination}')
