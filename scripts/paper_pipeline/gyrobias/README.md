# Derive de biais gyrometrique injectee

HRP5-P porte un gyro a fibre optique : son biais mesure 0.0003 deg/s, il ne derive pas. L'excellent
lacet du RI-EKF dans le papier vient de la, pas de l'estimateur. Ces scripts injectent la derive
qu'une centrale MEMS standard -- celles des quadrupedes -- produirait, et mesurent ce que chaque
estimateur en fait.

- `model.py`   la derive, generee sur l'axe du TEMPS puis interpolee sur les horodatages de chaque
               fichier : le bag et HartleyInput.txt n'ont pas la meme cadence et les deux
               estimateurs doivent voir la meme derive au meme instant.
- `riekf.py`   injecte dans HartleyInput.txt et relance le parseur autonome. Pas de re-tick.
- `ko.py`      reecrit le bag dans un cache suffixe, sans jamais toucher l'original.
- `run.py`     bruit blanc gaussien contre uniforme, a variance egale.
- `real.py`    modele complet d'une MEMS : ARW gaussien, marche aleatoire du biais, residu
               d'etalonnage, quantification.

## Resultat, RI-EKF sur LongWalk, sous-trajectoires de 10 m

    propre                     transl 195.53 mm   lacet 0.3739 deg   derive  -1.67 deg
    biaise, process actuel            221.27            2.4433              88.92
    biaise, process adapte            207.91            2.2389              27.86

Adapter le process de biais a la marche aleatoire injectee (q = sigma^2/T, 700 000 fois la valeur
actuelle) ne recupere que 51 % du biais et laisse le lacet six fois pire. Desserrer davantage fait
diverger. Ce n'est pas un defaut de reglage : le lacet du RI-EKF est un integral pur, et les
contacts ne l'atteignent que par les termes croises de covariance, un canal trop faible.

La quantification est uniforme mais 31 fois plus petite que l'ARW : invisible. Un bruit uniforme a
variance egale degrade le lacet de 4.6 %, mais il ne correspond a aucun capteur reel -- c'est une
sonde de sensibilite a la forme de la loi, pas un modele.
