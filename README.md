# Comptage d'insectes par traitement d'image classique

Compter les insectes sur chaque image d'un piège photographique (jeu *Insect Detect*), avec OpenCV uniquement (sans deep learning), et mesurer honnêtement ce que vaut la méthode.

**En bref :** la méthode retenue (seuil de luminosité + filtre de taille) tourne en une dizaine de secondes sur 1 101 images, mais elle **ne fait pas nettement mieux qu'une référence naïve qui répond toujours « 1 insecte »**. Elle trouve 46 % des insectes, une détection sur deux est fausse, et elle ne permet pas de classer les journées selon le nombre d'insectes. Les causes sont identifiées (ombres et vis aussi sombres et de même taille que les insectes, plaque qui bouge, insectes de quelques pixels), et j'en tire un cahier des charges pour l'acquisition.

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `test_agri_odor.ipynb` | toute la démarche : observation, protocole, essais, échecs, résultats (sorties incluses) |
| `compter.py` | la méthode retenue seule, en un script |
| `requirements.txt` | dépendances |

## Lancer le code

1. Télécharger le jeu *Insect Detect* (Zenodo, enregistrement [7725941](https://zenodo.org/records/7725941), version 416 × 416, 1 classe, format YOLO) et le décompresser dans le dossier du dépôt, de façon à avoir `train/`, `valid/` et `test/` à côté de `compter.py`.
2. Installer les dépendances : `pip install -r requirements.txt`
3. Lancer :
   - `python compter.py` : compte toutes les images des journées retenues, affiche les scores par journée et écrit `comptages.csv` (image, journée, prédit, vrai) ;
   - `python compter.py image.jpg` : compte une image et enregistre `resultat_image.png` avec les détections en vert ;
   - le notebook se lance de haut en bas (« Exécuter tout ») en 3 à 4 minutes.

Testé avec OpenCV 5.0, NumPy 2.2 et Matplotlib 3.10, sur un ordinateur portable.

## Données

- 1 335 images de 416 × 416 pixels, redimensionnées par Roboflow en « stretch » (proportions déformées). Une boîte YOLO par insecte, une seule classe.
- La date est dans le nom du fichier. Sur 23 journées, **7 ont au moins 50 images** (consigne) : 06, 07, 12, 17, 18, 19 et 20 octobre 2022, soit **1 101 images** et 1 931 insectes annotés.
- De 0 à 6 insectes par image, le plus souvent 1 ou 2. Les insectes sont petits : boîtes de 14 × 25 pixels en médiane, et 13 % font moins de 100 pixels² de surface (l’équivalent de 10 × 10).

## Protocole d'évaluation

- **Par image**, quatre mesures sur l'erreur (compté − vrai) : exactitude (compte exact), **MAE** (erreur absolue moyenne), **biais** (erreur moyenne, signe compris), exactitude à ±1.
- **Toujours par journée**, et pas seulement au global : une journée représente une situation (lumière, position de la plaque), et le 06/10 fait à lui seul 35 % des images.
- **Références naïves**, qui ne regardent pas l'image : « toujours 1 » et « moyenne des autres journées » (= 2). Une méthode qui ne les bat pas n'apporte rien.
- **Réglage en mettant une journée de côté** : chaque journée est testée avec des réglages choisis sur les six autres. Le découpage train / valid / test de Roboflow est aléatoire : des photos prises à quelques secondes d'écart, presque identiques, se retrouvent des deux côtés, ce qui gonflerait les scores.
- **Par détection** : une détection est juste si son centre tombe dans une boîte annotée (appariement un pour un), d'où précision et rappel. Cela vérifie qu'un bon compte ne vient pas d'erreurs qui se compensent.
- **Côté biologiste** : nombre moyen d'insectes par image pour chaque journée, et classement des journées.

## Ce que j'ai essayé

| Étape | Idée | Résultat (MAE globale, une journée de côté sauf mention) |
|---|---|---|
| Seuil simple | pixels ≤ 80, taches de 8 à 600 pixels ne touchant pas le bord | 4,19 : environ 5 fausses détections par image (ombres, vis, bords des coupelles) |
| Masque de la plaque | ne chercher que sur la plaque blanche | 4,24 : aucun effet, les fausses détections sont sur la plaque |
| **Seuil + taille réglés** | grille seuil × aire min × aire max, réglée en minimisant la MAE | **0,89 : méthode retenue** (seuil 70, aire 40–150) |
| Grille plus large | plus de valeurs possibles | 0,95 : pire, sur-ajustement aux journées de réglage |
| Soustraction de fond | fond = image médiane de la journée | 1,13 même réglée sur tout : **la plaque bouge** au cours de la journée |
| Black-hat morphologique | faire ressortir les petits détails sombres | 1,08 même réglé sur tout : ressort aussi les vis et les bords |
| Réglage par F1 | régler pour de bonnes détections plutôt qu'un bon compte | 1,97 : rappel 0,56 mais précision 0,35, le problème vient des caractéristiques, pas du critère de réglage |

## Solution retenue

Niveaux de gris → seuil (pixels ≤ 70) → composantes connexes → on garde les taches de 40 à 150 pixels qui ne touchent pas le bord → on compte.

| Évalué en « une journée de côté » | Exactitude | MAE globale | MAE moyenne des journées | Biais | ±1 |
|---|---|---|---|---|---|
| Toujours 1 | 0,44 | 0,80 | 0,77 | −0,75 | 0,81 |
| Moyenne des autres journées (= 2) | 0,35 | 0,73 | – | +0,25 | 0,94 |
| Seuil simple | 0,08 | 4,19 | – | +4,17 | 0,14 |
| **Méthode retenue** | **0,38** | **0,89** | **1,02** | **−0,03** | **0,80** |

Par journée, face à « toujours 1 » :

| Journée | 06/10 | 07/10 | 12/10 | 17/10 | 18/10 | 19/10 | 20/10 |
|---|---|---|---|---|---|---|---|
| MAE méthode | 0,62 | 1,17 | 0,46 | 0,93 | 0,83 | 1,15 | 1,94 |
| MAE toujours 1 | 0,65 | 1,51 | 0,11 | 0,74 | 0,97 | 1,09 | 0,35 |

- **Pas de différence nette avec la référence** : mieux sur 3 journées sur 7, moins bien sur 4, nettement moins bien le 12/10 et le 20/10.
- **Détections** : précision 0,53, rappel 0,46 (884 insectes trouvés, 1 047 ratés, 798 fausses détections). Le biais quasi nul vient d'une **compensation** entre ratés et fausses détections, pas d'une bonne détection.
- **Pour un biologiste** : l'erreur sur le total d'insectes d'une journée va de −42 % à +159 %, et le **classement des journées est faux** (la journée prédite la plus riche, le 20/10, est en réalité l'avant-dernière).
- Réglée et testée sur toutes les images (score optimiste, celui qu'affiche `compter.py`), la MAE est de 0,82.

## Ce qui ne marche pas, et pourquoi

- **Ombres et vis** : aussi sombres que les insectes et de la même taille. Seuil et aire ne suffisent pas à les séparer, quel que soit le critère de réglage.
- **La plaque bouge** au cours d'une journée : impossible d'utiliser un fond de référence, qui serait pourtant la méthode classique naturelle pour ce problème.
- **Insectes minuscules** : 13 % des boîtes font moins de 100 pixels² : la tache sombre qu'elles contiennent passe souvent sous l'aire minimale retenue (40 pixels). Certaines sont douteuses même à l'œil (point sombre qui reste au même endroit sur plusieurs photos).

## Ce que je demanderais pour l'acquisition

| Problème observé | Demande |
|---|---|
| La plaque bouge, la soustraction de fond échoue | caméra et plaque fixées rigidement l'une à l'autre |
| Les ombres sont la principale source de fausses détections | éclairage diffus, pas de soleil direct, objets mats |
| Vis et boutons détectés comme insectes | vis de la couleur de la plaque, ou hors du champ |
| Insectes de quelques pixels (13 % sous 100 pixels²) | résolution telle que le plus petit insecte d'intérêt fasse au moins 15 pixels de long ; pas de redimensionnement « stretch » |
| Balance des blancs qui change (17/10) | exposition et balance des blancs fixes |
| Un même insecte compté sur plusieurs photos | cadence adaptée à la question (individus ou présence) et horodatage précis |

## Limites de l'évaluation

- Seulement 7 journées, et des images très corrélées au sein d'une journée : des écarts de quelques centièmes de MAE ne sont pas significatifs. Seules les grandes tendances le sont.
- Deux petites fuites assumées : le seuil de départ (80) a été choisi en regardant une image du 18/10, et l'histogramme des tailles qui a guidé la grille utilise toutes les journées.
- J'ai supposé que la question du biologiste était de comparer les journées. Si c'est autre chose (présence, individus, temps passé sur une fleur), la métrique change. C'est la première chose à demander.
- La vérité terrain n'est pas parfaite et l'accord entre annotateurs est inconnu.

## Suite possible

1. Agir sur l'acquisition (tableau ci-dessus) : probablement le levier le plus efficace.
2. Écrire des règles d'annotation (taille minimale, insecte immobile) et mesurer l'accord entre deux annotateurs.
3. Tester des critères de forme et de couleur (allongement, contraste local) pour séparer insectes et vis.
4. Si nécessaire, un modèle appris, évalué avec le même protocole (journée de côté, mesures par journée).

## Temps passé et outils

- Environ **15 heures** au total.
- J'ai utilisé un assistant IA (Claude) pour m'aider à écrire et relire le code et les textes. Les choix d'évaluation et l'interprétation des résultats ont été discutés et vérifiés étape par étape.
