"""Comptage d'insectes avec la méthode retenue dans le notebook (OpenCV seulement).

Méthode : niveaux de gris -> seuil (pixels <= 70) -> composantes connexes
-> on garde les taches de 40 à 150 pixels qui ne touchent pas le bord -> on compte.

Utilisation :
    python compter.py                  # toutes les journées d'au moins 50 images -> comptages.csv + scores
    python compter.py image1.jpg ...   # compte les images données et enregistre une image annotée
"""
import csv
import sys
from pathlib import Path

import cv2

SEUIL = 70                # pixel "sombre" si <= 70
AIRE_MIN, AIRE_MAX = 40, 150
MIN_IMAGES_PAR_JOUR = 50


def detecter(image):
    """Boîtes (x1, y1, x2, y2) des taches retenues comme insectes."""
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, masque = cv2.threshold(gris, SEUIL, 255, cv2.THRESH_BINARY_INV)
    n, _, stats, _ = cv2.connectedComponentsWithStats(masque)
    hauteur, largeur = gris.shape
    boites = []
    for i in range(1, n):                       # 0 = le fond
        x, y, w, h, aire = stats[i]
        touche_le_bord = x == 0 or y == 0 or x + w == largeur or y + h == hauteur
        if AIRE_MIN <= aire <= AIRE_MAX and not touche_le_bord:
            boites.append((x, y, x + w, y + h))
    return boites


def nombre_annote(img_path):
    """Nombre de boîtes dans le fichier label YOLO (une ligne par insecte)."""
    label = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    return len(label.read_text().strip().splitlines())


def compter_images_donnees(chemins):
    for chemin in chemins:
        image = cv2.imread(chemin)
        if image is None:
            print(f"{chemin} : impossible de lire l'image")
            continue
        boites = detecter(image)
        for (x1, y1, x2, y2) in boites:
            cv2.rectangle(image, (int(x1) - 2, int(y1) - 2), (int(x2) + 2, int(y2) + 2), (0, 255, 0), 1)
        sortie = "resultat_" + Path(chemin).stem + ".png"
        cv2.imwrite(sortie, image)
        print(f"{chemin} : {len(boites)} insecte(s)  -> {sortie}")


def evaluer_dataset():
    # Images rangées par journée (les 8 premiers caractères du nom = la date)
    images_par_jour = {}
    for split in ["train", "valid", "test"]:
        for img in sorted(Path(split, "images").glob("*.jpg")):
            images_par_jour.setdefault(img.name[:8], []).append(img)
    jours = sorted(j for j in images_par_jour if len(images_par_jour[j]) >= MIN_IMAGES_PAR_JOUR)
    if not jours:
        print("Aucune image trouvée : lancer le script depuis le dossier qui contient train/, valid/ et test/.")
        return

    lignes = []
    print(f"{'journée':10} {'images':>6} {'MAE':>6} {'biais':>6} {'total vrai':>11} {'total prédit':>13}")
    for jour in jours:
        erreurs, total_vrai, total_predit = [], 0, 0
        for img in images_par_jour[jour]:
            predit = len(detecter(cv2.imread(str(img))))
            vrai = nombre_annote(img)
            lignes.append((img.name, jour, predit, vrai))
            erreurs.append(predit - vrai)
            total_vrai += vrai
            total_predit += predit
        mae = sum(abs(e) for e in erreurs) / len(erreurs)
        biais = sum(erreurs) / len(erreurs)
        print(f"{jour:10} {len(erreurs):6} {mae:6.2f} {biais:6.2f} {total_vrai:11} {total_predit:13}")

    erreurs = [p - v for (_, _, p, v) in lignes]
    print(f"{'GLOBAL':10} {len(erreurs):6} {sum(abs(e) for e in erreurs) / len(erreurs):6.2f} "
          f"{sum(erreurs) / len(erreurs):6.2f}")
    print("\nAttention : ces réglages ont été choisis sur ces mêmes images, ces scores sont donc optimistes.")
    print("Les scores honnêtes (une journée mise de côté) sont dans le notebook : MAE 0,89 contre 0,80 pour « toujours 1 ».")

    with open("comptages.csv", "w", newline="", encoding="utf-8") as f:
        ecrivain = csv.writer(f)
        ecrivain.writerow(["image", "journee", "predit", "vrai"])
        ecrivain.writerows(lignes)
    print(f"\n{len(lignes)} comptages enregistrés dans comptages.csv")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        compter_images_donnees(sys.argv[1:])
    else:
        evaluer_dataset()
