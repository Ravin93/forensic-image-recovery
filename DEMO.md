# Guide de démo — Forensic Image Recovery

Scénario de démonstration fiable pour la soutenance ESGI.  
Durée estimée : **5–8 minutes** pour les deux scénarios.

---

## Prérequis

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer le serveur API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Vérifier que l'API répond :
```
GET http://localhost:8000/health  →  {"status": "ok"}
```

---

## Scénario 1 — Interface web : image corrompue → reconstruction

### Étapes

**1. Ouvrir l'interface**
- Ouvrir `forensic_ui.html` dans le navigateur (double-clic ou `file://`)

**2. Uploader une image**
- Déposer n'importe quelle photo JPEG ou PNG (≤ 20 MB)
- Recommandé : image avec textures variées (paysage, visage, document)
- À éviter pour la démo : image très sombre ou noire (fausse la détection aveugle)

**3. Configurer l'analyse**
- Type de corruption : `scratch_lines` ou `multiple_bars` ← résultats visuels forts
- Sévérité : `medium`
- Mode : `assisted` ← résultats optimaux pour la démo (masque exact utilisé)
- Tentatives : `8` (défaut)

**4. Lancer l'analyse**
- Cliquer **Analyser**
- Durée : 3–8 secondes selon la taille de l'image

**5. Lire les résultats**
- Triptyque original / corrompu / reconstruit
- Score 0–100 avec décomposition PSNR/SSIM
- Tableau des 8+ candidats et leur score
- Stratégie sélectionnée automatiquement

**6. Télécharger le rapport**
- **Rapport PDF** : inclut chain of custody, SHA-256, méthodes
- **Rapport HTML** : auto-contenu, images en base64, ouvre sans serveur

### Ce qu'on montre

| Point clé | Preuve visuelle |
|---|---|
| Corruption réaliste | Rayures ou barres sur l'image |
| Reconstruction automatique | Zone reconstruite propre |
| Score objectif | PSNR/SSIM mesurés |
| 14 stratégies testées | Tableau des candidats |
| Rapport forensique | PDF avec chain of custody |

---

## Scénario 2 — Carving forensique : dump binaire → extraction JPEG

```bash
# Démo complète en ligne de commande
python scripts/demo_dump_recovery.py --no-api
```

**Sortie attendue :**
```
[1/4] Génération du dump binaire synthétique
  ✓ Dump généré : data/dumps/test_dataset/demo.bin  (309,507 octets)
  ✓ 3 images JPEG insérées parmi des octets aléatoires
  → Fragment #1 : offset=649  taille=298807 octets
  → Fragment #2 : offset=300032  taille=4204 octets

[2/4] Carving JPEG (détection marqueurs SOI/EOI)
  ✓ 3 fichier(s) JPEG extrait(s) en 0.15s

[3/4] Validation des JPEG extraits
  ✓ JPEG #1  298,807 octets  sha256=ac836236…  2400×1599px
  ✓ Recall parfait : 3/3 images récupérées
```

**Avec votre propre image :**
```bash
python scripts/demo_dump_recovery.py \
    --source chemin/vers/photo.jpeg \
    --no-api
```

### Ce qu'on montre

| Point clé | Preuve |
|---|---|
| Dump binaire synthétique | Fichier `.bin` avec JPEG + garbage |
| Détection SOI/EOI | Offsets exacts retrouvés |
| Validation JPEG | Images réouvertes, dimensions vérifiées |
| Recall 3/3 | Toutes les images récupérées |

---

## Démo bonus — Éditeur de masque manuel

Pour montrer le mode forensique avancé (image déjà corrompue sans pipeline automatique) :

1. Ouvrir `mask_editor.html`
2. Charger une image corrompue (depuis `data/corrupted/` après un scénario 1, ou uploader manuellement)
3. Peindre la zone abîmée au pinceau
4. Lancer la reconstruction
5. Comparer le résultat dans le panneau de droite

---

## Points de langage pour la soutenance

> "Le système ne prétend pas reconstruire une vérité forensique absolue. Il produit une reconstruction plausible, mesurable, et documentée, avec un score de confiance et des limites explicites."

> "Les 14 stratégies sont testées automatiquement. Le conservative — qui retourne l'image corrompue sans modification — sert de baseline : si une reconstruction dégrade l'image, le score le détecte."

> "LaMa deep inpainting est préparé mais non activé en production. Un résultat génératif ne peut pas être utilisé comme preuve forensique sans validation humaine."

---

## À éviter en démo

| Cas | Pourquoi |
|---|---|
| `large_deleted_square` + `heavy` | Zone >15% — limite documentée de l'inpainting OpenCV |
| Images très sombres | Fausse la détection en mode aveugle |
| `severity: heavy` + mode `blind_basic` | Détection imprécise, score plus faible |
| LaMa activé | Non testé en production, résultat non reproductible |
