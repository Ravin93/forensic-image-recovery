# Forensic Image Recovery

> Système forensique de reconstruction d'images corrompues — ESGI Projet Annuel 2026

[![Tests](https://img.shields.io/badge/tests-398%20passed-00e096)](tests/)
[![Python](https://img.shields.io/badge/python-3.13-00e5ff)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-a259ff)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-muted)](LICENSE)

---

## Présentation

**Forensic Image Recovery** est un pipeline complet de reconstruction forensique d'images numériques corrompues. Le système simule des dégradations réalistes (perte de secteur, rayures, barres, bruit, blocs supprimés), puis applique automatiquement jusqu'à 14 stratégies de reconstruction pour sélectionner la meilleure par scoring PSNR/SSIM.

Le projet couvre l'intégralité du cycle forensique :
- Corruption simulée (15 types)
- Reconstruction multi-stratégies avec moteur adaptatif
- Scoring supervisé et aveugle avec décomposition détaillée
- Analyse forensique avancée (EXIF, LSB, copy-move, PRNU, clustering)
- Rapports JSON / PDF / HTML avec chaîne de conservation légale
- Interface web dark + API REST complète

---

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Backend API | FastAPI + Uvicorn |
| Vision | OpenCV 4.13, scikit-image, Pillow |
| Analyse | NumPy, SciPy, scikit-learn |
| Rapports | ReportLab (PDF), HTML self-contained |
| Tests | pytest (398 tests, 0 échec) |
| Frontend | HTML/CSS/JS vanilla (3 pages) |
| Python | 3.13.2 |

---

## Installation

### Prérequis

- Python 3.11+
- pip

### Cloner et installer

```bash
git clone https://github.com/<user>/forensic-image-recovery.git
cd forensic-image-recovery

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Lancer le serveur

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

L'interface web est disponible en ouvrant `forensic_ui.html` dans le navigateur.
L'API est accessible sur `http://localhost:8000`.
La documentation interactive Swagger est sur `http://localhost:8000/docs`.

---

## Structure du projet

```
forensic-image-recovery/
├── app/
│   ├── api/
│   │   ├── routes/          # Endpoints FastAPI
│   │   │   ├── pipeline.py      # POST /pipeline/corrupt-and-repair
│   │   │   ├── reconstruction.py # /reconstruction/*
│   │   │   ├── analysis.py      # /analysis/*
│   │   │   ├── report.py        # /reports/*
│   │   │   ├── benchmark.py     # /benchmark/*
│   │   │   └── audit.py         # /audit/logs
│   │   └── schemas/
│   │       ├── requests.py
│   │       └── responses.py
│   ├── core/
│   │   ├── config.py            # Chemins, constantes
│   │   ├── logger.py
│   │   ├── upload_validator.py  # Magic bytes H1
│   │   ├── file_cleanup.py      # Nettoyage auto H3
│   │   └── audit_logger.py      # Logs JSONL H4
│   └── modules/
│       ├── corruption/          # 15 types de dégradation
│       ├── reconstruction/      # Pipeline adaptatif + PatchMatch
│       ├── evaluation/          # Scoring PSNR/SSIM supervisé/aveugle
│       ├── reporting/           # JSON + PDF + HTML + légal
│       ├── carving/             # Extraction JPEG depuis dumps
│       ├── analysis/            # Analyses persistantes
│       ├── benchmark/           # Benchmark automatique
│       ├── metadata/            # Analyse EXIF (K12)
│       ├── similarity/          # Hash perceptuel dHash/pHash (K14)
│       ├── tampering/           # Détection copy-move ORB (K17)
│       ├── ocr/                 # OCR Tesseract/EasyOCR (K11)
│       ├── steganalysis/        # Analyse LSB (K13)
│       ├── noise/               # Bruit résiduel PRNU (K18)
│       ├── fragments/           # Clustering DBSCAN (K15)
│       ├── raw/                 # Chargement RAW optionnel (K10)
│       ├── ai/                  # LaMa adapter optionnel (K9)
│       └── timeline/            # Timeline forensique (K16)
├── data/
│   ├── input/                   # Images uploadées (nettoyage auto >24h)
│   ├── corrupted/               # Images corrompues (>48h)
│   ├── reconstructed/           # Images reconstruites (>48h)
│   ├── masks/                   # Masques de corruption (>48h)
│   ├── reports/                 # Rapports JSON/PDF/HTML
│   ├── analyses/                # Analyses persistantes (>7j)
│   └── logs/                    # audit.jsonl
├── tests/                       # 398 tests pytest
├── scripts/
│   └── generate_fragmented_dataset.py
├── forensic_ui.html             # Interface principale
├── technique.html               # Documentation technique
├── mask_editor.html             # Éditeur de masque
└── requirements.txt
```

---

## Utilisation

### Interface web

1. Ouvrir `forensic_ui.html` dans le navigateur
2. Déposer une image JPEG ou PNG
3. Choisir le type de dégradation et la sévérité
4. Cliquer **Analyser**
5. Consulter le triptyque original/corrompu/reconstruit, le score et les jauges
6. Télécharger le rapport PDF ou ouvrir le rapport HTML

### Éditeur de masque (`mask_editor.html`)

Outil forensique manuel : charger une image corrompue, dessiner la zone abîmée au pinceau, lancer la reconstruction ciblée.

### API REST

```bash
# Pipeline complet
curl -X POST http://localhost:8000/pipeline/corrupt-and-repair \
  -F "image=@photo.jpg" \
  -F "corruption_type=scratch_lines" \
  -F "severity=medium" \
  -F "max_attempts=8"

# Reconstruction avec masque utilisateur
curl -X POST http://localhost:8000/reconstruction/repair-with-mask \
  -F "image=@corrupted.png" \
  -F "mask=@mask.png"

# Lancer une analyse en arrière-plan
curl -X POST http://localhost:8000/analysis/start \
  -F "image=@photo.jpg" \
  -F "corruption_type=zone_deletion" \
  -F "severity=heavy"
```

---

## Types de corruption disponibles

| Type | Description | Cas d'usage forensique |
|------|-------------|------------------------|
| `scratch_lines` | Rayures diagonales | Négatifs rayés, supports optiques |
| `multiple_bars` | Barres horizontales/verticales | Erreurs de transmission |
| `large_deleted_square` | Grand carré supprimé | Perte de cluster disque |
| `random_holes` | Trous aléatoires | Bad blocks |
| `local_noise` | Bruit gaussien local | Dégradation capteur |
| `zone_deletion` | Zone rectangulaire supprimée | Overwrite mémoire |
| `block_dropout` | Blocs JPEG supprimés | Perte de paquets réseau |
| `mixed` | Combinaison multiple | Scénario forensique réel |
| `local_blur` | Flou local | Défaut optique |
| `shift_region` | Décalage de région | Réassemblage incorrect |
| ... | +5 autres types | |

---

## Stratégies de reconstruction

Le moteur adaptatif teste jusqu'à 14 stratégies et sélectionne la meilleure par score :

| Famille | Stratégies | Meilleur pour |
|---------|-----------|----------------|
| Inpainting | `inpainting_r3/r5/r7` | Zones supprimées, barres |
| PatchMatch | `patchmatch_p7/p9/p11` | Grandes zones, textures répétitives |
| Composite | `denoise_then_inpaint`, `inpaint_then_sharpen`, ... | Corruptions mixtes |
| Denoise | `median_blur`, `gaussian`, `bilateral` | Bruit |
| Deblur | `deblur_light/strong` | Flou, artefacts JPEG |
| Block repair | `block_repair` | Blocs JPEG |
| Baseline | `conservative` | Référence |

---

## Scoring

**Mode supervisé** (original disponible) :
```
score = 0.60 × SSIM + 0.20 × PSNR/100 + 0.30 × gain_SSIM + 0.20 × gain_PSNR
```

**Mode aveugle** (sans original) :
Score heuristique basé sur netteté (35%), bruit (25%), continuité des contours (20%), cohérence (10%) + entropie locale et cohérence couleur.

| Score | Qualité |
|-------|---------|
| ≥ 80 | Excellent — zones indétectables |
| 60–79 | Bon — légères traces |
| 40–59 | Moyen — reconstruction partielle |
| < 40 | Faible — corruption trop étendue |

---

## Modules forensiques avancés

| Module | Fonctionnalité |
|--------|---------------|
| K8 PatchMatch | Inpainting exemplar-based sans GPU |
| K12 EXIF | Détection logiciels retouche, GPS, incohérences dates |
| K14 pHash | Comparaison perceptuelle dHash/pHash + distance Hamming |
| K17 Copy-Move | Détection zones copiées-collées (ORB + BFMatcher) |
| K11 OCR | Extraction texte avant/après (Tesseract/EasyOCR) |
| K19 Rapport légal | Chain of custody, SHA-256, disclaimer juridique |
| K13 LSB | Détection stéganographie par analyse bits de poids faible |
| K18 PRNU | Cohérence bruit résiduel + heatmap + corrélation capteur |
| K15 Clustering | Regroupement fragments par DBSCAN (107 features) |
| K10 RAW | Chargement DNG/CR2/NEF/ARW (optionnel, rawpy) |
| K9 LaMa | Deep inpainting optionnel (LAMA_ENABLED=true + torch) |
| K16 Timeline | Chronologie forensique des événements d'analyse |

---

## Sécurité

- **H1** Magic bytes : validation JPEG (`\xff\xd8\xff`) et PNG (`\x89PNG`) avant traitement
- **H3** Nettoyage automatique au démarrage : input >24h, corrupted/masks/reconstructed >48h, analyses >7j
- **H4** Audit logging JSONL : chaque requête loggée avec `request_id`, IP, SHA-256, temps, statut
- **H2** `GET /files/serve` restreint aux dossiers `data/` uniquement

---

## Tests

```bash
# Tous les tests
pytest

# Tests par catégorie
pytest tests/test_e2e.py          # End-to-end (8 scénarios)
pytest tests/test_patchmatch.py   # K8 PatchMatch
pytest tests/test_security.py     # H1+H3+H4
pytest tests/test_benchmark.py    # Benchmark

# Avec verbose
pytest -v --tb=short
```

**398 tests, 0 échec** — couvrant pipeline, sécurité, scoring, modules forensiques, E2E.

---

## Génération de datasets forensiques

```bash
# Générer un dataset fragmenté avec perte et bruit
python scripts/generate_fragmented_dataset.py \
    --images data/input/photo.jpeg \
    --output data/dumps/test_dataset \
    --fragments 6 \
    --shuffle \
    --loss-ratio 0.2 \
    --noise
```

Génère : dump binaire synthétique + `ground_truth.json` avec SHA-256, offsets, métadonnées de chaque fragment.

---

## Endpoints principaux

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| POST | `/pipeline/corrupt-and-repair` | Pipeline complet |
| POST | `/reconstruction/repair-with-mask` | Masque utilisateur |
| POST | `/reconstruction/compare-masks` | Auto vs utilisateur |
| POST | `/analysis/start` | Analyse async |
| GET | `/analysis/{id}/status` | Statut |
| GET | `/analysis/{id}/result` | Résultat |
| GET | `/reports/html/{id}` | Rapport HTML inline |
| GET | `/reports/pdf/{id}` | Rapport PDF |
| GET | `/audit/logs` | Logs d'audit |
| POST | `/benchmark/run` | Benchmark |
| GET | `/health` | Health check |

Documentation complète : `http://localhost:8000/docs`

---

## Configuration optionnelle

```bash
# Activer LaMa deep inpainting (nécessite torch + modèle)
LAMA_ENABLED=true uvicorn app.main:app --reload
```

---

## Auteur

Ravin THILAGARASA
Projet Annuel ESGI 4ème année — 2025/2026

---

## Avertissement forensique

Les reconstructions produites par ce système constituent des **hypothèses visuelles algorithmiques** et ne sauraient constituer des preuves judiciaires. Toute utilisation dans un contexte légal nécessite la validation d'un expert forensique habilité.