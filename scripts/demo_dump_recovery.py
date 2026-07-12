"""scripts/demo_dump_recovery.py — Démo forensique : dump binaire → carving JPEG → validation.

Scénario :
  1. Génère un dump binaire synthétique contenant 3 images JPEG
     entourées de garbage bytes (simulation d'un secteur disque partiellement écrasé).
  2. Lance le carving JPEG (détection marqueurs SOI/EOI).
  3. Valide et affiche les résultats.
  4. (Optionnel) envoie la première image extraite au pipeline de reconstruction.

Usage :
    python scripts/demo_dump_recovery.py
    python scripts/demo_dump_recovery.py --source data/dumps/test_dataset/demo_source.jpeg
    python scripts/demo_dump_recovery.py --no-api        # skip la reconstruction API
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

# Résoudre les imports depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ── Palette de couleurs pour la console ─────────────────────────────────────
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def ok(msg: str)   -> None: print(f"  {GREEN}✓{RESET} {msg}")
def info(msg: str) -> None: print(f"  {CYAN}→{RESET} {msg}")
def warn(msg: str) -> None: print(f"  {YELLOW}⚠{RESET} {msg}")
def err(msg: str)  -> None: print(f"  {RED}✗{RESET} {msg}")
def step(n: int, title: str) -> None:
    print(f"\n{BOLD}[{n}/4] {title}{RESET}")
    print(f"  {'─' * (len(title) + 6)}")


# ── Génération du dump binaire ───────────────────────────────────────────────

def _make_jpeg_bytes(width: int, height: int, color: tuple, label: str) -> bytes:
    """Crée un JPEG en mémoire avec une étiquette texte."""
    img = Image.new("RGB", (width, height), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 4, width - 4, height - 4], outline="white", width=2)
    draw.text((10, 10), label, fill="white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def generate_dump(output_path: Path, source_image: Path | None = None) -> dict:
    """Génère un dump binaire synthétique avec 3 JPEG et du bruit entre eux.

    Retourne les métadonnées ground-truth : offsets, tailles, sha256.
    """
    rng = np.random.default_rng(42)

    def garbage(size: int) -> bytes:
        return bytes(rng.integers(0, 256, size, dtype=np.uint8).tolist())

    images: list[bytes] = []

    if source_image and source_image.exists():
        with source_image.open("rb") as f:
            images.append(f.read())
        info(f"Image source chargée : {source_image.name} ({len(images[0]):,} octets)")
    else:
        images.append(_make_jpeg_bytes(320, 240, (30, 120, 80), "Fragment #1 — Forensic Demo"))

    images.append(_make_jpeg_bytes(200, 150, (120, 60, 30), "Fragment #2 — Forensic Demo"))
    images.append(_make_jpeg_bytes(160, 160, (40, 80, 160), "Fragment #3 — Forensic Demo"))

    # Construction du dump : garbage | img1 | garbage | img2 | garbage | img3 | garbage
    offsets = []
    dump = bytearray()

    dump += garbage(rng.integers(512, 2048))

    for i, jpeg in enumerate(images):
        offsets.append({"index": i + 1, "start": len(dump), "size": len(jpeg)})
        dump += jpeg
        if i < len(images) - 1:
            dump += garbage(rng.integers(256, 1024))

    dump += garbage(rng.integers(512, 1024))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(dump))

    return {
        "dump_size": len(dump),
        "n_images": len(images),
        "ground_truth": offsets,
    }


# ── Carving ──────────────────────────────────────────────────────────────────

def run_carving(dump_path: Path, output_dir: Path) -> list[dict]:
    """Lance le carving JPEG via le module existant."""
    from app.modules.carving.extractor import extract_jpegs_from_dump
    from app.core.config import ensure_directories
    ensure_directories()

    results = extract_jpegs_from_dump(dump_path)
    return results


# ── Validation ───────────────────────────────────────────────────────────────

def validate_extracted(results: list[dict], ground_truth: list[dict]) -> None:
    """Vérifie que les JPEG extraits sont des images valides et affiche les stats."""
    import hashlib

    for item in results:
        path = Path(item["path"])
        if not path.exists():
            warn(f"Fichier absent : {path}")
            continue

        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()[:12]

        try:
            img = Image.open(path)
            img.verify()
            ok(f"JPEG #{item['index']}  {item['size']:>7,} octets  offset={item['start_offset']}  sha256={sha}…  {img.size[0]}×{img.size[1]}px")
        except Exception as e:
            warn(f"JPEG #{item['index']} invalide : {e}")


# ── Appel API optionnel ───────────────────────────────────────────────────────

def call_reconstruction_api(image_path: Path, api_base: str = "http://localhost:8000") -> None:
    """Envoie la première image extraite au pipeline de reconstruction pour démo."""
    import urllib.request
    import urllib.error

    # Vérification que l'API est joignable
    try:
        with urllib.request.urlopen(f"{api_base}/health", timeout=3) as r:
            health = json.loads(r.read())
        if health.get("status") != "ok":
            warn("API répond mais status != ok")
            return
    except Exception:
        warn(f"API non joignable sur {api_base} — reconstruction skippée")
        warn("Lancez d'abord : uvicorn app.main:app --reload")
        return

    info(f"API joignable — envoi de {image_path.name} au pipeline…")

    import urllib.parse

    boundary = "----ForensicDemoBoundary"

    with image_path.open("rb") as f:
        img_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + img_data + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="corruption_type"\r\n\r\n'
        f"scratch_lines"
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="severity"\r\n\r\n'
        f"medium"
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="execution_mode"\r\n\r\n'
        f"assisted"
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="max_attempts"\r\n\r\n'
        f"6"
        f"\r\n--{boundary}--\r\n"
    ).encode()

    req = urllib.request.Request(
        f"{api_base}/pipeline/corrupt-and-repair",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read())
        elapsed = time.time() - t0

        score = result.get("score", 0)
        strategy = result.get("selected_repair_strategy", "?")
        ok(f"Reconstruction terminée en {elapsed:.1f}s")
        ok(f"Score : {score:.1f}/100  |  Stratégie : {strategy}")
        if result.get("report_id"):
            info(f"Rapport disponible : {api_base}/reports/html/{result['report_id']}")
    except urllib.error.HTTPError as e:
        err(f"Erreur API {e.code} : {e.read().decode()[:200]}")
    except Exception as e:
        err(f"Erreur inattendue : {e}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Démo forensique : dump binaire → carving JPEG → reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python scripts/demo_dump_recovery.py
  python scripts/demo_dump_recovery.py --source data/dumps/test_dataset/demo_source.jpeg
  python scripts/demo_dump_recovery.py --no-api
        """,
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Image JPEG source à insérer dans le dump (optionnel)",
    )
    parser.add_argument(
        "--dump", type=Path, default=Path("data/dumps/test_dataset/demo.bin"),
        help="Chemin du dump à générer (défaut: data/dumps/test_dataset/demo.bin)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/extracted"),
        help="Dossier de sortie pour les JPEG extraits",
    )
    parser.add_argument(
        "--no-api", action="store_true",
        help="Ne pas appeler l'API de reconstruction",
    )
    parser.add_argument(
        "--api", default="http://localhost:8000",
        help="Base URL de l'API FastAPI (défaut: http://localhost:8000)",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{'═' * 58}{RESET}")
    print(f"{BOLD}  Forensic Image Recovery — Démo Dump Carving{RESET}")
    print(f"{DIM}  ESGI Projet Annuel 2026 — Pipeline forensique{RESET}")
    print(f"{BOLD}{'═' * 58}{RESET}")

    # ── Étape 1 : Génération du dump ──
    step(1, "Génération du dump binaire synthétique")
    source = args.source or Path("data/dumps/test_dataset/demo_source.jpeg")
    metadata = generate_dump(args.dump, source_image=source)

    ok(f"Dump généré : {args.dump}  ({metadata['dump_size']:,} octets)")
    ok(f"{metadata['n_images']} images JPEG insérées parmi des octets aléatoires")
    for gt in metadata["ground_truth"]:
        info(f"  Fragment #{gt['index']} : offset={gt['start']}  taille={gt['size']} octets")

    # ── Étape 2 : Carving ──
    step(2, "Carving JPEG (détection marqueurs SOI/EOI)")
    t0 = time.time()
    try:
        results = run_carving(args.dump, args.output_dir)
        elapsed = time.time() - t0
        ok(f"{len(results)} fichier(s) JPEG extrait(s) en {elapsed:.2f}s")
    except Exception as e:
        err(f"Carving échoué : {e}")
        sys.exit(1)

    if not results:
        warn("Aucun JPEG détecté dans le dump.")
        sys.exit(1)

    # ── Étape 3 : Validation ──
    step(3, "Validation des JPEG extraits")
    validate_extracted(results, metadata["ground_truth"])

    expected = metadata["n_images"]
    found = len(results)
    if found == expected:
        ok(f"Recall parfait : {found}/{expected} images récupérées")
    elif found > 0:
        warn(f"Récupération partielle : {found}/{expected} images trouvées")
    else:
        err("Aucune image récupérée — vérifiez les marqueurs SOI/EOI")

    # ── Étape 4 : Reconstruction API (optionnel) ──
    step(4, "Reconstruction via API FastAPI")
    if args.no_api:
        info("--no-api : reconstruction skippée")
        info(f"Pour lancer manuellement : uvicorn app.main:app --reload")
        info(f"Puis ouvrir forensic_ui.html dans le navigateur")
    else:
        first_extracted = Path(results[0]["path"])
        if first_extracted.exists():
            call_reconstruction_api(first_extracted, args.api)
        else:
            warn("Premier fichier extrait introuvable pour la reconstruction")

    # ── Résumé ──
    print(f"\n{BOLD}{'─' * 58}{RESET}")
    print(f"{BOLD}  Résumé{RESET}")
    print(f"{'─' * 58}")
    print(f"  Dump analysé    : {args.dump}")
    print(f"  Images extraites: {len(results)}/{metadata['n_images']}")
    print(f"  Sortie          : {args.output_dir}/")
    if not args.no_api:
        print(f"  API             : {args.api}/docs")
    print(f"\n{DIM}  Le système ne prétend pas reconstruire une vérité forensique{RESET}")
    print(f"{DIM}  absolue. Il produit une reconstruction plausible, mesurable,{RESET}")
    print(f"{DIM}  et documentée, avec un score de confiance et des limites explicites.{RESET}")
    print(f"{BOLD}{'═' * 58}{RESET}\n")


if __name__ == "__main__":
    main()
