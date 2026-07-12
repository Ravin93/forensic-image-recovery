"""scripts/generate_fragmented_dataset.py — G1+G2.

Génère un dataset forensique fragmenté à partir d'images sources :
  - Fragmentation contrôlée (taille, nombre, shuffle, perte)
  - Bruit entre fragments
  - Mélange de fragments de plusieurs images
  - Dump synthétique binaire
  - ground_truth.json avec les métadonnées

Usage :
    python scripts/generate_fragmented_dataset.py \
        --images data/input/demo_real.jpeg \
        --output data/dumps/fragmented_dataset \
        --fragments 6 --shuffle --loss-ratio 0.2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_image_bytes(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.read()


def _fragment_bytes(
    data: bytes,
    n_fragments: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Divise des données en n fragments de taille approximativement égale.

    Les coupures sont légèrement randomisées (±10%) pour simuler
    des conditions réelles de fragmentation disque.
    """
    size = len(data)
    if n_fragments <= 0 or size == 0:
        return []

    base_size = size // n_fragments
    fragments = []
    offset = 0

    for i in range(n_fragments):
        # Variation de ±10% de la taille de base
        variation = int(base_size * 0.10)
        frag_size = base_size + rng.randint(-variation, variation)
        frag_size = max(64, frag_size)

        if i == n_fragments - 1:
            frag_size = size - offset  # dernier fragment = reste

        end = min(offset + frag_size, size)
        frag_data = data[offset:end]

        fragments.append({
            "index":           i,
            "offset":          offset,
            "size":            len(frag_data),
            "data":            frag_data,
            "sha256":          _sha256(frag_data),
            "source_image":    "",
            "original_offset": offset,
        })
        offset = end
        if offset >= size:
            break

    return fragments


def _add_noise_between_fragments(
    dump: bytearray,
    frag_boundaries: list[int],
    noise_size: int,
    rng: random.Random,
) -> bytearray:
    """Insère du bruit aléatoire entre les fragments dans le dump."""
    result = bytearray()
    prev = 0
    offsets_shift = 0

    for boundary in sorted(frag_boundaries):
        result.extend(dump[prev:boundary])
        # Insérer du bruit
        noise = bytes([rng.randint(0, 255) for _ in range(noise_size)])
        result.extend(noise)
        prev = boundary

    result.extend(dump[prev:])
    return result


def generate_fragmented_dataset(
    image_paths: list[Path],
    output_dir: Path,
    n_fragments: int = 6,
    shuffle: bool = True,
    loss_ratio: float = 0.0,
    noise_between: bool = False,
    noise_size: int = 512,
    mix_images: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Génère le dataset fragmenté complet.

    G1 — Dataset de base avec fragments, dump, ground_truth.json
    G2 — Options avancées : shuffle, perte, bruit, mélange multi-images

    Args:
        image_paths:  chemins des images sources
        output_dir:   répertoire de sortie
        n_fragments:  nombre de fragments par image
        shuffle:      mélanger les fragments
        loss_ratio:   fraction de fragments à supprimer (0.0–0.8)
        noise_between: insérer du bruit entre les fragments
        noise_size:   taille du bruit en octets
        mix_images:   mélanger les fragments de plusieurs images
        seed:         graine aléatoire

    Returns:
        dict ground_truth avec tous les métadonnées
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fragments_dir = output_dir / "fragments"
    fragments_dir.mkdir(exist_ok=True)

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    all_fragments: list[dict[str, Any]] = []
    source_info: list[dict[str, Any]] = []

    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"[WARN] Image introuvable : {img_path}, ignorée")
            continue

        data = _read_image_bytes(img_path)
        fragments = _fragment_bytes(data, n_fragments, rng)

        for f in fragments:
            f["source_image"] = img_path.name

        source_info.append({
            "path":       str(img_path),
            "name":       img_path.name,
            "size":       len(data),
            "sha256":     _sha256(data),
            "n_fragments": len(fragments),
        })
        all_fragments.extend(fragments)

    if not all_fragments:
        raise ValueError("Aucun fragment généré — vérifiez les chemins des images")

    # G2 — Mélange multi-images
    if mix_images and len(image_paths) > 1:
        rng.shuffle(all_fragments)

    # G2 — Shuffle
    if shuffle:
        rng.shuffle(all_fragments)

    # G2 — Perte de fragments
    loss_ratio = float(np.clip(loss_ratio, 0.0, 0.8))
    n_lost = int(len(all_fragments) * loss_ratio)
    lost_indices = set(rng.sample(range(len(all_fragments)), n_lost)) if n_lost > 0 else set()

    retained_fragments: list[dict[str, Any]] = []
    lost_fragments: list[dict[str, Any]] = []

    for i, frag in enumerate(all_fragments):
        if i in lost_indices:
            frag["reject_reason"] = "simulated_loss"
            lost_fragments.append(frag)
        else:
            retained_fragments.append(frag)

    # G2 — Bruit entre fragments
    dump = bytearray()
    dump_offsets: list[dict[str, Any]] = []

    for frag in retained_fragments:
        if noise_between and dump:
            noise = bytes([rng.randint(0, 255) for _ in range(noise_size)])
            dump.extend(noise)

        dump_offset = len(dump)
        dump.extend(frag["data"])
        dump_offsets.append({
            "fragment_index":   frag["index"],
            "source_image":     frag["source_image"],
            "dump_offset":      dump_offset,
            "original_offset":  frag["original_offset"],
            "size":             frag["size"],
            "sha256":           frag["sha256"],
        })

    # Écriture du dump synthétique
    ts = int(time.time())
    dump_path = output_dir / f"synthetic_dump_{ts}.bin"
    with dump_path.open("wb") as f:
        f.write(dump)

    # Écriture des fragments individuels (pour debug/test)
    for i, frag in enumerate(retained_fragments):
        frag_path = fragments_dir / f"fragment_{i:03d}_{frag['source_image']}.bin"
        with frag_path.open("wb") as f:
            f.write(frag["data"])
        frag["fragment_file"] = str(frag_path)

    # Ground truth JSON
    ground_truth = {
        "generated_at":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed":              seed,
        "sources":           source_info,
        "config": {
            "n_fragments":   n_fragments,
            "shuffle":       shuffle,
            "loss_ratio":    loss_ratio,
            "noise_between": noise_between,
            "noise_size":    noise_size if noise_between else 0,
            "mix_images":    mix_images,
        },
        "dump": {
            "path":          str(dump_path),
            "size":          len(dump),
            "sha256":        _sha256(bytes(dump)),
        },
        "fragments": {
            "total":         len(all_fragments),
            "retained":      len(retained_fragments),
            "lost":          len(lost_fragments),
            "offsets":       dump_offsets,
        },
        "lost_fragments":    [
            {
                "index":          f["index"],
                "source_image":   f["source_image"],
                "original_offset": f["original_offset"],
                "size":           f["size"],
                "sha256":         f["sha256"],
            }
            for f in lost_fragments
        ],
    }

    gt_path = output_dir / "ground_truth.json"
    with gt_path.open("w", encoding="utf-8") as f:
        import json as _json
        _json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"[OK] Dataset généré dans : {output_dir}")
    print(f"     Fragments : {len(retained_fragments)} retenus / {len(all_fragments)} total")
    print(f"     Perdus    : {len(lost_fragments)}")
    print(f"     Dump      : {dump_path.name} ({len(dump):,} octets)")
    print(f"     GT        : {gt_path.name}")

    return ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère un dataset forensique fragmenté",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--images", nargs="+", required=True,
                        help="Chemins des images sources")
    parser.add_argument("--output", default="data/dumps/fragmented_dataset",
                        help="Répertoire de sortie")
    parser.add_argument("--fragments", type=int, default=6,
                        help="Nombre de fragments par image")
    parser.add_argument("--shuffle", action="store_true",
                        help="Mélanger les fragments")
    parser.add_argument("--loss-ratio", type=float, default=0.0,
                        help="Fraction de fragments perdus (0.0–0.8)")
    parser.add_argument("--noise", action="store_true",
                        help="Ajouter du bruit entre les fragments")
    parser.add_argument("--noise-size", type=int, default=512,
                        help="Taille du bruit en octets")
    parser.add_argument("--mix", action="store_true",
                        help="Mélanger les fragments de plusieurs images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    image_paths = [Path(p) for p in args.images]
    generate_fragmented_dataset(
        image_paths=image_paths,
        output_dir=Path(args.output),
        n_fragments=args.fragments,
        shuffle=args.shuffle,
        loss_ratio=args.loss_ratio,
        noise_between=args.noise,
        noise_size=args.noise_size,
        mix_images=args.mix,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()