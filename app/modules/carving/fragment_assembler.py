"""app/modules/carving/fragment_assembler.py — G3+G4.

Reconstruction de fragments par overlap matching, rolling hash,
score de continuité, greedy chaining. Rapport d'offsets inclus.
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logger import logger


# ---------------------------------------------------------------------------
# G3 — Algorithmes d'assemblage
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rolling_hash(data: bytes, window: int = 64) -> list[int]:
    """Calcule un rolling hash de Rabin sur les fenêtres du fragment."""
    if len(data) < window:
        return [hash(data)]
    hashes = []
    for i in range(len(data) - window + 1):
        hashes.append(hash(data[i:i + window]))
    return hashes


def _rolling_hash_similarity(tail: bytes, head: bytes, window: int) -> float:
    tail_hashes = set(_rolling_hash(tail, window=window))
    head_hashes = set(_rolling_hash(head, window=window))
    if not tail_hashes or not head_hashes:
        return 0.0
    common = len(tail_hashes & head_hashes)
    total = max(len(tail_hashes), len(head_hashes), 1)
    return float(common / total)


def _overlap_score(tail: bytes, head: bytes, overlap: int = 64) -> float:
    """Score de continuité entre la fin d'un fragment et le début du suivant.

    Compare les overlap octets de la queue avec ceux de la tête.
    Retourne un score entre 0.0 (aucune continuité) et 1.0 (continuité parfaite).
    """
    if len(tail) < overlap or len(head) < overlap:
        return 0.0
    tail_end = np.frombuffer(tail[-overlap:], dtype=np.uint8).astype(np.float32)
    head_start = np.frombuffer(head[:overlap], dtype=np.uint8).astype(np.float32)
    mse = float(np.mean((tail_end - head_start) ** 2))
    mse_score = float(max(0.0, 1.0 - mse / (255.0 ** 2)))
    window = max(8, min(64, overlap // 2))
    hash_score = _rolling_hash_similarity(tail[-overlap:], head[:overlap], window)
    return float(0.65 * mse_score + 0.35 * hash_score)


def _continuity_score(frag_a: bytes, frag_b: bytes) -> float:
    """Score composite de continuité entre deux fragments binaires."""
    # 1. Overlap sur 64 octets
    overlap = _overlap_score(frag_a, frag_b, overlap=64)
    # 2. Cohérence de taille (fragments proches en taille = plus vraisemblable)
    size_ratio = min(len(frag_a), len(frag_b)) / max(len(frag_a), len(frag_b), 1)
    return 0.7 * overlap + 0.3 * size_ratio


def greedy_chain(
    fragments: list[dict[str, Any]],
    overlap: int = 64,
) -> list[dict[str, Any]]:
    """Assemble les fragments par chaînage glouton.

    À chaque étape, choisit le fragment suivant avec le meilleur score
    de continuité par rapport au dernier fragment assemblé.

    Args:
        fragments: liste de dicts avec 'data' (bytes), 'index', 'offset'
        overlap: taille de la fenêtre de chevauchement en octets

    Returns:
        Liste de fragments dans l'ordre assemblé, avec 'continuity_score'
    """
    if not fragments:
        return []

    remaining = list(fragments)
    # Trier par offset pour partir du début
    remaining.sort(key=lambda f: f.get("offset", 0))
    chain = [remaining.pop(0)]
    chain[0]["continuity_score"] = 1.0

    while remaining:
        last_data = chain[-1].get("data", b"")
        best_score = -1.0
        best_idx = 0

        for i, frag in enumerate(remaining):
            score = _continuity_score(last_data, frag.get("data", b""))
            if score > best_score:
                best_score = score
                best_idx = i

        next_frag = remaining.pop(best_idx)
        next_frag["continuity_score"] = float(best_score)
        chain.append(next_frag)

    return chain


def assemble_fragments(
    chain: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Assemble les fragments dans l'ordre de la chaîne et écrit le fichier résultant.

    Returns:
        dict avec path, size, sha256, fragment_count, assembly_score
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assembled = b"".join(f.get("data", b"") for f in chain)

    with output_path.open("wb") as f:
        f.write(assembled)

    assembly_score = float(np.mean([
        f.get("continuity_score", 0.0) for f in chain
    ])) if chain else 0.0

    return {
        "path":            str(output_path),
        "size":            len(assembled),
        "sha256":          _sha256(assembled),
        "fragment_count":  len(chain),
        "assembly_score":  assembly_score,
    }


# ---------------------------------------------------------------------------
# G4 — Rapport d'offsets et de fragments
# ---------------------------------------------------------------------------

def build_fragment_report(
    source_dump: str | Path,
    fragments: list[dict[str, Any]],
    chain: list[dict[str, Any]],
    assembly_result: dict[str, Any],
    rejected_fragments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construit le rapport d'analyse de fragments (G4).

    Contenu :
        offsets             : liste des offsets de chaque fragment
        fragments_used      : fragments retenus dans la chaîne
        fragments_rejected  : fragments écartés
        assembly            : résultat de l'assemblage
        score               : score moyen de continuité
        sha256_dump         : hash du dump source
    """
    source_path = Path(source_dump)
    dump_sha256 = ""
    if source_path.exists():
        dump_sha256 = _sha256(source_path.read_bytes())

    offsets = [
        {
            "index":  f.get("index"),
            "offset": f.get("offset", 0),
            "size":   len(f.get("data", b"")),
            "sha256": _sha256(f.get("data", b"")),
        }
        for f in fragments
    ]

    fragments_used = [
        {
            "index":             f.get("index"),
            "offset":            f.get("offset", 0),
            "size":              len(f.get("data", b"")),
            "continuity_score":  f.get("continuity_score", 0.0),
            "sha256":            _sha256(f.get("data", b"")),
        }
        for f in chain
    ]

    fragments_rejected_out = [
        {
            "index":  f.get("index"),
            "offset": f.get("offset", 0),
            "size":   len(f.get("data", b"")),
            "reason": f.get("reject_reason", "unknown"),
            "sha256": _sha256(f.get("data", b"")),
        }
        for f in (rejected_fragments or [])
    ]

    return {
        "source_dump":        str(source_dump),
        "sha256_dump":        dump_sha256,
        "total_fragments":    len(fragments),
        "used_fragments":     len(chain),
        "rejected_fragments": len(fragments_rejected_out),
        "offsets":            offsets,
        "fragments_used":     fragments_used,
        "fragments_rejected": fragments_rejected_out,
        "assembly":           assembly_result,
        "assembly_score":     assembly_result.get("assembly_score", 0.0),
    }
