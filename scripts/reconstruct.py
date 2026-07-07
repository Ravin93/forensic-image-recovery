"""CLI de reconstruction forensique depuis dump ou fragments.

Usage:
    python scripts/reconstruct.py --input data/dumps/test.dd --out output/ --formats jpeg --report json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import permutations
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.modules.carving.extractor import extract_jpegs_from_dump, extract_pngs_from_dump
from app.modules.carving.fragment_assembler import build_fragment_report, greedy_chain
from app.modules.carving.jpeg_scanner import scan_jpeg_offsets, scan_png_offsets
from app.modules.validation.verifier import verify_image


def _parse_formats(raw: str) -> list[str]:
    formats = [fmt.strip().lower() for fmt in raw.split(",") if fmt.strip()]
    return formats or ["jpeg"]


def _allowed_image_formats(formats: list[str]) -> list[str]:
    aliases = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG"}
    return [aliases.get(fmt, fmt.upper()) for fmt in formats]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fragments_from_extracted(extracted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    for item in extracted:
        path = Path(str(item.get("path", "")))
        if not path.exists():
            continue
        data = path.read_bytes()
        fragments.append({
            "index": item.get("index", len(fragments) + 1),
            "offset": item.get("start_offset", 0),
            "data": data,
            "path": str(path),
            "size": len(data),
            "sha256": _sha256(data),
        })
    return fragments


def _fragments_from_directory(input_dir: Path) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    for idx, path in enumerate(sorted(p for p in input_dir.iterdir() if p.is_file()), start=1):
        data = path.read_bytes()
        fragments.append({
            "index": idx,
            "offset": idx - 1,
            "data": data,
            "path": str(path),
            "size": len(data),
            "sha256": _sha256(data),
        })
    return fragments


def _write_chain_candidate(chain: list[dict[str, Any]], out_dir: Path, formats: list[str]) -> dict[str, Any]:
    preferred = formats[0] if formats else "jpeg"
    suffix = ".png" if preferred == "png" else ".jpg"
    output_path = out_dir / f"reconstructed{suffix}"
    assembled = b"".join(fragment.get("data", b"") for fragment in chain)
    output_path.write_bytes(assembled)
    scores = [float(fragment.get("continuity_score", 0.0)) for fragment in chain]
    return {
        "path": str(output_path),
        "size": len(assembled),
        "sha256": _sha256(assembled),
        "fragment_count": len(chain),
        "assembly_score": sum(scores) / len(scores) if scores else 0.0,
    }


def _verify_candidate(path: Path, allowed_formats: list[str]) -> dict[str, Any]:
    return {
        "path": str(path),
        **verify_image(path, allowed_formats=allowed_formats),
    }


def _find_report_source(input_path: Path) -> Path:
    if input_path.is_dir():
        return list(input_path.glob("*.bin"))[0]
    return input_path


def _try_valid_permutation(
    fragments: list[dict[str, Any]],
    out_dir: Path,
    formats: list[str],
    allowed_formats: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]] | None:
    if len(fragments) > 4:
        return None

    for order in permutations(fragments):
        candidate_chain = [dict(fragment) for fragment in order]
        if candidate_chain:
            candidate_chain[0]["continuity_score"] = 1.0
        assembly_result = _write_chain_candidate(candidate_chain, out_dir, formats)
        validation = _verify_candidate(Path(assembly_result["path"]), allowed_formats)
        if validation["valid"]:
            return candidate_chain, assembly_result, validation

    return None


def reconstruct(input_path: Path, out_dir: Path, formats: list[str]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed_formats = _allowed_image_formats(formats)

    offsets: list[int] = []
    extracted: list[dict[str, Any]] = []
    if input_path.is_file():
        if "jpeg" in formats or "jpg" in formats:
            jpeg_offsets = scan_jpeg_offsets(input_path)
            offsets.extend(jpeg_offsets)
            extracted.extend(extract_jpegs_from_dump(input_path))
        if "png" in formats:
            png_offsets = scan_png_offsets(input_path)
            offsets.extend(png_offsets)
            extracted.extend(extract_pngs_from_dump(input_path))
        offsets.sort()
        extracted.sort(key=lambda item: int(item.get("start_offset", 0)))
        fragments = _fragments_from_extracted(extracted)
    elif input_path.is_dir():
        fragments = _fragments_from_directory(input_path)
    else:
        raise FileNotFoundError(f"Entrée introuvable : {input_path}")

    chain = greedy_chain(fragments)

    validation_targets = [Path(item["path"]) for item in extracted if item.get("path")]
    assembly_result: dict[str, Any]
    if input_path.is_dir() or not validation_targets:
        assembly_result = _write_chain_candidate(chain, out_dir, formats)
        validation = _verify_candidate(Path(assembly_result["path"]), allowed_formats)
        if not validation["valid"]:
            permutation_result = _try_valid_permutation(fragments, out_dir, formats, allowed_formats)
            if permutation_result is not None:
                chain, assembly_result, validation = permutation_result
        validations = [validation]
    else:
        scores = [float(fragment.get("continuity_score", 0.0)) for fragment in chain]
        assembly_result = {
            "path": str(validation_targets[0]) if validation_targets else "",
            "size": sum(int(fragment.get("size", 0)) for fragment in fragments),
            "sha256": "",
            "fragment_count": len(chain),
            "assembly_score": sum(scores) / len(scores) if scores else 0.0,
        }
        validations = [_verify_candidate(path, allowed_formats) for path in validation_targets]

    fragment_report = build_fragment_report(
        _find_report_source(input_path),
        fragments,
        chain,
        assembly_result,
    )

    return {
        "input": str(input_path),
        "formats": formats,
        "offsets": offsets,
        "extracted": extracted,
        "validations": validations,
        "fragment_report": fragment_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruit des images depuis un dump ou un dossier de fragments.")
    parser.add_argument("--input", required=True, help="Chemin dump .dd/.bin ou dossier de fragments")
    parser.add_argument("--out", required=True, help="Dossier de sortie")
    parser.add_argument("--formats", default="jpeg", help="Formats autorisés, séparés par virgule (défaut: jpeg)")
    parser.add_argument("--report", default="json", choices=["json"], help="Format du rapport (défaut: json)")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    formats = _parse_formats(args.formats)

    report = reconstruct(input_path, out_dir, formats)
    report_path = out_dir / "reconstruction_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
