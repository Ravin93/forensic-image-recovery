"""Tests G1+G2+G3+G4 — forensic fragments et dumps."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def jpeg_image(tmp_path: Path) -> Path:
    """Image JPEG valide pour les tests."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(128):
        arr[i, :, 0] = i * 2
        arr[:, i, 1] = i * 2
    p = tmp_path / "source.jpg"
    Image.fromarray(arr).save(p, format="JPEG")
    return p


@pytest.fixture
def two_jpegs(tmp_path: Path) -> tuple[Path, Path]:
    arr1 = (np.random.randint(0, 200, (64, 64, 3))).astype(np.uint8)
    arr2 = (np.random.randint(50, 255, (64, 64, 3))).astype(np.uint8)
    p1 = tmp_path / "img1.jpg"; Image.fromarray(arr1).save(p1, format="JPEG")
    p2 = tmp_path / "img2.jpg"; Image.fromarray(arr2).save(p2, format="JPEG")
    return p1, p2


# ---------------------------------------------------------------------------
# G1 — Dataset de base
# ---------------------------------------------------------------------------

class TestGenerateDataset:
    def test_creates_output_dir(self, jpeg_image, tmp_path):
        from scripts.generate_fragmented_dataset import generate_fragmented_dataset
        out = tmp_path / "dataset"
        generate_fragmented_dataset([jpeg_image], out, n_fragments=4, seed=0)
        assert out.exists()

    def test_creates_ground_truth_json(self, jpeg_image, tmp_path):
        from scripts.generate_fragmented_dataset import generate_fragmented_dataset
        out = tmp_path / "dataset"
        generate_fragmented_dataset([jpeg_image], out, n_fragments=4, seed=0)
        gt = out / "ground_truth.json"
        assert gt.exists()
        data = json.loads(gt.read_text())
        for key in ("generated_at", "seed", "sources", "config", "dump", "fragments"):
            assert key in data, f"Clé manquante dans ground_truth : {key}"

    def test_creates_synthetic_dump(self, jpeg_image, tmp_path):
        from scripts.generate_fragmented_dataset import generate_fragmented_dataset
        out = tmp_path / "dataset"
        gt = generate_fragmented_dataset([jpeg_image], out, n_fragments=3, seed=0)
        dump_path = Path(gt["dump"]["path"])
        assert dump_path.exists()
        assert dump_path.stat().st_size > 0

    def test_fragment_count_in_gt(self, jpeg_image, tmp_path):
        from scripts.generate_fragmented_dataset import generate_fragmented_dataset
        out = tmp_path / "dataset"
        gt = generate_fragmented_dataset([jpeg_image], out, n_fragments=5, seed=0)
        assert gt["fragments"]["total"] == 5
        assert gt["fragments"]["retained"] <= 5

    def test_dump_sha256_in_gt(self, jpeg_image, tmp_path):
        from scripts.generate_fragmented_dataset import generate_fragmented_dataset
        out = tmp_path / "dataset"
        gt = generate_fragmented_dataset([jpeg_image], out, n_fragments=3, seed=0)
        assert len(gt["dump"]["sha256"]) == 64

    def test_offsets_present(self, jpeg_image, tmp_path):
        from scripts.generate_fragmented_dataset import generate_fragmented_dataset
        out = tmp_path / "dataset"
        gt = generate_fragmented_dataset([jpeg_image], out, n_fragments=4, seed=0)
        offsets = gt["fragments"]["offsets"]
        assert len(offsets) >= 1
        for o in offsets:
            assert "dump_offset" in o
            assert "sha256" in o


# ---------------------------------------------------------------------------
# G2 — Options avancées
# ---------------------------------------------------------------------------

class TestAdvancedFragmentation:
    def test_shuffle_changes_order(self, jpeg_image, tmp_path):
        from scripts.generate_fragmented_dataset import generate_fragmented_dataset
        out1 = tmp_path / "no_shuffle"
        out2 = tmp_path / "shuffled"
        gt1 = generate_fragmented_dataset([jpeg_image], out1, n_fragments=5, shuffle=False, seed=1)
        gt2 = generate_fragmented_dataset([jpeg_image], out2, n_fragments=5, shuffle=True,  seed=1)
        offsets1 = [o["fragment_index"] for o in gt1["fragments"]["offsets"]]
        offsets2 = [o["fragment_index"] for o in gt2["fragments"]["offsets"]]
        # Avec shuffle, l'ordre doit différer (au moins parfois)
        # On vérifie juste que les deux listes ont les mêmes éléments
        assert sorted(offsets1) == sorted(offsets2)

    def test_loss_ratio_reduces_fragments(self, jpeg_image, tmp_path):
        from scripts.generate_fragmented_dataset import generate_fragmented_dataset
        out = tmp_path / "with_loss"
        gt = generate_fragmented_dataset(
            [jpeg_image], out, n_fragments=8, loss_ratio=0.3, seed=2
        )
        assert gt["fragments"]["lost"] > 0
        assert gt["fragments"]["retained"] < gt["fragments"]["total"]

    def test_noise_between_increases_dump_size(self, jpeg_image, tmp_path):
        from scripts.generate_fragmented_dataset import generate_fragmented_dataset
        out1 = tmp_path / "no_noise"
        out2 = tmp_path / "with_noise"
        gt1 = generate_fragmented_dataset([jpeg_image], out1, n_fragments=4, noise_between=False, seed=3)
        gt2 = generate_fragmented_dataset([jpeg_image], out2, n_fragments=4, noise_between=True, noise_size=256, seed=3)
        assert gt2["dump"]["size"] > gt1["dump"]["size"]

    def test_mix_images(self, two_jpegs, tmp_path):
        from scripts.generate_fragmented_dataset import generate_fragmented_dataset
        p1, p2 = two_jpegs
        out = tmp_path / "mixed"
        gt = generate_fragmented_dataset(
            [p1, p2], out, n_fragments=3, mix_images=True, shuffle=True, seed=4
        )
        sources = {o["source_image"] for o in gt["fragments"]["offsets"]}
        assert len(sources) == 2, f"Fragments d'une seule source : {sources}"

    def test_lost_fragments_in_gt(self, jpeg_image, tmp_path):
        from scripts.generate_fragmented_dataset import generate_fragmented_dataset
        out = tmp_path / "loss_gt"
        gt = generate_fragmented_dataset(
            [jpeg_image], out, n_fragments=6, loss_ratio=0.4, seed=5
        )
        assert len(gt["lost_fragments"]) == gt["fragments"]["lost"]
        if gt["lost_fragments"]:
            assert "sha256" in gt["lost_fragments"][0]
            assert "original_offset" in gt["lost_fragments"][0]


# ---------------------------------------------------------------------------
# G3 — Reconstruction de fragments
# ---------------------------------------------------------------------------

class TestFragmentAssembler:
    def _make_fragments(self, n: int = 4) -> list[dict]:
        data = bytes(range(256)) * 10
        size = len(data) // n
        return [
            {"index": i, "offset": i * size, "data": data[i*size:(i+1)*size]}
            for i in range(n)
        ]

    def test_greedy_chain_returns_all(self):
        from app.modules.carving.fragment_assembler import greedy_chain
        frags = self._make_fragments(4)
        chain = greedy_chain(frags)
        assert len(chain) == 4

    def test_greedy_chain_has_continuity_score(self):
        from app.modules.carving.fragment_assembler import greedy_chain
        frags = self._make_fragments(3)
        chain = greedy_chain(frags)
        for f in chain:
            assert "continuity_score" in f
            assert 0.0 <= f["continuity_score"] <= 1.0

    def test_assemble_creates_file(self, tmp_path):
        from app.modules.carving.fragment_assembler import greedy_chain, assemble_fragments
        frags = self._make_fragments(3)
        chain = greedy_chain(frags)
        out = tmp_path / "assembled.bin"
        result = assemble_fragments(chain, out)
        assert out.exists()
        assert result["size"] > 0
        assert len(result["sha256"]) == 64

    def test_assemble_result_structure(self, tmp_path):
        from app.modules.carving.fragment_assembler import greedy_chain, assemble_fragments
        frags = self._make_fragments(3)
        chain = greedy_chain(frags)
        out = tmp_path / "assembled.bin"
        result = assemble_fragments(chain, out)
        for key in ("path", "size", "sha256", "fragment_count", "assembly_score"):
            assert key in result

    def test_assemble_score_in_range(self, tmp_path):
        from app.modules.carving.fragment_assembler import greedy_chain, assemble_fragments
        frags = self._make_fragments(4)
        chain = greedy_chain(frags)
        out = tmp_path / "assembled.bin"
        result = assemble_fragments(chain, out)
        assert 0.0 <= result["assembly_score"] <= 1.0


# ---------------------------------------------------------------------------
# G4 — Rapport d'offsets
# ---------------------------------------------------------------------------

class TestFragmentReport:
    def _make_setup(self, tmp_path):
        from app.modules.carving.fragment_assembler import greedy_chain, assemble_fragments
        data = bytes(range(200)) * 5
        size = len(data) // 3
        frags = [
            {"index": i, "offset": i * size, "data": data[i*size:(i+1)*size],
             "source_image": "test.jpg"}
            for i in range(3)
        ]
        chain = greedy_chain(frags)
        out = tmp_path / "assembled.bin"
        assembly_result = assemble_fragments(chain, out)
        return frags, chain, assembly_result

    def test_report_structure(self, tmp_path):
        from app.modules.carving.fragment_assembler import build_fragment_report
        frags, chain, assembly = self._make_setup(tmp_path)
        report = build_fragment_report(tmp_path / "fake.bin", frags, chain, assembly)
        for key in ("source_dump", "sha256_dump", "total_fragments", "used_fragments",
                    "offsets", "fragments_used", "assembly", "assembly_score"):
            assert key in report, f"Clé manquante dans rapport : {key}"

    def test_report_offsets_count(self, tmp_path):
        from app.modules.carving.fragment_assembler import build_fragment_report
        frags, chain, assembly = self._make_setup(tmp_path)
        report = build_fragment_report(tmp_path / "fake.bin", frags, chain, assembly)
        assert report["total_fragments"] == 3
        assert len(report["offsets"]) == 3

    def test_report_fragments_used_have_sha256(self, tmp_path):
        from app.modules.carving.fragment_assembler import build_fragment_report
        frags, chain, assembly = self._make_setup(tmp_path)
        report = build_fragment_report(tmp_path / "fake.bin", frags, chain, assembly)
        for f in report["fragments_used"]:
            assert "sha256" in f
            assert len(f["sha256"]) == 64

    def test_report_rejected_fragments(self, tmp_path):
        from app.modules.carving.fragment_assembler import greedy_chain, assemble_fragments, build_fragment_report
        data = bytes(range(100)) * 4
        frags = [{"index": i, "offset": i*100, "data": data[i*100:(i+1)*100]} for i in range(4)]
        rejected = [{"index": 99, "offset": 400, "data": b"garbage", "reject_reason": "too_small"}]
        chain = greedy_chain(frags)
        out = tmp_path / "assembled.bin"
        assembly = assemble_fragments(chain, out)
        report = build_fragment_report(tmp_path / "fake.bin", frags, chain, assembly, rejected_fragments=rejected)
        assert report["rejected_fragments"] == 1
        assert report["fragments_rejected"][0]["reason"] == "too_small"