"""Tests K15 — Clustering de fragments."""
from __future__ import annotations
import numpy as np
import pytest
from PIL import Image
from pathlib import Path


@pytest.fixture
def fragment_set(tmp_path):
    """3 fragments similaires (rouge) + 2 differents (bleu)."""
    paths = []
    # Groupe 1 : rouge
    for i in range(3):
        arr = np.zeros((32,32,3),dtype=np.uint8)
        arr[:,:,2] = 180 + i*5  # rouge
        p = tmp_path/f"frag_r{i}.png"; Image.fromarray(arr).save(p); paths.append(p)
    # Groupe 2 : bleu
    for i in range(2):
        arr = np.zeros((32,32,3),dtype=np.uint8)
        arr[:,:,0] = 180 + i*5  # bleu
        p = tmp_path/f"frag_b{i}.png"; Image.fromarray(arr).save(p); paths.append(p)
    return paths


class TestFragmentClustering:
    def test_returns_dict(self, fragment_set):
        from app.modules.fragments.fragment_clustering import cluster_fragments_dbscan
        r = cluster_fragments_dbscan(fragment_set)
        assert isinstance(r, dict)

    def test_required_keys(self, fragment_set):
        from app.modules.fragments.fragment_clustering import cluster_fragments_dbscan
        r = cluster_fragments_dbscan(fragment_set)
        for k in ("clusters","noise","n_clusters","n_noise","coherence_score"):
            assert k in r, f"Cle manquante : {k}"

    def test_coherence_in_range(self, fragment_set):
        from app.modules.fragments.fragment_clustering import cluster_fragments_dbscan
        r = cluster_fragments_dbscan(fragment_set)
        assert 0.0 <= r["coherence_score"] <= 1.0

    def test_similar_fragments_grouped(self, fragment_set):
        from app.modules.fragments.fragment_clustering import cluster_fragments_dbscan
        r = cluster_fragments_dbscan(fragment_set, eps=0.5, min_samples=2)
        # Au moins 1 cluster doit etre forme
        total_clustered = sum(c["size"] for c in r["clusters"])
        assert total_clustered + r["n_noise"] == r["n_fragments"]

    def test_empty_list(self):
        from app.modules.fragments.fragment_clustering import cluster_fragments_dbscan
        r = cluster_fragments_dbscan([])
        assert r["n_clusters"] == 0
        assert r["clusters"] == []

    def test_extract_features_shape(self, fragment_set):
        from app.modules.fragments.fragment_clustering import extract_fragment_features
        feat = extract_fragment_features(fragment_set[0])
        assert feat.ndim == 1
        assert len(feat) > 50

    def test_single_fragment(self, fragment_set):
        from app.modules.fragments.fragment_clustering import cluster_fragments_dbscan
        r = cluster_fragments_dbscan([fragment_set[0]])
        assert r["n_fragments"] == 1