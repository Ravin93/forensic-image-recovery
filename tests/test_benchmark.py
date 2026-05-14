from __future__ import annotations
import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

@pytest.fixture
def gradient_image(tmp_path):
    arr = np.zeros((128,128,3),dtype=np.uint8)
    for i in range(128):
        arr[i,:,0]=i*2; arr[:,i,1]=i*2; arr[i,:,2]=255-i*2
    p = tmp_path/"source.jpg"
    Image.fromarray(arr).save(p, format="JPEG")
    return p

class TestBenchmarkRunner:
    def test_run_single_type(self, gradient_image, tmp_path):
        from app.modules.benchmark.benchmark_runner import run_benchmark
        result = run_benchmark(
            image_paths=[gradient_image],
            output_dir=tmp_path/"bench",
            corruption_types=["scratch_lines"],
            max_attempts=2, seed=42)
        assert "rows" in result
        assert len(result["rows"]) == 1
        assert result["rows"][0]["corruption_type"] == "scratch_lines"

    def test_result_has_psnr_ssim(self, gradient_image, tmp_path):
        from app.modules.benchmark.benchmark_runner import run_benchmark
        result = run_benchmark(
            image_paths=[gradient_image],
            output_dir=tmp_path/"bench",
            corruption_types=["zone_deletion"],
            max_attempts=2, seed=42)
        row = result["rows"][0]
        for k in ("psnr_before","psnr_after","psnr_gain","ssim_before","ssim_after","ssim_gain","score","strategy","elapsed_s"):
            assert k in row, f"Cle manquante : {k}"

    def test_psnr_improves_after_reconstruction(self, gradient_image, tmp_path):
        from app.modules.benchmark.benchmark_runner import run_benchmark
        result = run_benchmark(
            image_paths=[gradient_image],
            output_dir=tmp_path/"bench",
            corruption_types=["scratch_lines"],
            max_attempts=3, seed=42)
        row = result["rows"][0]
        if row["psnr_gain"] is not None:
            assert row["psnr_gain"] >= 0, "PSNR apres < avant"

    def test_summary_built(self, gradient_image, tmp_path):
        from app.modules.benchmark.benchmark_runner import run_benchmark
        result = run_benchmark(
            image_paths=[gradient_image],
            output_dir=tmp_path/"bench",
            corruption_types=["scratch_lines","multiple_bars"],
            max_attempts=2, seed=42)
        assert "summary" in result
        assert "scratch_lines" in result["summary"]
        for k in ("avg_psnr_gain","avg_score","best_strategy","avg_time_s"):
            assert k in result["summary"]["scratch_lines"]

    def test_csv_created(self, gradient_image, tmp_path):
        from app.modules.benchmark.benchmark_runner import run_benchmark
        result = run_benchmark(
            image_paths=[gradient_image],
            output_dir=tmp_path/"bench",
            corruption_types=["scratch_lines"],
            max_attempts=2, seed=42)
        assert Path(result["paths"]["csv"]).exists()

    def test_json_created(self, gradient_image, tmp_path):
        from app.modules.benchmark.benchmark_runner import run_benchmark
        result = run_benchmark(
            image_paths=[gradient_image],
            output_dir=tmp_path/"bench",
            corruption_types=["scratch_lines"],
            max_attempts=2, seed=42)
        import json
        data = json.loads(Path(result["paths"]["json"]).read_text())
        assert "rows" in data and "summary" in data

    def test_plots_generated(self, gradient_image, tmp_path):
        from app.modules.benchmark.benchmark_runner import run_benchmark
        result = run_benchmark(
            image_paths=[gradient_image],
            output_dir=tmp_path/"bench",
            corruption_types=["scratch_lines","zone_deletion"],
            max_attempts=2, seed=42)
        for k in ("plot_psnr","plot_ssim","plot_time"):
            assert k in result["paths"], f"Graphique manquant : {k}"
            assert Path(result["paths"][k]).exists()

    def test_missing_image_skipped(self, tmp_path):
        from app.modules.benchmark.benchmark_runner import run_benchmark
        result = run_benchmark(
            image_paths=[tmp_path/"nonexistent.jpg"],
            output_dir=tmp_path/"bench",
            corruption_types=["scratch_lines"],
            max_attempts=2)
        assert result["rows"] == []

class TestBenchmarkAPI:
    def test_results_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.get("/benchmark/results")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_download_nonexistent(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.get("/benchmark/download/benchmark_99999.csv")
        assert resp.status_code == 404

    def test_download_path_traversal_blocked(self):
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        resp = client.get("/benchmark/download/../../../etc/passwd")
        assert resp.status_code in (403, 404, 422)