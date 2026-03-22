import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.pipeline_service import run_demo_pipeline


LEVELS = [10, 20, 40, 60, 80]


def load_img(path: str | Path):
    return Image.open(path)


def main() -> None:
    source_image = PROJECT_ROOT / "data" / "input" / "demo_real.jpeg"
    if not source_image.exists():
        raise FileNotFoundError(f"Image introuvable : {source_image}")

    reports_dir = PROJECT_ROOT / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    fig, axes = plt.subplots(len(LEVELS), 3, figsize=(12, 4 * len(LEVELS)))
    if len(LEVELS) == 1:
        axes = [axes]

    for i, level in enumerate(LEVELS):
        result = run_demo_pipeline(
            source_image_path=source_image,
            corruption_level=level,
            seed=42,
        )

        corr = result["evaluation"]["original_vs_corrupted"]
        reco = result["evaluation"]["original_vs_reconstructed"]

        row = {
            "corruption_level": level,
            "corruption_profile": result["mode"]["corruption_profile"],
            "psnr_corrupted": corr["psnr"],
            "ssim_corrupted": corr["ssim"],
            "psnr_reconstructed": reco["psnr"],
            "ssim_reconstructed": reco["ssim"],
            "improvement": result["evaluation"]["improvement"],
            "recoverability_status": result["recoverability_status"],
            "corrupted_path": result["corruption"]["path"],
            "reconstructed_path": result["reconstruction"]["path"],
            "report_path": result["report_path"],
        }
        rows.append(row)

        corrupted = load_img(result["corruption"]["path"])
        reconstructed = load_img(result["reconstruction"]["path"])

        axes[i][0].imshow(load_img(result["source_image"]))
        axes[i][0].set_title(f"Niveau {level} - Originale")
        axes[i][0].axis("off")

        axes[i][1].imshow(corrupted)
        axes[i][1].set_title(f"Corrompue ({row['recoverability_status']})")
        axes[i][1].axis("off")

        axes[i][2].imshow(reconstructed)
        axes[i][2].set_title(
            f"Reconstruite\nPSNR={float(reco['psnr']):.2f} SSIM={float(reco['ssim']):.3f}"
        )
        axes[i][2].axis("off")

    plt.tight_layout()
    comparison_path = reports_dir / "comparison_levels.png"
    fig.savefig(comparison_path)

    json_path = reports_dir / "corruption_levels_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, default=str)

    csv_path = reports_dir / "corruption_levels_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "corruption_level",
                "corruption_profile",
                "psnr_corrupted",
                "ssim_corrupted",
                "psnr_reconstructed",
                "ssim_reconstructed",
                "improvement",
                "recoverability_status",
                "corrupted_path",
                "reconstructed_path",
                "report_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Comparaison multi-niveaux sauvegardée : {comparison_path}")
    print(f"Résultats JSON sauvegardés : {json_path}")
    print(f"Résultats CSV sauvegardés  : {csv_path}")

    for row in rows:
        print(
            f"Niveau {row['corruption_level']:>3} | "
            f"{row['recoverability_status']:<22} | "
            f"PSNR reco={float(row['psnr_reconstructed']):.2f} | "
            f"SSIM reco={float(row['ssim_reconstructed']):.3f}"
        )

    plt.show()


if __name__ == "__main__":
    main()