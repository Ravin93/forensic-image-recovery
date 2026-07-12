import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.pipeline_service import run_demo_pipeline


def show_row(ax_row, images, titles):
    for ax, img, title in zip(ax_row, images, titles):
        if "Masque" in title:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")


def main() -> None:
    source_image = PROJECT_ROOT / "data" / "input" / "demo_real.jpeg"
    if not source_image.exists():
        raise FileNotFoundError(f"Image introuvable : {source_image}")

    # Cas A : masque parfait
    result_a = run_demo_pipeline(
        source_image_path=source_image,
        corruption_type="combined",
        corruption_params={"fill_value": 0, "sigma": 10.0},
        use_mask=True,
        imperfect_mask=False,
        randomize=True,
        seed=42,
    )

    # Cas B : masque approximatif / blind
    result_b = run_demo_pipeline(
        source_image_path=source_image,
        corruption_type="combined",
        corruption_params={"fill_value": 0, "sigma": 10.0},
        use_mask=False,
        imperfect_mask=True,
        randomize=True,
        seed=42,
    )

    original = Image.open(result_a["source_image"])

    corrupted_a = Image.open(result_a["corruption"]["path"])
    mask_a = Image.open(result_a["mode"]["reconstruction_mask_path"])
    reconstructed_a = Image.open(result_a["reconstruction"]["path"])

    corrupted_b = Image.open(result_b["corruption"]["path"])
    mask_b = Image.open(result_b["mode"]["reconstruction_mask_path"])
    reconstructed_b = Image.open(result_b["reconstruction"]["path"])

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    show_row(
        axes[0],
        [original, corrupted_a, mask_a, reconstructed_a],
        ["Originale", "Corrompue A", "Masque parfait", "Reconstruite A"],
    )
    show_row(
        axes[1],
        [original, corrupted_b, mask_b, reconstructed_b],
        ["Originale", "Corrompue B", "Masque approx.", "Reconstruite B"],
    )

    plt.tight_layout()

    output = PROJECT_ROOT / "data" / "reports" / "comparison_ab.png"
    fig.savefig(output)
    print(f"Comparaison A/B sauvegardée : {output}")

    print("Métriques A :", result_a["evaluation"])
    print("Métriques B :", result_b["evaluation"])

    plt.show()


if __name__ == "__main__":
    main()