import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.pipeline_service import run_demo_pipeline


def show_images(original, corrupted, mask, reconstructed):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    titles = ["Originale", "Corrompue", "Masque", "Reconstruite"]
    images = [original, corrupted, mask, reconstructed]

    for ax, img, title in zip(axes, images, titles):
        if title == "Masque":
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    return fig


def zoom_region(img, box):
    return img.crop(box)


def show_zoom(original, corrupted, reconstructed, box):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    titles = ["Originale (zoom)", "Corrompue (zoom)", "Reconstruite (zoom)"]
    images = [
        zoom_region(original, box),
        zoom_region(corrupted, box),
        zoom_region(reconstructed, box),
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    return fig


def main() -> None:
    print("🚀 Démo réaliste en cours...")

    source_image = PROJECT_ROOT / "data" / "input" / "demo_real.jpeg"

    if not source_image.exists():
        raise FileNotFoundError(f"Image de démo introuvable : {source_image}")

    result = run_demo_pipeline(
        source_image_path=source_image,
        corruption_type="rectangle_mask",
        corruption_params={
            "x": 80,
            "y": 60,
            "width": 120,
            "height": 90,
            "fill_value": 0,
        },
        inpaint_method="opencv_inpaint",
        radius=3,
    )

    print("✅ Pipeline terminé")

    original_path = result["source_image"]
    corrupted_path = result["corruption"]["path"]
    mask_path = result["corruption"]["mask_path"]
    reconstructed_path = result["reconstruction"]["path"]

    print("📂 Chargement des images...")
    original = Image.open(original_path)
    corrupted = Image.open(corrupted_path)
    mask = Image.open(mask_path)
    reconstructed = Image.open(reconstructed_path)

    print("🖼️ Génération des comparaisons...")

    reports_dir = PROJECT_ROOT / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    fig_global = show_images(original, corrupted, mask, reconstructed)
    global_output = reports_dir / "comparison_real_case.png"
    fig_global.savefig(global_output)

    w, h = original.size
    box = (w // 3, h // 3, (w // 3) * 2, (h // 3) * 2)

    fig_zoom = show_zoom(original, corrupted, reconstructed, box)
    zoom_output = reports_dir / "comparison_zoom.png"
    fig_zoom.savefig(zoom_output)

    print(f"📸 Vue globale sauvegardée : {global_output}")
    print(f"📸 Vue zoom sauvegardée   : {zoom_output}")

    plt.show()


if __name__ == "__main__":
    main()