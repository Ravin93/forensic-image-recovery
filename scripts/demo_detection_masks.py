import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.modules.corruption.detection import detect_advanced_mask, detect_dark_regions_mask
from app.modules.corruption.simulator import corrupt_image


def main() -> None:
    source = PROJECT_ROOT / "data" / "input" / "demo_real.jpeg"

    corruption = corrupt_image(
        image_path=source,
        corruption_type="rectangle_mask",
        randomize=True,
        imperfect_mask=False,
        seed=42,
        min_size=60,
        max_size=100,
        fill_value=0,
    )

    true_mask = np.array(Image.open(corruption["mask_path"]).convert("L"))
    basic_mask = np.array(detect_dark_regions_mask(corruption["path"]).convert("L"))
    advanced_mask = np.array(detect_advanced_mask(corruption["path"])["mask"].convert("L"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(true_mask, cmap="gray")
    axes[0].set_title("Masque vrai")
    axes[0].axis("off")

    axes[1].imshow(basic_mask, cmap="gray")
    axes[1].set_title("Masque basic")
    axes[1].axis("off")

    axes[2].imshow(advanced_mask, cmap="gray")
    axes[2].set_title("Masque advanced")
    axes[2].axis("off")

    plt.tight_layout()

    output_path = PROJECT_ROOT / "data" / "reports" / "detection_masks_comparison.png"
    plt.savefig(output_path)
    print(f"Comparaison des masques sauvegardée : {output_path}")
    plt.show()


if __name__ == "__main__":
    main()