from pathlib import Path

from app.core.config import ensure_directories
from app.services.pipeline_service import run_demo_pipeline
from app.core.full_pipeline_service import run_full_pipeline

def main() -> None:
    ensure_directories()

    source_image = Path("data/extracted/sample.jpg")

    if not source_image.exists():
        print(f"Image source absente : {source_image}")
        return

    result = run_demo_pipeline(
        source_image_path=source_image,
        corruption_type="rectangle_mask",
        corruption_params={
            "x": 40,
            "y": 40,
            "width": 80,
            "height": 80,
            "fill_value": 0,
        },
        inpaint_method="opencv_inpaint",
        radius=3,
    )

    print("\n=== DEMO PIPELINE ===")
    print("Source         :", result["source_image"])
    print("Corrompue      :", result["corruption"]["path"])
    print("Masque         :", result["corruption"]["mask_path"])
    print("Reconstruite   :", result["reconstruction"]["path"])
    print("Évaluation     :", result["evaluation"])
    print("Rapport JSON   :", result["report_path"])


def main() -> None:
    dump_path = "data/samples/dumps/dump.bin"

    result = run_full_pipeline(dump_path)

    print("=== FULL PIPELINE RESULT ===")
    print(result)


if __name__ == "__main__":
    main()