import os
from pathlib import Path
from PIL import Image
import numpy as np

OUTPUT_PATH = Path("data/dumps/dump.bin")


def generate_random_bytes(size: int) -> bytes:
    return os.urandom(size)


def create_test_image(path: Path):
    img = Image.new("RGB", (128, 128), color="blue")
    img.save(path, format="JPEG")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_img_path = Path("temp_test.jpg")
    create_test_image(temp_img_path)

    with open(temp_img_path, "rb") as f:
        jpeg_bytes = f.read()

    with open(OUTPUT_PATH, "wb") as dump:
        # bruit avant
        dump.write(generate_random_bytes(1024))

        # image JPEG
        dump.write(jpeg_bytes)

        # bruit après
        dump.write(generate_random_bytes(2048))

    temp_img_path.unlink()

    print(f"Dump généré : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()