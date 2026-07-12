from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
DUMPS_DIR = DATA_DIR / "dumps"
EXTRACTED_DIR = DATA_DIR / "extracted"
CORRUPTED_DIR = DATA_DIR / "corrupted"
RECONSTRUCTED_DIR = DATA_DIR / "reconstructed"
REPORTS_DIR = DATA_DIR / "reports"
MASKS_DIR = DATA_DIR / "masks"

JPEG_SOI = b"\xFF\xD8"
JPEG_EOI = b"\xFF\xD9"

DEFAULT_ENCODING = "utf-8"
DEFAULT_MIN_JPEG_SIZE = 128
DEFAULT_READ_CHUNK_SIZE = 1024 * 1024
DEFAULT_INPAINT_RADIUS = 3


def ensure_directories() -> None:
    for path in [
        DATA_DIR,
        DUMPS_DIR,
        EXTRACTED_DIR,
        CORRUPTED_DIR,
        RECONSTRUCTED_DIR,
        REPORTS_DIR,
        MASKS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def build_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_extracted_image_path(dump_name: str, index: int) -> Path:
    ensure_directories()
    stem = Path(dump_name).stem
    timestamp = build_timestamp()
    return EXTRACTED_DIR / f"{stem}_{timestamp}_img_{index:03d}.jpg"


def build_corrupted_image_path(source_name: str, corruption_type: str) -> Path:
    ensure_directories()
    stem = Path(source_name).stem
    timestamp = build_timestamp()
    return CORRUPTED_DIR / f"{stem}_{corruption_type}_{timestamp}.png"


def build_mask_path(source_name: str, corruption_type: str) -> Path:
    ensure_directories()
    stem = Path(source_name).stem
    timestamp = build_timestamp()
    return MASKS_DIR / f"{stem}_{corruption_type}_{timestamp}_mask.png"


def build_reconstructed_image_path(source_name: str, method: str) -> Path:
    ensure_directories()
    stem = Path(source_name).stem
    timestamp = build_timestamp()
    return RECONSTRUCTED_DIR / f"{stem}_{method}_{timestamp}.png"