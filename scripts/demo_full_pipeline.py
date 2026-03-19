import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.full_pipeline_service import run_full_pipeline


def main() -> None:
    dump_path = PROJECT_ROOT / "data" / "dumps" / "dump.bin"

    result = run_full_pipeline(dump_path)

    print("=== FULL PIPELINE RESULT ===")
    print(result)


if __name__ == "__main__":
    main()