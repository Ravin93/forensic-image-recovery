import json
from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    reports_dir = project_root / "data" / "reports"
    csv_path = reports_dir / "benchmark_results.csv"

    df = pd.read_csv(csv_path)

    summary = (
        df.groupby("execution_mode")
        .agg(
            avg_psnr_reconstructed=("psnr_reconstructed", "mean"),
            avg_ssim_reconstructed=("ssim_reconstructed", "mean"),
            avg_iou=("iou", "mean"),
            improvement_rate=("improvement", "mean"),
        )
        .reset_index()
    )

    summary_csv = reports_dir / "benchmark_summary.csv"
    summary_json = reports_dir / "benchmark_summary.json"

    summary.to_csv(summary_csv, index=False)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    print(f"Synthèse CSV : {summary_csv}")
    print(f"Synthèse JSON: {summary_json}")


if __name__ == "__main__":
    main()