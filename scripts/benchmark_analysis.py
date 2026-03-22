import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    csv_path = PROJECT_ROOT / "data" / "reports" / "benchmark_results.csv"
    df = pd.read_csv(csv_path)

    reports_dir = PROJECT_ROOT / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for mode in df["execution_mode"].unique():
        sub = df[df["execution_mode"] == mode].sort_values("corruption_level")
        plt.plot(sub["corruption_level"], sub["psnr_reconstructed"], marker="o", label=mode)
    plt.xlabel("Corruption level")
    plt.ylabel("PSNR reconstructed")
    plt.title("Corruption level vs PSNR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "benchmark_psnr.png")

    plt.figure(figsize=(8, 5))
    for mode in df["execution_mode"].unique():
        sub = df[df["execution_mode"] == mode].sort_values("corruption_level")
        plt.plot(sub["corruption_level"], sub["ssim_reconstructed"], marker="o", label=mode)
    plt.xlabel("Corruption level")
    plt.ylabel("SSIM reconstructed")
    plt.title("Corruption level vs SSIM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "benchmark_ssim.png")

    plt.figure(figsize=(8, 5))
    for mode in df["execution_mode"].unique():
        sub = df[df["execution_mode"] == mode].sort_values("corruption_level")
        plt.plot(sub["corruption_level"], sub["iou"], marker="o", label=mode)
    plt.xlabel("Corruption level")
    plt.ylabel("IoU")
    plt.title("Corruption level vs IoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "benchmark_iou.png")

    plt.show()


if __name__ == "__main__":
    main()