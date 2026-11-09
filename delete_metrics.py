"""
Delete all records for a given (task, algo) pair from metrics.csv
and recompute metrics_summary.csv from the remaining rows.

Usage:
  python delete_metrics.py --task coded_pr --algo ddpm
  python delete_metrics.py --task coded_pr --algo ddpm --metrics metrics.csv
"""

import csv
import argparse
from pathlib import Path


FIELDNAMES         = ["task", "algo", "id", "psnr", "ssim", "lpips"]
SUMMARY_FIELDNAMES = ["task", "algo", "n", "psnr_mean", "ssim_mean", "lpips_mean"]


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)





def main():
    parser = argparse.ArgumentParser(
        description="Delete all records for a (task, algo) pair from metrics CSV files."
    )
    parser.add_argument("--task",    required=True, help="Task name to delete.")
    parser.add_argument("--algo",    required=True, help="Algorithm name to delete.")
    parser.add_argument("--metrics", default="metrics.csv",
                        help="Path to the per-image metrics CSV (default: metrics.csv).")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    summary_path = metrics_path.with_name(metrics_path.stem + "_summary" + metrics_path.suffix)

    # ── per-image CSV ─────────────────────────────────────────────────────────
    all_rows  = read_csv(metrics_path)
    kept_rows = [r for r in all_rows
                 if not (r["task"] == args.task and r["algo"] == args.algo)]
    deleted_n = len(all_rows) - len(kept_rows)

    if deleted_n == 0:
        print(f"No records found for task='{args.task}' algo='{args.algo}' in {metrics_path}.")
    else:
        write_csv(metrics_path, kept_rows, FIELDNAMES)
        print(f"Deleted {deleted_n} row(s) from {metrics_path}.")

    # ── summary CSV ───────────────────────────────────────────────────────────
    summary_rows  = read_csv(summary_path)
    kept_summary  = [r for r in summary_rows
                     if not (r["task"] == args.task and r["algo"] == args.algo)]
    deleted_s     = len(summary_rows) - len(kept_summary)

    if deleted_s == 0:
        print(f"No records found for task='{args.task}' algo='{args.algo}' in {summary_path}.")
    else:
        write_csv(summary_path, kept_summary, SUMMARY_FIELDNAMES)
        print(f"Deleted {deleted_s} row(s) from {summary_path}.")


if __name__ == "__main__":
    main()