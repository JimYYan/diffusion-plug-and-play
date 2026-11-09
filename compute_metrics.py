"""
Compute PSNR, SSIM, and LPIPS for one (task, algo) run at a time.
Results are appended to a shared CSV so all runs accumulate in one file.
After each run, a summary CSV of per-(task, algo) means is also updated.

Expected folder structure inside --root:
  <root>/
    label/
      <id><algo>.png          e.g. 00000frites.png
    recon/
      <id><algo>.png          e.g. 00000frites.png

Usage:
  python compute_metrics.py \
      --root   ~/experiments/super_resolution \
      --task   super_resolution \
      --algo   frites \
      --output metrics.csv

  Per-image rows are appended to metrics.csv.
  Means across all accumulated rows are saved to metrics_summary.csv
  (or <output>_summary.csv if a custom --output name is given).

Run once per (task, algo) pair. Rows are appended; existing rows are kept.
Re-running the same (task, algo) pair will append duplicates — delete the CSV
or filter the existing rows beforehand if you need to recompute.
"""

import csv
import argparse
import numpy as np
from pathlib import Path
from skimage import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import lpips


# ── image helpers ─────────────────────────────────────────────────────────────

def load_image_np(path: Path) -> np.ndarray:
    """Load PNG as float32 numpy array in [0, 1], shape (H, W, C)."""
    img = io.imread(str(path))
    if img.ndim == 2:           # grayscale → RGB
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:       # RGBA → RGB
        img = img[..., :3]
    return img.astype(np.float32) / 255.0


def np_to_tensor(img: np.ndarray) -> torch.Tensor:
    """(H, W, C) float32 [0,1]  →  (1, C, H, W) float32 [-1, 1]."""
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return t * 2.0 - 1.0


# ── metric functions ──────────────────────────────────────────────────────────

def compute_psnr(gt: np.ndarray, recon: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(gt, recon, data_range=1.0))


def compute_ssim(gt: np.ndarray, recon: np.ndarray) -> float:
    return float(structural_similarity(gt, recon, data_range=1.0, channel_axis=2))


def compute_lpips(gt: np.ndarray, recon: np.ndarray, lpips_fn) -> float:
    with torch.no_grad():
        gt_t    = np_to_tensor(gt)
        recon_t = np_to_tensor(recon)
        if torch.cuda.is_available():
            gt_t, recon_t = gt_t.cuda(), recon_t.cuda()
        score = lpips_fn(gt_t, recon_t)
    return float(score.item())


# ── CSV helpers ───────────────────────────────────────────────────────────────

FIELDNAMES         = ["task", "algo", "id", "psnr", "ssim", "lpips"]
SUMMARY_FIELDNAMES = ["task", "algo", "n", "psnr_mean", "ssim_mean", "lpips_mean"]


def append_rows(csv_path: Path, rows: list[dict]) -> None:
    """Append per-image rows to CSV, writing header only if the file is new/empty."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def read_all_rows(csv_path: Path) -> list[dict]:
    """Read every per-image row from the shared CSV."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def write_summary(summary_path: Path, all_rows: list[dict]) -> list[dict]:
    """
    Compute per-(task, algo) means from all_rows and write to summary_path.
    Returns the list of summary dicts.
    """
    from collections import defaultdict

    buckets: dict[tuple, list] = defaultdict(list)
    for r in all_rows:
        buckets[(r["task"], r["algo"])].append(r)

    summary_rows = []
    for (task, algo), group in sorted(buckets.items()):
        summary_rows.append({
            "task":        task,
            "algo":        algo,
            "n":           len(group),
            "psnr_mean":   round(np.mean([float(r["psnr"])  for r in group]), 4),
            "ssim_mean":   round(np.mean([float(r["ssim"])  for r in group]), 4),
            "lpips_mean":  round(np.mean([float(r["lpips"]) for r in group]), 4),
        })

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute PSNR / SSIM / LPIPS for one task + algorithm at a time."
    )
    parser.add_argument("--root",   required=True,
                        help="Main experiment folder containing label/ and recon/.")
    parser.add_argument("--task",   required=True,
                        help="Task name, used only as a label in the CSV.")
    parser.add_argument("--algo",   required=True,
                        help="Algorithm suffix used in reconstruction filenames, e.g. 'frites'.")
    parser.add_argument("--output", default="metrics.csv",
                        help="Shared output CSV file (rows are appended). Default: metrics.csv.")
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"],
                        help="LPIPS backbone network (default: alex).")
    parser.add_argument("--recon_folder", default="recon",
                        help="Subfolder to read reconstructions from (default: recon).")
    parser.add_argument("--label_algo", default=None,
                        help="Algo suffix used in label filenames. Defaults to --algo.")
    parser.add_argument("--algo_label", default=None,
                        help="Name to use in the CSV algo column. Defaults to --algo.")

    args = parser.parse_args()

    root       = Path(args.root).expanduser()
    recon_dir  = root / args.recon_folder
    label_dir  = root / "label"
    output_csv   = Path(args.output)
    summary_csv  = output_csv.with_name(output_csv.stem + "_summary" + output_csv.suffix)

    # Validate directories
    for d, name in [(recon_dir, "recon"), (label_dir, "label")]:
        if not d.is_dir():
            raise FileNotFoundError(f"{name} folder not found: {d}")

    # Initialise LPIPS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device  : {device}")
    print(f"Task    : {args.task}")
    print(f"Algo    : {args.algo}")
    print(f"Output  : {output_csv}")
    print(f"Summary : {summary_csv}\n")

    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device)
    lpips_fn.eval()

    # Collect reconstruction files that match the algo suffix
    recon_files = sorted(
        p for p in recon_dir.glob("*.png") if p.stem.endswith(args.algo)
    )

    if not recon_files:
        print(f"No files ending with '{args.algo}' found in {recon_dir}")
        return

    print(f"Found {len(recon_files)} reconstruction(s) for algo '{args.algo}'.\n")

    rows = []
    for recon_path in recon_files:
        img_id     = recon_path.stem[: -len(args.algo)]   # strip algo suffix
        label_algo = args.label_algo or args.algo
        label_path = label_dir / f"{img_id}{label_algo}.png"

        if not label_path.exists():
            print(f"  MISS  label not found: {label_path}")
            continue

        try:
            gt    = load_image_np(label_path)
            recon = load_image_np(recon_path)

            if gt.shape != recon.shape:
                print(f"  WARN  shape mismatch {gt.shape} vs {recon.shape}: {recon_path.name}")

            psnr = compute_psnr(gt, recon)
            ssim = compute_ssim(gt, recon)
            lp   = compute_lpips(gt, recon, lpips_fn)

            rows.append({
                "task":  args.task,
                "algo":  args.algo_label or args.algo,
                "id":    img_id,
                "psnr":  round(psnr, 4),
                "ssim":  round(ssim, 4),
                "lpips": round(lp,   4),
            })
            print(f"  OK  {recon_path.name}  PSNR={psnr:.2f}  SSIM={ssim:.4f}  LPIPS={lp:.4f}")

        except Exception as e:
            print(f"  ERROR  {recon_path.name}: {e}")

    # Append new rows to shared per-image CSV
    append_rows(output_csv, rows)
    print(f"\nAppended {len(rows)} row(s) → {output_csv}")

    # Rebuild summary from the full CSV (all accumulated runs)
    all_rows     = read_all_rows(output_csv)
    summary_rows = write_summary(summary_csv, all_rows)
    print(f"Summary updated ({len(summary_rows)} group(s)) → {summary_csv}")

    # Print summary table to terminal
    print(f"\n── Means across all accumulated runs {'─' * 30}")
    print(f"  {'Task':<22} {'Algo':<12} {'N':>5} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8}")
    print(f"  {'─'*22} {'─'*12} {'─'*5} {'─'*8} {'─'*8} {'─'*8}")
    for s in summary_rows:
        print(f"  {s['task']:<22} {s['algo']:<12} {s['n']:>5} "
              f"{s['psnr_mean']:>8.4f} {s['ssim_mean']:>8.4f} {s['lpips_mean']:>8.4f}")


if __name__ == "__main__":
    main()
