import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def compute_stats(paths_file: str, root: str):
    paths = Path(paths_file).read_text().splitlines()
    paths = [p.strip() for p in paths if p.strip()]

    sum_rgb = np.zeros(3, dtype=np.float64)
    sum_sq_rgb = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    total = len(paths)
    for i, rel_path in enumerate(paths, 1):
        img_path = str(Path(root) / rel_path) + '.jpg'
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float64) / 255.0
        h, w, _ = img.shape
        n = h * w
        sum_rgb += img.reshape(-1, 3).sum(axis=0)
        sum_sq_rgb += (img ** 2).reshape(-1, 3).sum(axis=0)
        pixel_count += n
        print(f"\r{i}/{total}", end="", flush=True)

    mean = sum_rgb / pixel_count
    std = np.sqrt(sum_sq_rgb / pixel_count - mean ** 2)

    print()
    print(f"Processed {len(paths)} images ({pixel_count:,} pixels)")
    print(f"Mean: R={mean[0]:.6f}  G={mean[1]:.6f}  B={mean[2]:.6f}")
    print(f"Std:  R={std[0]:.6f}  G={std[1]:.6f}  B={std[2]:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths_file", help="Text file with one relative path per line")
    parser.add_argument("--root", default=".", help="Root directory to prepend to each path")
    args = parser.parse_args()
    compute_stats(args.paths_file, args.root)