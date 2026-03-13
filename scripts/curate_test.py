"""
For each class report the count of exactly 512×512 images in the train and
test splits, then print the class with the minimum count in each split.
"""

from collections import defaultdict
from pathlib import Path

from PIL import Image

from data.dataset import Food101Dataset

ROOT = Path(__file__).parent


def count_512x512_per_class(split: str, classes: list[str]) -> dict[str, int]:
    dataset = Food101Dataset(ROOT, split=split, transform=None)
    print(f"[{split}]  scanning {len(dataset)} samples ...")

    counts: dict[str, int] = defaultdict(int)
    for idx, rel_path in enumerate(dataset.samples):
        class_name = rel_path.split("/")[0]
        with Image.open(dataset._image_path(rel_path)) as img:
            w, h = img.size
        if w == 512 and h == 512:
            counts[class_name] += 1
        if (idx + 1) % 10000 == 0:
            print(f"  {idx + 1}/{len(dataset)} ...")

    # ensure every class is represented even if count is 0
    return {cls: counts[cls] for cls in classes}


if __name__ == "__main__":
    ref = Food101Dataset(ROOT, split="train", transform=None)
    classes = ref.classes

    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "test"):
        split_counts[split] = count_512x512_per_class(split, classes)

    # ── per-class table ───────────────────────────────────────────────────
    print(f"\n{'class':<35} {'train':>6} {'test':>6}")
    print("-" * 50)
    for cls in classes:
        print(f"  {cls:<33} {split_counts['train'][cls]:>6} {split_counts['test'][cls]:>6}")

    # ── minimum per split ─────────────────────────────────────────────────
    for split in ("train", "test"):
        min_cls = min(split_counts[split], key=split_counts[split].get)
        min_val = split_counts[split][min_cls]
        print(f"\n[{split}] minimum 512×512 count → '{min_cls}' : {min_val}")
