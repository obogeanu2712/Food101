"""
curate.py
---------
For each split scan all 512×512 images per class.
  • Discard any class that has fewer than MIN_PER_CLASS[split] qualifying images.
  • From the remaining classes keep exactly MAX_PER_CLASS[split] samples (first
    encountered in split order).

Outputs
-------
  meta/train_curated.txt   – kept sample paths  (class_name/image_id)
  meta/test_curated.txt    – kept sample paths
  notes.txt                – human-readable summary of excluded classes + counts
"""

from collections import defaultdict
from pathlib import Path

from PIL import Image

from data.dataset import Food101Dataset

ROOT = Path(__file__).parent

# minimum 512×512 representatives a class must have to be included
MIN_PER_CLASS  = {"train": 400, "test": 100}
# exact number of samples to keep per surviving class
MAX_PER_CLASS  = {"train": 400, "test": 100}


def curate_split(
    split: str,
    allowed_classes: set[str] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Return (kept_paths, excluded_counts).
    
    If allowed_classes is given, any class not in that set is excluded
    regardless of its 512×512 count (used to enforce train→test class parity).
    """
    dataset  = Food101Dataset(ROOT, split=split, transform=None)
    min_keep = MIN_PER_CLASS[split]
    max_keep = MAX_PER_CLASS[split]
    print(f"\n[{split}]  scanning {len(dataset)} samples  "
          f"(min={min_keep}, keep={max_keep}) ...")

    all_512: dict[str, list[str]] = defaultdict(list)
    for idx, rel_path in enumerate(dataset.samples):
        class_name = rel_path.split("/")[0]
        with Image.open(dataset._image_path(rel_path)) as img:
            w, h = img.size
        if w == 512 and h == 512:
            all_512[class_name].append(rel_path)
        if (idx + 1) % 10000 == 0:
            print(f"  {idx + 1}/{len(dataset)} ...")

    excluded: dict[str, int] = {}
    curated:  list[str]      = []

    for cls in dataset.classes:
        paths = all_512.get(cls, [])
        if allowed_classes is not None and cls not in allowed_classes:
            excluded[cls] = len(paths)  # dropped due to train exclusion
        elif len(paths) < min_keep:
            excluded[cls] = len(paths)
        else:
            curated.extend(paths[:max_keep])

    print(f"  kept classes  : {len(dataset.classes) - len(excluded)} / {len(dataset.classes)}")
    print(f"  kept samples  : {len(curated)}")
    print(f"  excluded      : {len(excluded)} classes")
    if excluded:
        for cls, n in sorted(excluded.items(), key=lambda x: x[1]):
            print(f"    {cls:<35} {n}")

    return curated, excluded


if __name__ == "__main__":
    notes_lines: list[str] = ["curate.py – curation summary", "=" * 50]
    all_excluded: dict[str, dict[str, int]] = {}
    kept_train_classes: set[str] | None = None

    for split in ("train", "test"):
        curated, excluded = curate_split(split, allowed_classes=kept_train_classes)
        all_excluded[split] = excluded

        if split == "train":
            dataset = Food101Dataset(ROOT, split="train", transform=None)
            kept_train_classes = set(dataset.classes) - set(excluded.keys())

        out_path = ROOT / "meta" / f"{split}_curated.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(curated) + "\n")
        print(f"  saved → {out_path}  ({len(curated)} lines)")

        notes_lines += [
            "",
            f"[{split}]",
            f"  threshold     : {MIN_PER_CLASS[split]} (min 512×512 to qualify)",
            f"  kept per class: {MAX_PER_CLASS[split]}",
            f"  kept samples  : {len(curated)}",
            f"  excluded classes ({len(excluded)}):",
        ]
        if excluded:
            for cls, n in sorted(excluded.items(), key=lambda x: x[1]):
                notes_lines.append(f"    {cls:<35} {n:>4} 512×512 images")
        else:
            notes_lines.append("    (none)")

    notes_path = ROOT / "notes.txt"
    with open(notes_path, "w") as f:
        f.write("\n".join(notes_lines) + "\n")
    print(f"\nNotes saved → {notes_path}")