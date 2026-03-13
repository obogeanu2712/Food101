from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class Food101Dataset(Dataset):
    """
    PyTorch Dataset for the Food-101 benchmark.

    Parameters
    ----------
    root : str | Path
        Root directory that contains the ``images/`` and ``meta/`` folders.
    split : {"train", "test"}
        Which split to load.
    transform : callable, optional
        Transform applied to the PIL image before returning it.
    """

    def __init__(self, root, split="train", transform=None):
        assert split in ("train", "test"), "split must be 'train' or 'test'"

        self.root = Path(root)
        self.split = split
        self.transform = transform

        # ── class list ────────────────────────────────────────────────────
        classes_file = self.root / "meta" / "classes_curated.txt"
        with open(classes_file) as f:
            self.classes = [line.strip() for line in f if line.strip()]

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # ── split file: each line is "class_name/image_id" ────────────────
        split_file = self.root / "meta" / f"{split}_curated.txt"
        with open(split_file) as f:
            self.samples = [line.strip() for line in f if line.strip()]
        # self.samples[i] → e.g. "apple_pie/1005649"

    # ── helpers ───────────────────────────────────────────────────────────

    def _image_path(self, rel_path: str) -> Path:
        """Return the absolute path to a JPEG given 'class/id'."""
        return self.root / "images" / (rel_path + ".jpg")

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path = self.samples[idx]                       # "apple_pie/1005649"
        class_name = rel_path.split("/")[0]                # "apple_pie"
        label = self.class_to_idx[class_name]              # integer index

        image = Image.open(self._image_path(rel_path)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __repr__(self):
        return (
            f"Food101Dataset(split={self.split!r}, "
            f"num_classes={len(self.classes)}, "
            f"num_samples={len(self.samples)})"
        )
