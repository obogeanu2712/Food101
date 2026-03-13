from torch.utils.data import DataLoader
from .dataset import Food101Dataset
from .transforms import train_transform, val_transform


def get_dataloaders(
    root,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
):
    """
    Build train and test DataLoaders for Food-101.

    Parameters
    ----------
    root : str | Path
        Root directory of the Food-101 dataset.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of subprocesses for data loading.
    pin_memory : bool
        Pin tensors in memory for faster GPU transfers.

    Returns
    -------
    train_loader : DataLoader
    test_loader  : DataLoader
    """
    train_dataset = Food101Dataset(root, split="train", transform=train_transform)
    test_dataset  = Food101Dataset(root, split="test",  transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader
