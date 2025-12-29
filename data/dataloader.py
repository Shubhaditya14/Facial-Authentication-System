"""DataLoader factory for FAS datasets.

Provides convenient functions to create train/val/test dataloaders
from configuration.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from data.dataset import FASDataset
from data.preprocessing.face_detector import FaceDetector
from data.transforms import get_train_transforms, get_val_transforms
from utils.seed import worker_init_fn


def create_dataloaders(
    config: Dict[str, Any],
    face_detector: Optional[FaceDetector] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders from config.

    Args:
        config: Configuration dictionary with keys:
            - data.data_root: Path to dataset
            - data.dataset_name: Name of dataset
            - data.modalities: List of modalities
            - data.img_size: Image size
            - data.batch_size: Batch size
            - data.num_workers: Number of workers
            - data.pin_memory: Pin memory flag
            - data.protocol: Dataset protocol
            - data.augmentation: Augmentation config (optional)
        face_detector: Optional face detector for cropping.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Example:
        >>> from utils.config import load_config
        >>> config = load_config("configs/baseline_rgb.yaml")
        >>> train_loader, val_loader, test_loader = create_dataloaders(config)
    """
    # Extract data config (handle both dict and Config object)
    if hasattr(config, "to_dict"):
        config = config.to_dict()

    data_config = config.get("data", config)

    # Get parameters with defaults
    data_root = data_config.get("data_root", "data/datasets")
    dataset_name = data_config.get("dataset_name", "casia_surf")
    modalities = data_config.get("modalities", ["rgb"])
    img_size = data_config.get("img_size", 224)
    batch_size = data_config.get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)
    pin_memory = data_config.get("pin_memory", True)
    protocol = data_config.get("protocol", 1)
    aug_config = data_config.get("augmentation", {})

    # Create transforms
    train_transform = get_train_transforms(img_size, aug_config)
    val_transform = get_val_transforms(img_size)

    # Create datasets
    train_dataset = FASDataset(
        data_root=data_root,
        split="train",
        modalities=modalities,
        transform=train_transform,
        face_detector=face_detector,
        protocol=protocol,
        dataset_name=dataset_name,
    )

    val_dataset = FASDataset(
        data_root=data_root,
        split="val",
        modalities=modalities,
        transform=val_transform,
        face_detector=face_detector,
        protocol=protocol,
        dataset_name=dataset_name,
    )

    test_dataset = FASDataset(
        data_root=data_root,
        split="test",
        modalities=modalities,
        transform=val_transform,
        face_detector=face_detector,
        protocol=protocol,
        dataset_name=dataset_name,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


def create_single_dataloader(
    data_root: str,
    split: str,
    modalities: Optional[List[str]] = None,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: Optional[bool] = None,
    dataset_name: str = "casia_surf",
    protocol: int = 1,
    face_detector: Optional[FaceDetector] = None,
    augmentation: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    """Create a single dataloader.

    Args:
        data_root: Path to dataset root.
        split: Dataset split ("train", "val", or "test").
        modalities: List of modalities. Defaults to ["rgb"].
        batch_size: Batch size.
        img_size: Image size.
        num_workers: Number of workers.
        pin_memory: Pin memory flag.
        shuffle: Whether to shuffle. Defaults to True for train, False otherwise.
        dataset_name: Name of dataset.
        protocol: Dataset protocol.
        face_detector: Optional face detector.
        augmentation: Augmentation config for training.

    Returns:
        DataLoader instance.
    """
    modalities = modalities or ["rgb"]

    # Determine shuffle
    if shuffle is None:
        shuffle = split == "train"

    # Create transform based on split
    if split == "train":
        transform = get_train_transforms(img_size, augmentation)
    else:
        transform = get_val_transforms(img_size)

    # Create dataset
    dataset = FASDataset(
        data_root=data_root,
        split=split,
        modalities=modalities,
        transform=transform,
        face_detector=face_detector,
        protocol=protocol,
        dataset_name=dataset_name,
    )

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=(split == "train"),
    )

    return loader


def collate_multimodal(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for multi-modal batches.

    Handles batches where some modalities might be None for certain samples.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Collated batch dictionary with stacked tensors.
    """
    result = {}

    # Get all keys from first sample
    keys = batch[0].keys()

    for key in keys:
        if key == "meta":
            # Handle metadata separately
            result["meta"] = {
                meta_key: [sample["meta"][meta_key] for sample in batch]
                for meta_key in batch[0]["meta"].keys()
            }
        elif key in ["rgb", "depth", "ir"]:
            # Stack tensors, handling None values
            values = [sample[key] for sample in batch]
            if all(v is not None for v in values):
                result[key] = torch.stack(values)
            else:
                result[key] = None
        elif key == "label":
            # Stack labels
            result["label"] = torch.stack([sample["label"] for sample in batch])
        else:
            # Generic handling
            values = [sample[key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values)
            else:
                result[key] = values

    return result


class InfiniteDataLoader:
    """DataLoader wrapper that loops infinitely.

    Useful for training when you want to iterate by steps rather than epochs.

    Example:
        >>> loader = InfiniteDataLoader(train_loader)
        >>> for step in range(10000):
        ...     batch = next(loader)
        ...     train_step(batch)
    """

    def __init__(self, dataloader: DataLoader):
        """Initialize infinite dataloader.

        Args:
            dataloader: Base DataLoader to wrap.
        """
        self.dataloader = dataloader
        self._iterator = iter(dataloader)

    def __iter__(self):
        """Return self as iterator."""
        return self

    def __next__(self):
        """Get next batch, restarting if exhausted."""
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            batch = next(self._iterator)
        return batch

    def __len__(self):
        """Return length of underlying dataloader."""
        return len(self.dataloader)
