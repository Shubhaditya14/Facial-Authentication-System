"""FAS Dataset implementation for multi-modal face anti-spoofing.

Supports CASIA-SURF, OULU-NPU, and generic folder structure datasets.
Label convention: 0 = spoof, 1 = real.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocessing.face_detector import FaceDetector


class FASDataset(Dataset):
    """Multi-modal Face Anti-Spoofing Dataset.

    Supports loading RGB, depth, and IR modalities from various dataset
    structures including CASIA-SURF, OULU-NPU, and generic folder layouts.

    Label Convention:
        0 = spoof/attack
        1 = real/bonafide

    Example:
        >>> from data.transforms import get_train_transforms
        >>> dataset = FASDataset(
        ...     data_root="data/datasets/casia_surf",
        ...     split="train",
        ...     modalities=["rgb", "depth"],
        ...     transform=get_train_transforms(224),
        ... )
        >>> sample = dataset[0]
        >>> print(sample["rgb"].shape)  # (3, 224, 224)
        >>> print(sample["label"])  # tensor(0) or tensor(1)
    """

    # Mapping of dataset names to their loaders
    SUPPORTED_DATASETS = ["casia_surf", "oulu_npu", "replay_attack", "mock", "generic"]

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        modalities: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        face_detector: Optional[FaceDetector] = None,
        protocol: int = 1,
        dataset_name: str = "casia_surf",
    ):
        """Initialize FAS dataset.

        Args:
            data_root: Path to dataset root directory.
            split: Dataset split ("train", "val", or "test").
            modalities: List of modalities to load (["rgb", "depth", "ir"]).
                Defaults to ["rgb"].
            transform: Albumentations transform to apply.
            face_detector: Optional FaceDetector for cropping faces.
            protocol: Dataset protocol number (dataset-specific).
            dataset_name: Name of dataset structure ("casia_surf", "oulu_npu",
                "replay_attack", "mock", "generic").

        Raises:
            ValueError: If dataset_name is not supported or data_root doesn't exist.
        """
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.modalities = modalities or ["rgb"]
        self.transform = transform
        self.face_detector = face_detector
        self.protocol = protocol
        self.dataset_name = dataset_name.lower()

        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported: {self.SUPPORTED_DATASETS}"
            )

        self.samples: List[Dict[str, Any]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Load sample list based on dataset structure."""
        if self.dataset_name == "casia_surf":
            self._load_casia_surf()
        elif self.dataset_name == "oulu_npu":
            self._load_oulu_npu()
        elif self.dataset_name == "replay_attack":
            self._load_replay_attack()
        elif self.dataset_name == "mock":
            self._load_mock()
        else:
            self._load_generic()

    def _load_casia_surf(self) -> None:
        """Load CASIA-SURF dataset structure.

        Expected structure:
        CASIA-SURF/
        ├── Training/
        │   ├── real/{subject_id}/{sample}_rgb.jpg, _depth.jpg, _ir.jpg
        │   └── fake/{subject_id}/{sample}_rgb.jpg, _depth.jpg, _ir.jpg
        ├── Val/
        └── Test/
        """
        # Map split names
        split_map = {
            "train": "Training",
            "val": "Val",
            "test": "Test",
        }
        split_dir = split_map.get(self.split, self.split.capitalize())
        base_path = self.data_root / split_dir

        if not base_path.exists():
            # Try alternative naming
            base_path = self.data_root / self.split
            if not base_path.exists():
                return

        for label_name in ["real", "fake"]:
            label = 1 if label_name == "real" else 0
            label_path = base_path / label_name

            if not label_path.exists():
                continue

            # Walk through subject directories
            for subject_dir in label_path.iterdir():
                if not subject_dir.is_dir():
                    continue

                subject_id = subject_dir.name

                # Find unique sample basenames
                rgb_files = list(subject_dir.glob("*_rgb.jpg")) + \
                           list(subject_dir.glob("*_rgb.png"))

                for rgb_path in rgb_files:
                    basename = rgb_path.name.replace("_rgb.jpg", "").replace("_rgb.png", "")

                    sample = {
                        "rgb_path": str(rgb_path),
                        "depth_path": None,
                        "ir_path": None,
                        "label": label,
                        "attack_type": None if label == 1 else "unknown",
                        "subject_id": subject_id,
                    }

                    # Check for depth
                    for ext in [".jpg", ".png"]:
                        depth_path = subject_dir / f"{basename}_depth{ext}"
                        if depth_path.exists():
                            sample["depth_path"] = str(depth_path)
                            break

                    # Check for IR
                    for ext in [".jpg", ".png"]:
                        ir_path = subject_dir / f"{basename}_ir{ext}"
                        if ir_path.exists():
                            sample["ir_path"] = str(ir_path)
                            break

                    self.samples.append(sample)

    def _load_oulu_npu(self) -> None:
        """Load OULU-NPU dataset structure.

        Expected structure varies by protocol. Generic loading.
        """
        # Map split names
        split_map = {
            "train": "Train",
            "val": "Dev",
            "test": "Test",
        }
        split_dir = split_map.get(self.split, self.split)
        base_path = self.data_root / split_dir

        if not base_path.exists():
            base_path = self.data_root / self.split

        if not base_path.exists():
            return

        # OULU-NPU structure: {split}/{phone_id}_{session_id}_{user_id}_{video_id}.avi
        # or frames extracted
        for video_dir in base_path.iterdir():
            if video_dir.is_dir():
                # Video ID format: phone_session_user_type
                parts = video_dir.name.split("_")
                if len(parts) >= 4:
                    attack_type = parts[3]
                    label = 1 if attack_type == "1" else 0
                    subject_id = parts[2] if len(parts) >= 3 else "unknown"
                else:
                    label = 0
                    attack_type = "unknown"
                    subject_id = "unknown"

                # Find RGB frames
                for rgb_path in video_dir.glob("*.jpg"):
                    self.samples.append({
                        "rgb_path": str(rgb_path),
                        "depth_path": None,
                        "ir_path": None,
                        "label": label,
                        "attack_type": attack_type if label == 0 else None,
                        "subject_id": subject_id,
                    })

    def _load_replay_attack(self) -> None:
        """Load Replay-Attack dataset structure."""
        split_map = {
            "train": "train",
            "val": "devel",
            "test": "test",
        }
        split_dir = split_map.get(self.split, self.split)

        for label_name in ["real", "attack"]:
            label = 1 if label_name == "real" else 0
            label_path = self.data_root / split_dir / label_name

            if not label_path.exists():
                continue

            for img_path in label_path.rglob("*.png"):
                subject_id = img_path.parent.name

                self.samples.append({
                    "rgb_path": str(img_path),
                    "depth_path": None,
                    "ir_path": None,
                    "label": label,
                    "attack_type": label_path.name if label == 0 else None,
                    "subject_id": subject_id,
                })

    def _load_mock(self) -> None:
        """Load mock dataset for testing.

        Expected structure:
        mock/
        ├── train/
        │   ├── real/{id}_rgb.jpg
        │   └── spoof/{id}_rgb.jpg
        """
        base_path = self.data_root / self.split

        if not base_path.exists():
            return

        for label_name in ["real", "spoof"]:
            label = 1 if label_name == "real" else 0
            label_path = base_path / label_name

            if not label_path.exists():
                continue

            for img_path in label_path.glob("*.jpg"):
                self.samples.append({
                    "rgb_path": str(img_path),
                    "depth_path": None,
                    "ir_path": None,
                    "label": label,
                    "attack_type": None,
                    "subject_id": "mock",
                })

            for img_path in label_path.glob("*.png"):
                self.samples.append({
                    "rgb_path": str(img_path),
                    "depth_path": None,
                    "ir_path": None,
                    "label": label,
                    "attack_type": None,
                    "subject_id": "mock",
                })

    def _load_generic(self) -> None:
        """Load generic folder structure.

        Expected structure:
        data/
        ├── {split}/
        │   ├── real/
        │   │   └── *.jpg, *.png
        │   └── spoof/ (or fake/, attack/)
        │       └── *.jpg, *.png
        """
        base_path = self.data_root / self.split

        if not base_path.exists():
            return

        # Try different label folder names
        label_folders = {
            1: ["real", "bonafide", "genuine", "live"],
            0: ["spoof", "fake", "attack", "impostor"],
        }

        for label, folder_names in label_folders.items():
            for folder_name in folder_names:
                label_path = base_path / folder_name

                if not label_path.exists():
                    continue

                for ext in ["*.jpg", "*.png", "*.jpeg"]:
                    for img_path in label_path.rglob(ext):
                        self.samples.append({
                            "rgb_path": str(img_path),
                            "depth_path": None,
                            "ir_path": None,
                            "label": label,
                            "attack_type": None,
                            "subject_id": "generic",
                        })

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing:
            - rgb: Tensor [3, H, W] if "rgb" in modalities
            - depth: Tensor [1, H, W] if "depth" in modalities
            - ir: Tensor [1, H, W] if "ir" in modalities
            - label: Tensor scalar (0 or 1)
            - meta: Dict with attack_type, subject_id, paths
        """
        sample = self.samples[idx]
        result: Dict[str, Any] = {}

        # Load images for requested modalities
        images = {}

        if "rgb" in self.modalities and sample["rgb_path"]:
            rgb = cv2.imread(sample["rgb_path"])
            if rgb is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                images["rgb"] = rgb

        if "depth" in self.modalities and sample.get("depth_path"):
            depth = cv2.imread(sample["depth_path"], cv2.IMREAD_GRAYSCALE)
            if depth is not None:
                images["depth"] = depth

        if "ir" in self.modalities and sample.get("ir_path"):
            ir = cv2.imread(sample["ir_path"], cv2.IMREAD_GRAYSCALE)
            if ir is not None:
                images["ir"] = ir

        # Apply face detection/cropping if enabled
        if self.face_detector and "rgb" in images:
            if len(images) > 1:
                # Multi-modal: use aligned cropping
                images = self.face_detector.crop_aligned(images)
            else:
                # Single modal: just crop RGB
                images["rgb"] = self.face_detector.crop_face(images["rgb"])

        # Apply transforms
        if self.transform and "rgb" in images:
            # Standard transform (RGB only or handles multi-modal)
            if hasattr(self.transform, "additional_targets"):
                # Multi-modal transform
                transform_input = {"image": images.get("rgb")}
                if "depth" in images:
                    depth_3ch = np.repeat(
                        np.expand_dims(images["depth"], -1), 3, axis=-1
                    )
                    transform_input["depth"] = depth_3ch
                if "ir" in images:
                    ir_3ch = np.repeat(
                        np.expand_dims(images["ir"], -1), 3, axis=-1
                    )
                    transform_input["ir"] = ir_3ch

                transformed = self.transform(**transform_input)
                result["rgb"] = transformed["image"]

                if "depth" in transformed:
                    # Extract single channel
                    if isinstance(transformed["depth"], torch.Tensor):
                        result["depth"] = transformed["depth"][0:1]
                    else:
                        result["depth"] = torch.from_numpy(
                            transformed["depth"][:, :, 0:1].transpose(2, 0, 1)
                        ).float() / 255.0

                if "ir" in transformed:
                    if isinstance(transformed["ir"], torch.Tensor):
                        result["ir"] = transformed["ir"][0:1]
                    else:
                        result["ir"] = torch.from_numpy(
                            transformed["ir"][:, :, 0:1].transpose(2, 0, 1)
                        ).float() / 255.0
            else:
                # Standard RGB transform
                transformed = self.transform(image=images.get("rgb"))
                result["rgb"] = transformed["image"]

                # Handle depth/IR separately if present
                if "depth" in images:
                    depth = images["depth"]
                    if len(depth.shape) == 2:
                        depth = np.expand_dims(depth, axis=0)
                    result["depth"] = torch.from_numpy(depth.astype(np.float32)) / 255.0

                if "ir" in images:
                    ir = images["ir"]
                    if len(ir.shape) == 2:
                        ir = np.expand_dims(ir, axis=0)
                    result["ir"] = torch.from_numpy(ir.astype(np.float32)) / 255.0
        else:
            # No transform, convert to tensors directly
            for mod in self.modalities:
                if mod in images:
                    img = images[mod]
                    if len(img.shape) == 2:
                        img = np.expand_dims(img, axis=0)
                    elif len(img.shape) == 3:
                        img = img.transpose(2, 0, 1)
                    result[mod] = torch.from_numpy(img.astype(np.float32)) / 255.0

        # Add None for missing modalities
        for mod in self.modalities:
            if mod not in result:
                result[mod] = None

        # Label
        result["label"] = torch.tensor(sample["label"], dtype=torch.long)

        # Metadata
        result["meta"] = {
            "attack_type": sample.get("attack_type"),
            "subject_id": sample.get("subject_id"),
            "rgb_path": sample.get("rgb_path"),
            "depth_path": sample.get("depth_path"),
            "ir_path": sample.get("ir_path"),
        }

        return result

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets.

        Returns:
            Tensor of shape (2,) with weights for [spoof, real] classes.
        """
        labels = [s["label"] for s in self.samples]
        num_spoof = labels.count(0)
        num_real = labels.count(1)
        total = len(labels)

        if num_spoof == 0 or num_real == 0:
            return torch.ones(2)

        weight_spoof = total / (2 * num_spoof)
        weight_real = total / (2 * num_real)

        return torch.tensor([weight_spoof, weight_real], dtype=torch.float32)

    def get_stats(self) -> Dict[str, int]:
        """Get dataset statistics.

        Returns:
            Dictionary with sample counts.
        """
        labels = [s["label"] for s in self.samples]
        return {
            "total": len(self.samples),
            "real": labels.count(1),
            "spoof": labels.count(0),
        }
