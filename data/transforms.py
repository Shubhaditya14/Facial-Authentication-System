"""Image transforms using Albumentations.

Provides training and validation transforms for FAS, including
multi-modal transforms that apply consistent spatial transformations
across RGB, depth, and IR images.
"""

from typing import Any, Dict, List, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Grayscale normalization (for depth/IR)
GRAYSCALE_MEAN = [0.5]
GRAYSCALE_STD = [0.5]


def get_train_transforms(
    img_size: int = 224,
    config: Optional[Dict[str, Any]] = None,
) -> A.Compose:
    """Get training augmentations for RGB images.

    Augmentations are designed to be safe for FAS - they don't destroy
    spoof cues that the model needs to learn (e.g., moire patterns,
    print artifacts).

    Args:
        img_size: Output image size (square).
        config: Optional augmentation config dict with keys:
            - horizontal_flip: bool (default: True)
            - rotation: int, max rotation degrees (default: 10)
            - brightness: float (default: 0.2)
            - contrast: float (default: 0.2)
            - saturation: float (default: 0.2)
            - hue: float (default: 0.1)
            - gaussian_noise: float, noise variance (default: 0.02)

    Returns:
        Albumentations Compose transform.
    """
    config = config or {}

    transforms = [
        # Resize with aspect ratio preservation
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,
            fill=0,
        ),
        # Center crop to exact size
        A.CenterCrop(height=img_size, width=img_size),
    ]

    # Horizontal flip
    if config.get("horizontal_flip", True):
        transforms.append(A.HorizontalFlip(p=0.5))

    # Rotation (mild - preserve spoof cues)
    rotation = config.get("rotation", 10)
    if rotation > 0:
        transforms.append(
            A.Rotate(
                limit=rotation,
                border_mode=0,
                p=0.5,
            )
        )

    # Color jitter (mild - preserve color-based spoof cues)
    brightness = config.get("brightness", 0.2)
    contrast = config.get("contrast", 0.2)
    saturation = config.get("saturation", 0.2)
    hue = config.get("hue", 0.1)

    if any([brightness, contrast, saturation, hue]):
        transforms.append(
            A.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
                p=0.5,
            )
        )

    # Gaussian noise (very mild - preserve fine-grained artifacts)
    gaussian_noise = config.get("gaussian_noise", 0.02)
    if gaussian_noise > 0:
        transforms.append(
            A.GaussNoise(
                std_range=(0, gaussian_noise),
                p=0.3,
            )
        )

    # Normalize and convert to tensor
    transforms.extend([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return A.Compose(transforms)


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """Get validation/test transforms (no augmentation).

    Args:
        img_size: Output image size (square).

    Returns:
        Albumentations Compose transform.
    """
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,
            fill=0,
        ),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_multimodal_transforms(
    img_size: int = 224,
    train: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> A.Compose:
    """Get multi-modal transforms with additional_targets for depth and IR.

    Applies consistent spatial transforms (flip, rotation, crop) across
    all modalities while only applying color transforms to RGB.

    Args:
        img_size: Output image size (square).
        train: If True, include training augmentations.
        config: Optional augmentation config (see get_train_transforms).

    Returns:
        Albumentations Compose with additional_targets for 'depth' and 'ir'.

    Example:
        >>> transform = get_multimodal_transforms(224, train=True)
        >>> result = transform(image=rgb, depth=depth_img, ir=ir_img)
        >>> rgb_tensor = result['image']
        >>> depth_tensor = result['depth']
        >>> ir_tensor = result['ir']
    """
    config = config or {}

    # Base spatial transforms (applied to all modalities)
    transforms = [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,
            fill=0,  # Scalar for grayscale compatibility
        ),
        A.CenterCrop(height=img_size, width=img_size),
    ]

    if train:
        # Horizontal flip (spatial - all modalities)
        if config.get("horizontal_flip", True):
            transforms.append(A.HorizontalFlip(p=0.5))

        # Rotation (spatial - all modalities)
        rotation = config.get("rotation", 10)
        if rotation > 0:
            transforms.append(
                A.Rotate(
                    limit=rotation,
                    border_mode=0,
                    p=0.5,
                )
            )

    return A.Compose(
        transforms,
        additional_targets={
            "depth": "image",
            "ir": "image",
        },
    )


def get_rgb_normalize() -> A.Compose:
    """Get RGB normalization transform only.

    Returns:
        Albumentations Compose for RGB normalization.
    """
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_grayscale_normalize() -> A.Compose:
    """Get grayscale (depth/IR) normalization transform.

    Returns:
        Albumentations Compose for grayscale normalization.
    """
    return A.Compose([
        A.Normalize(mean=GRAYSCALE_MEAN, std=GRAYSCALE_STD),
        ToTensorV2(),
    ])


class MultiModalTransform:
    """Multi-modal transform wrapper that handles normalization separately.

    Applies spatial transforms consistently and normalizes each modality
    appropriately (RGB with ImageNet stats, depth/IR with grayscale stats).

    Example:
        >>> transform = MultiModalTransform(img_size=224, train=True)
        >>> result = transform(rgb=rgb_img, depth=depth_img, ir=ir_img)
        >>> rgb_tensor = result['rgb']  # [3, 224, 224]
        >>> depth_tensor = result['depth']  # [1, 224, 224]
    """

    def __init__(
        self,
        img_size: int = 224,
        train: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize multi-modal transform.

        Args:
            img_size: Output image size.
            train: Whether to use training augmentations.
            config: Augmentation configuration.
        """
        self.spatial_transform = get_multimodal_transforms(
            img_size=img_size,
            train=train,
            config=config,
        )
        self.rgb_normalize = get_rgb_normalize()
        self.grayscale_normalize = get_grayscale_normalize()

    def __call__(
        self,
        rgb: Optional["np.ndarray"] = None,
        depth: Optional["np.ndarray"] = None,
        ir: Optional["np.ndarray"] = None,
    ) -> Dict[str, Any]:
        """Apply transforms to multi-modal inputs.

        Args:
            rgb: RGB image (H, W, 3).
            depth: Depth image (H, W) or (H, W, 1).
            ir: IR image (H, W) or (H, W, 1).

        Returns:
            Dictionary with transformed tensors for each provided modality.
        """
        import numpy as np

        result = {}

        # Prepare inputs for spatial transform
        spatial_input = {}
        if rgb is not None:
            spatial_input["image"] = rgb
        if depth is not None:
            # Ensure 3D for albumentations
            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, axis=-1)
            # Repeat to 3 channels for spatial transform
            depth_3ch = np.repeat(depth, 3, axis=-1) if depth.shape[-1] == 1 else depth
            spatial_input["depth"] = depth_3ch
        if ir is not None:
            if len(ir.shape) == 2:
                ir = np.expand_dims(ir, axis=-1)
            ir_3ch = np.repeat(ir, 3, axis=-1) if ir.shape[-1] == 1 else ir
            spatial_input["ir"] = ir_3ch

        # Apply spatial transforms
        if spatial_input:
            transformed = self.spatial_transform(**spatial_input)

            # Normalize each modality appropriately
            if "image" in transformed and rgb is not None:
                rgb_norm = self.rgb_normalize(image=transformed["image"])
                result["rgb"] = rgb_norm["image"]

            if "depth" in transformed and depth is not None:
                # Convert back to single channel
                depth_1ch = transformed["depth"][:, :, 0:1]
                depth_norm = self.grayscale_normalize(image=depth_1ch)
                result["depth"] = depth_norm["image"]

            if "ir" in transformed and ir is not None:
                ir_1ch = transformed["ir"][:, :, 0:1]
                ir_norm = self.grayscale_normalize(image=ir_1ch)
                result["ir"] = ir_norm["image"]

        return result
