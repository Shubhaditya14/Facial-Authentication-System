"""Test script to verify all components work."""

import os
import shutil
import sys

import numpy as np


def test_config():
    """Test config loading and Config class."""
    from utils.config import load_config, Config

    config = load_config("configs/baseline_rgb.yaml")
    assert "model" in config
    assert "data" in config
    print("✓ Config loading works")

    # Test Config wrapper
    cfg = Config(config)
    assert cfg.model.backbone == "resnet18"
    assert cfg.data.img_size == 224
    print("✓ Config dot notation works")


def test_seed():
    """Test seed reproducibility."""
    from utils.seed import set_seed
    import torch

    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)
    print("✓ Seed reproducibility works")


def test_metrics():
    """Test FAS metrics calculations."""
    from utils.metrics import (
        FASMetrics,
        calculate_eer,
        calculate_acer,
        calculate_apcer,
        calculate_bpcer,
        calculate_auc,
    )

    # Simulated predictions and labels
    # Higher values = more likely real (label 1)
    preds = np.array([0.9, 0.8, 0.3, 0.2, 0.7, 0.1])
    labels = np.array([1, 1, 0, 0, 1, 0])

    # Test individual metrics
    apcer = calculate_apcer(preds, labels, threshold=0.5)
    bpcer = calculate_bpcer(preds, labels, threshold=0.5)
    acer = calculate_acer(preds, labels, threshold=0.5)
    eer, threshold = calculate_eer(preds, labels)
    auc = calculate_auc(preds, labels)

    assert 0 <= apcer <= 1, f"APCER out of range: {apcer}"
    assert 0 <= bpcer <= 1, f"BPCER out of range: {bpcer}"
    assert 0 <= acer <= 1, f"ACER out of range: {acer}"
    assert 0 <= eer <= 1, f"EER out of range: {eer}"
    assert 0 <= auc <= 1, f"AUC out of range: {auc}"

    print(f"✓ Individual metrics work - APCER: {apcer:.4f}, BPCER: {bpcer:.4f}, ACER: {acer:.4f}")
    print(f"  EER: {eer:.4f} @ threshold {threshold:.4f}, AUC: {auc:.4f}")

    # Test FASMetrics accumulator
    metrics = FASMetrics(threshold=0.5)
    metrics.update(preds[:3], labels[:3])
    metrics.update(preds[3:], labels[3:])
    results = metrics.compute()

    assert "acer" in results
    assert "eer" in results
    assert results["num_samples"] == 6
    print(f"✓ FASMetrics accumulator works - {results['num_samples']} samples")


def test_face_detector():
    """Test face detector initialization and methods."""
    from data.preprocessing.face_detector import FaceDetector

    detector = FaceDetector(backend="mediapipe")

    # Create dummy image (random pixels - may not detect face)
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test detect (may return empty list for random image)
    detections = detector.detect(dummy_img)
    assert isinstance(detections, list)

    # Test crop_face (should return original if no face)
    result = detector.crop_face(dummy_img)
    assert result is not None
    assert result.shape[0] > 0 and result.shape[1] > 0

    # Test crop_aligned
    multi_result = detector.crop_aligned({
        "rgb": dummy_img,
        "depth": dummy_img[:, :, 0],
    })
    assert "rgb" in multi_result
    assert "depth" in multi_result

    detector.close()
    print("✓ Face detector works")


def test_transforms():
    """Test image transforms."""
    from data.transforms import (
        get_train_transforms,
        get_val_transforms,
        get_multimodal_transforms,
    )

    train_tf = get_train_transforms(224)
    val_tf = get_val_transforms(224)

    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    train_result = train_tf(image=dummy_img)
    val_result = val_tf(image=dummy_img)

    assert train_result["image"].shape == (3, 224, 224), f"Got {train_result['image'].shape}"
    assert val_result["image"].shape == (3, 224, 224), f"Got {val_result['image'].shape}"
    print("✓ Train/Val transforms work")

    # Test multi-modal transforms
    mm_tf = get_multimodal_transforms(224, train=True)
    depth_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    mm_result = mm_tf(image=dummy_img, depth=depth_img)
    assert "image" in mm_result
    assert "depth" in mm_result
    print("✓ Multi-modal transforms work")


def test_dataset_mock():
    """Test dataset with mock data structure."""
    from data.dataset import FASDataset
    from data.transforms import get_train_transforms
    from PIL import Image

    # Create mock dataset structure
    mock_root = "data/datasets/mock_test"
    os.makedirs(f"{mock_root}/train/real", exist_ok=True)
    os.makedirs(f"{mock_root}/train/spoof", exist_ok=True)

    # Create dummy images
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(f"{mock_root}/train/real/{i}_rgb.jpg")
        img.save(f"{mock_root}/train/spoof/{i}_rgb.jpg")

    # Test dataset
    dataset = FASDataset(
        data_root=mock_root,
        split="train",
        modalities=["rgb"],
        transform=get_train_transforms(224),
        dataset_name="mock"
    )

    assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"

    sample = dataset[0]
    assert "rgb" in sample
    assert "label" in sample
    assert sample["rgb"].shape == (3, 224, 224), f"Got {sample['rgb'].shape}"

    stats = dataset.get_stats()
    assert stats["total"] == 10
    assert stats["real"] == 5
    assert stats["spoof"] == 5

    print(f"✓ Dataset works - {len(dataset)} samples loaded")
    print(f"  Stats: {stats}")

    # Cleanup
    shutil.rmtree(mock_root)
    print("✓ Mock data cleaned up")


def test_dataloader():
    """Test dataloader creation."""
    from data.dataset import FASDataset
    from data.dataloader import create_single_dataloader, InfiniteDataLoader
    from data.transforms import get_train_transforms
    from torch.utils.data import DataLoader
    from PIL import Image

    # Create mock data
    mock_root = "data/datasets/mock_loader_test"
    os.makedirs(f"{mock_root}/train/real", exist_ok=True)
    os.makedirs(f"{mock_root}/train/spoof", exist_ok=True)

    for i in range(8):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(f"{mock_root}/train/real/{i}_rgb.jpg")
        img.save(f"{mock_root}/train/spoof/{i}_rgb.jpg")

    # Test create_single_dataloader
    loader = create_single_dataloader(
        data_root=mock_root,
        split="train",
        batch_size=4,
        num_workers=0,  # Use 0 for testing
        dataset_name="mock",
    )

    batch = next(iter(loader))
    assert batch["rgb"].shape[0] == 4  # batch size
    assert batch["rgb"].shape[1:] == (3, 224, 224)
    assert batch["label"].shape[0] == 4
    print("✓ Single dataloader works")

    # Test InfiniteDataLoader
    inf_loader = InfiniteDataLoader(loader)
    for i in range(5):  # Get 5 batches (more than dataset)
        batch = next(inf_loader)
        assert batch["rgb"].shape[0] == 4
    print("✓ Infinite dataloader works")

    # Cleanup
    shutil.rmtree(mock_root)


def test_logging():
    """Test logging utilities."""
    from utils.logging import setup_logger, get_logger, ExperimentLogger
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test basic logger
        logger = setup_logger("test_logger", log_dir=tmpdir)
        logger.info("Test message")
        print("✓ Basic logger works")

        # Test get_logger
        same_logger = get_logger("test_logger")
        assert same_logger is logger
        print("✓ get_logger works")

        # Test ExperimentLogger
        exp_logger = ExperimentLogger(
            name="test_exp",
            log_dir=tmpdir,
            use_tensorboard=False,  # Skip TensorBoard for faster test
            use_wandb=False,
        )
        exp_logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=1)
        exp_logger.info("Test experiment message")
        exp_logger.close()
        print("✓ ExperimentLogger works")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 50)
    print("Running Pipeline Tests")
    print("=" * 50 + "\n")

    tests = [
        ("Config", test_config),
        ("Seed", test_seed),
        ("Metrics", test_metrics),
        ("Face Detector", test_face_detector),
        ("Transforms", test_transforms),
        ("Dataset", test_dataset_mock),
        ("DataLoader", test_dataloader),
        ("Logging", test_logging),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        print(f"\n--- Testing {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"✗ {name} failed: {e}")

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if errors:
        print("\nErrors:")
        for name, error in errors:
            print(f"  - {name}: {error}")
        print()
        return False
    else:
        print("\nAll tests passed! ✓")
        print("=" * 50 + "\n")
        return True


if __name__ == "__main__":
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    success = run_all_tests()
    sys.exit(0 if success else 1)
