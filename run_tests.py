#!/usr/bin/env python3
"""
Run all tests for the Multi-Modal FAS project.
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tests.test_pipeline import (
    test_config, test_seed, test_metrics, test_face_detector,
    test_transforms, test_dataset_mock, test_dataloader, test_logging,
    test_backbone, test_fusion, test_head, test_full_model,
    test_model_with_dataloader, test_losses, test_callbacks, test_trainer_init
)

def run_tests():
    print("\n" + "=" * 60)
    print("Running Multi-Modal FAS Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Config", test_config),
        ("Seed", test_seed),
        ("Metrics", test_metrics),
        ("Face Detector", test_face_detector),
        ("Transforms", test_transforms),
        ("Dataset", test_dataset_mock),
        ("DataLoader", test_dataloader),
        ("Logging", test_logging),
        ("Backbone", test_backbone),
        ("Fusion", test_fusion),
        ("Head", test_head),
        ("Full Model", test_full_model),
        ("Model with DataLoader", test_model_with_dataloader),
        ("Losses", test_losses),
        ("Callbacks", test_callbacks),
        ("Trainer", test_trainer_init),
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

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if errors:
        print("\nErrors:")
        for name, error in errors:
            print(f"  - {name}: {error}")
        print()
        return False
    else:
        print("\nAll tests passed! ✓")
        print("=" * 60 + "\n")
        return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
