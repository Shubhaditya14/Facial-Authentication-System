#!/usr/bin/env python3
"""
Main training script for Multi-Modal FAS.

Usage:
    python training/train.py --config configs/baseline_rgb.yaml
    python training/train.py --config configs/baseline_rgb.yaml --resume outputs/last.pth
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from utils.config import load_config
from utils.seed import set_seed
from utils.logging import ExperimentLogger
from data.dataloader import create_dataloaders
from data.preprocessing.face_detector import FaceDetector
from models.fas_model import create_model
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Modal FAS Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = load_config(args.config)
    
    if args.seed:
        config["experiment"]["seed"] = args.seed
    if args.device:
        config["hardware"]["device"] = args.device
    
    set_seed(config["experiment"]["seed"])
    
    logger = ExperimentLogger(
        experiment_name=config["experiment"]["name"],
        output_dir=config["experiment"]["output_dir"],
        config=config,
        use_tensorboard=config["logging"].get("use_tensorboard", True),
        use_wandb=config["logging"].get("use_wandb", False),
        wandb_project=config["logging"].get("wandb_project"),
    )
    
    logger.info(f"Starting experiment: {config['experiment']['name']}")
    logger.info(f"Config: {args.config}")
    
    face_detector = None
    if config["data"].get("face_crop", False):
        face_detector = FaceDetector(backend=config["data"].get("face_detector", "mediapipe"))
    
    train_loader, val_loader, test_loader = create_dataloaders(
        config=config,
        face_detector=face_detector,
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    model = create_model(config["model"])
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config['model'].get('backbone_rgb', 'mobilenetv3_large')}")
    logger.info(f"Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={**config["training"], **config["hardware"], "output_dir": config["experiment"]["output_dir"]},
        logger=logger,
    )
    
    if args.resume:
        trainer.resume(args.resume)
    
    history = trainer.fit()
    
    logger.info("Training complete!")
    
    logger.close()


if __name__ == "__main__":
    main()
