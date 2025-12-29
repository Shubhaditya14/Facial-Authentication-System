import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import time

from utils.metrics import FASMetrics
from utils.logging import ExperimentLogger
from .losses import FASLoss
from .callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback


class Trainer:
    """
    Trainer for Multi-Modal FAS Model.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Multi-loss training
    - Early stopping & checkpointing
    - W&B / TensorBoard logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        logger: Optional[ExperimentLogger] = None,
    ):
        """
        Args:
            model: FAS model
            train_loader: training dataloader
            val_loader: validation dataloader
            config: training configuration
            logger: experiment logger
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        
        self.use_amp = config.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        self.accumulation_steps = config.get("gradient_accumulation_steps", 1)
        
        self.criterion = FASLoss(
            bce_weight=config.get("bce_weight", 1.0),
            depth_weight=config.get("depth_weight", 0.5),
            contrastive_weight=config.get("contrastive_weight", 0.1),
            label_smoothing=config.get("label_smoothing", 0.0),
        )
        
        self.optimizer = self._create_optimizer()
        
        self.scheduler = self._create_scheduler()
        self.scheduler_callback = LRSchedulerCallback(self.scheduler, step_on="epoch")
        
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 10),
            mode="min",
        )
        
        self.checkpoint = ModelCheckpoint(
            save_dir=config.get("output_dir", "outputs"),
            monitor="val_acer",
            mode="min",
        )
        
        self.train_metrics = FASMetrics()
        self.val_metrics = FASMetrics()
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_acer = float("inf")
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_name = self.config.get("optimizer", "adamw").lower()
        lr = self.config.get("lr", 1e-3)
        weight_decay = self.config.get("weight_decay", 1e-4)
        
        if opt_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
            
    def _create_scheduler(self):
        """Create LR scheduler from config."""
        scheduler_name = self.config.get("scheduler", "cosine").lower()
        epochs = self.config.get("epochs", 50)
        warmup_epochs = self.config.get("warmup_epochs", 5)
        min_lr = self.config.get("min_lr", 1e-6)
        
        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr
            )
        elif scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )
        else:
            return torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0.0
        loss_components = {"bce": 0.0, "depth": 0.0, "contrastive": 0.0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [Train]")
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            batch = self._to_device(batch)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model.forward_dict(batch)
                targets = {"label": batch["label"], "depth": batch.get("depth")}
                losses = self.criterion(outputs, targets)
                loss = losses["total"] / self.accumulation_steps
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            with torch.no_grad():
                scores = outputs["score"].squeeze().cpu().numpy()
                labels = batch["label"].cpu().numpy()
                self.train_metrics.update(scores, labels)
            
            total_loss += losses["total"].item()
            for k in loss_components:
                loss_components[k] += losses[k].item()
            
            pbar.set_postfix({"loss": f"{losses['total'].item():.4f}"})
        
        num_batches = len(self.train_loader)
        metrics = self.train_metrics.compute()
        metrics["loss"] = total_loss / num_batches
        for k in loss_components:
            metrics[f"loss_{k}"] = loss_components[k] / num_batches
            
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch} [Val]")
        
        for batch in pbar:
            batch = self._to_device(batch)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model.forward_dict(batch)
                targets = {"label": batch["label"], "depth": batch.get("depth")}
                losses = self.criterion(outputs, targets)
            
            total_loss += losses["total"].item()
            
            scores = outputs["score"].squeeze().cpu().numpy()
            labels = batch["label"].cpu().numpy()
            self.val_metrics.update(scores, labels)
        
        metrics = self.val_metrics.compute()
        metrics["loss"] = total_loss / len(self.val_loader)
        
        return metrics
    
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device."""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result
    
    def fit(self) -> Dict[str, Any]:
        """
        Full training loop.
        
        Returns:
            Training history dict
        """
        epochs = self.config.get("epochs", 50)
        history = {"train": [], "val": []}
        
        print(f"\n{'='*60}")
        print(f"Starting training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Accumulation Steps: {self.accumulation_steps}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            train_metrics = self.train_epoch()
            history["train"].append(train_metrics)
            
            val_metrics = self.validate()
            history["val"].append(val_metrics)
            
            self.scheduler_callback.step_epoch(val_metrics.get("acer", val_metrics["loss"]))
            
            epoch_time = time.time() - epoch_start
            self._log_epoch(epoch, train_metrics, val_metrics, epoch_time)
            
            saved_path = self.checkpoint.save(
                self.model, self.optimizer, epoch, val_metrics, self.scheduler
            )
            if saved_path:
                self.best_val_acer = val_metrics["acer"]
                print(f"  ✓ Saved best model (ACER: {self.best_val_acer:.4f})")
            
            if self.early_stopping(val_metrics["acer"]):
                print(f"\n Early stopping triggered at epoch {epoch}")
                break
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best Val ACER: {self.best_val_acer:.4f}")
        print(f"{'='*60}\n")
        
        return history
    
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float,
    ):
        """Log epoch results."""
        print(f"\nEpoch {epoch} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, ACER: {train_metrics['acer']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, ACER: {val_metrics['acer']:.4f}, "
              f"APCER: {val_metrics['apcer']:.4f}, BPCER: {val_metrics['bpcer']:.4f}")
        print(f"  LR: {self.scheduler_callback.get_lr():.6f}")
        
        if self.logger:
            self.logger.log_metrics(train_metrics, self.global_step, prefix="train")
            self.logger.log_metrics(val_metrics, self.global_step, prefix="val")
    
    def resume(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}")
