import torch
from pathlib import Path
from typing import Dict, Any, Optional
import json


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = "min",
    ):
        """
        Args:
            patience: epochs to wait before stopping
            min_delta: minimum change to qualify as improvement
            mode: "min" (loss) or "max" (accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def __call__(self, metric: float) -> bool:
        """Returns True if should stop."""
        if self.best_score is None:
            self.best_score = metric
            return False
        
        if self.mode == "min":
            improved = metric < self.best_score - self.min_delta
        else:
            improved = metric > self.best_score + self.min_delta
        
        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class ModelCheckpoint:
    """Save model checkpoints."""
    
    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_acer",
        mode: str = "min",
        save_best: bool = True,
        save_last: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best = save_best
        self.save_last = save_last
        self.best_score = None
        
    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        scheduler: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Save checkpoint if improved.
        
        Returns path if saved best, None otherwise.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        if scheduler:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        if self.save_last:
            torch.save(checkpoint, self.save_dir / "last.pth")
        
        if self.save_best:
            current_score = metrics.get(self.monitor)
            if current_score is None:
                return None
            
            if self.best_score is None:
                self.best_score = current_score
                best_path = self.save_dir / "best.pth"
                torch.save(checkpoint, best_path)
                return str(best_path)
            else:
                if self.mode == "min":
                    improved = current_score < self.best_score
                else:
                    improved = current_score > self.best_score
                
                if improved:
                    self.best_score = current_score
                    best_path = self.save_dir / "best.pth"
                    torch.save(checkpoint, best_path)
                    return str(best_path)
        
        return None


class LRSchedulerCallback:
    """Wrapper for learning rate schedulers."""
    
    def __init__(self, scheduler, step_on: str = "epoch"):
        """
        Args:
            scheduler: torch LR scheduler
            step_on: "epoch" or "batch"
        """
        self.scheduler = scheduler
        self.step_on = step_on
        
    def step_epoch(self, metric: Optional[float] = None):
        """Step at end of epoch."""
        if self.step_on == "epoch":
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if metric is not None:
                    self.scheduler.step(metric)
            else:
                self.scheduler.step()
                
    def step_batch(self):
        """Step at end of batch."""
        if self.step_on == "batch":
            self.scheduler.step()
            
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.scheduler.get_last_lr()[0]
