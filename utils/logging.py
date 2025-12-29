"""Logging utilities for experiment tracking.

Provides console + file logging and combined experiment logger
for TensorBoard and W&B integration.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Global logger registry
_LOGGERS: Dict[str, logging.Logger] = {}


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Setup a logger with console and optional file output.

    Creates a logger that outputs to console and optionally to a timestamped
    log file in the specified directory.

    Args:
        name: Logger name (used for retrieval and log file naming).
        log_dir: Directory for log files. If None, file logging is disabled.
        level: Logging level (default: INFO).
        console: Whether to output to console (default: True).

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logger('train', log_dir='logs')
        >>> logger.info('Training started')
    """
    # Check if logger already exists
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Formatter
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = log_path / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    # Register logger
    _LOGGERS[name] = logger

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get an existing logger by name.

    Args:
        name: Logger name. If None, returns the root logger.

    Returns:
        Logger instance.

    Raises:
        KeyError: If named logger doesn't exist and name is not None.
    """
    if name is None:
        return logging.getLogger()

    if name in _LOGGERS:
        return _LOGGERS[name]

    # Create a basic logger if it doesn't exist
    return setup_logger(name)


class ExperimentLogger:
    """Combined experiment logger for TensorBoard, W&B, and console.

    Provides a unified interface for logging metrics, images, and text
    to multiple backends simultaneously.

    Example:
        >>> logger = ExperimentLogger(
        ...     name='baseline_rgb',
        ...     log_dir='logs',
        ...     use_tensorboard=True,
        ...     use_wandb=False
        ... )
        >>> logger.log_metrics({'loss': 0.5, 'accuracy': 0.9}, step=100)
        >>> logger.log_text('epoch_summary', 'Epoch 1 completed')
        >>> logger.close()
    """

    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize experiment logger.

        Args:
            name: Experiment name.
            log_dir: Directory for logs.
            use_tensorboard: Enable TensorBoard logging.
            use_wandb: Enable Weights & Biases logging.
            wandb_project: W&B project name (required if use_wandb=True).
            wandb_config: Configuration dict to log to W&B.
        """
        self.name = name
        self.log_dir = Path(log_dir) / name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Console logger
        self.console_logger = setup_logger(
            name=f"exp_{name}",
            log_dir=str(self.log_dir),
        )

        # TensorBoard
        self.use_tensorboard = use_tensorboard
        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_writer = SummaryWriter(log_dir=str(self.log_dir))
                self.console_logger.info(f"TensorBoard logging to {self.log_dir}")
            except ImportError:
                self.console_logger.warning(
                    "TensorBoard not available. Install with: pip install tensorboard"
                )
                self.use_tensorboard = False

        # Weights & Biases
        self.use_wandb = use_wandb
        self._wandb_run = None
        if use_wandb:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=wandb_project or "face-antispoofing",
                    name=name,
                    config=wandb_config,
                    dir=str(self.log_dir),
                )
                self.console_logger.info(f"W&B run: {wandb.run.url}")
            except ImportError:
                self.console_logger.warning(
                    "W&B not available. Install with: pip install wandb"
                )
                self.use_wandb = False
            except Exception as e:
                self.console_logger.warning(f"Failed to initialize W&B: {e}")
                self.use_wandb = False

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log scalar metrics to all backends.

        Args:
            metrics: Dictionary of metric names and values.
            step: Global step / iteration number.
            prefix: Optional prefix for metric names (e.g., 'train/', 'val/').
        """
        # Add prefix if provided
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"

        prefixed_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        # Console log (summarized)
        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in prefixed_metrics.items())
        self.console_logger.info(f"Step {step} | {metric_str}")

        # TensorBoard
        if self._tb_writer:
            for name, value in prefixed_metrics.items():
                self._tb_writer.add_scalar(name, value, step)

        # W&B
        if self._wandb_run:
            import wandb

            wandb.log(prefixed_metrics, step=step)

    def log_image(
        self,
        tag: str,
        image: "torch.Tensor",
        step: int,
    ) -> None:
        """Log image to TensorBoard and W&B.

        Args:
            tag: Image tag/name.
            image: Image tensor (C, H, W) or (H, W, C).
            step: Global step.
        """
        if self._tb_writer:
            self._tb_writer.add_image(tag, image, step)

        if self._wandb_run:
            import wandb

            # Convert tensor to numpy if needed
            if hasattr(image, "numpy"):
                image = image.numpy()
            wandb.log({tag: wandb.Image(image)}, step=step)

    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """Log text to console and backends.

        Args:
            tag: Text tag/name.
            text: Text content.
            step: Optional global step.
        """
        self.console_logger.info(f"{tag}: {text}")

        if self._tb_writer and step is not None:
            self._tb_writer.add_text(tag, text, step)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters.

        Args:
            params: Dictionary of hyperparameters.
        """
        self.console_logger.info(f"Hyperparameters: {params}")

        if self._tb_writer:
            self._tb_writer.add_hparams(params, {})

        if self._wandb_run:
            import wandb

            wandb.config.update(params)

    def info(self, message: str) -> None:
        """Log info message to console.

        Args:
            message: Message to log.
        """
        self.console_logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message to console.

        Args:
            message: Warning message.
        """
        self.console_logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message to console.

        Args:
            message: Error message.
        """
        self.console_logger.error(message)

    def close(self) -> None:
        """Close all logging backends."""
        if self._tb_writer:
            self._tb_writer.close()

        if self._wandb_run:
            import wandb

            wandb.finish()

        self.console_logger.info("Experiment logger closed")

    def __enter__(self) -> "ExperimentLogger":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
