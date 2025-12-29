"""Face Anti-Spoofing metrics following ISO/IEC 30107-3 standard.

Implements APCER, BPCER, ACER, HTER, EER, and AUC metrics for evaluating
face anti-spoofing systems.

Label Convention:
    0 = spoof/attack
    1 = real/bonafide
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import roc_curve


def calculate_apcer(
    preds: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> float:
    """Calculate Attack Presentation Classification Error Rate.

    APCER is the proportion of attack presentations incorrectly classified
    as bonafide (false negatives for spoof detection).

    Args:
        preds: Prediction probabilities for being real (0-1).
        labels: Ground truth labels (0=spoof, 1=real).
        threshold: Decision threshold. Predictions >= threshold are classified as real.

    Returns:
        APCER value between 0 and 1.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    # Attack samples are where label == 0
    attack_mask = labels == 0

    if not np.any(attack_mask):
        return 0.0  # No attack samples

    attack_preds = preds[attack_mask]

    # False negatives: attacks classified as real (pred >= threshold)
    false_negatives = np.sum(attack_preds >= threshold)

    return float(false_negatives / len(attack_preds))


def calculate_bpcer(
    preds: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> float:
    """Calculate Bonafide Presentation Classification Error Rate.

    BPCER is the proportion of bonafide presentations incorrectly classified
    as attacks (false positives for spoof detection).

    Args:
        preds: Prediction probabilities for being real (0-1).
        labels: Ground truth labels (0=spoof, 1=real).
        threshold: Decision threshold. Predictions >= threshold are classified as real.

    Returns:
        BPCER value between 0 and 1.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    # Bonafide samples are where label == 1
    bonafide_mask = labels == 1

    if not np.any(bonafide_mask):
        return 0.0  # No bonafide samples

    bonafide_preds = preds[bonafide_mask]

    # False positives: bonafides classified as attacks (pred < threshold)
    false_positives = np.sum(bonafide_preds < threshold)

    return float(false_positives / len(bonafide_preds))


def calculate_acer(
    preds: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> float:
    """Calculate Average Classification Error Rate.

    ACER = (APCER + BPCER) / 2

    Args:
        preds: Prediction probabilities for being real (0-1).
        labels: Ground truth labels (0=spoof, 1=real).
        threshold: Decision threshold.

    Returns:
        ACER value between 0 and 1.
    """
    apcer = calculate_apcer(preds, labels, threshold)
    bpcer = calculate_bpcer(preds, labels, threshold)

    return (apcer + bpcer) / 2


def calculate_hter(
    preds: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> float:
    """Calculate Half Total Error Rate.

    HTER = (FAR + FRR) / 2, which equals ACER when:
    - FAR (False Acceptance Rate) = APCER
    - FRR (False Rejection Rate) = BPCER

    Args:
        preds: Prediction probabilities for being real (0-1).
        labels: Ground truth labels (0=spoof, 1=real).
        threshold: Decision threshold.

    Returns:
        HTER value between 0 and 1.
    """
    # HTER is equivalent to ACER in this context
    return calculate_acer(preds, labels, threshold)


def calculate_eer(preds: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Calculate Equal Error Rate and corresponding threshold.

    EER is the point where False Acceptance Rate equals False Rejection Rate.
    This is where APCER equals BPCER.

    Args:
        preds: Prediction probabilities for being real (0-1).
        labels: Ground truth labels (0=spoof, 1=real).

    Returns:
        Tuple of (EER, threshold) where EER is the equal error rate
        and threshold is the operating point achieving it.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    # Get ROC curve
    # Note: roc_curve expects higher scores for positive class (real=1)
    fpr, tpr, thresholds = roc_curve(labels, preds)

    # FRR (False Rejection Rate) = 1 - TPR = BPCER
    # FAR (False Acceptance Rate) = FPR = APCER
    frr = 1 - tpr

    # Find the point where FAR and FRR are closest
    abs_diffs = np.abs(fpr - frr)
    min_idx = np.argmin(abs_diffs)

    # EER is the average of FAR and FRR at that point
    eer = (fpr[min_idx] + frr[min_idx]) / 2

    # Get corresponding threshold
    threshold = float(thresholds[min_idx])

    return float(eer), threshold


def calculate_auc(preds: np.ndarray, labels: np.ndarray) -> float:
    """Calculate Area Under ROC Curve.

    Args:
        preds: Prediction probabilities for being real (0-1).
        labels: Ground truth labels (0=spoof, 1=real).

    Returns:
        AUC value between 0 and 1.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    # Handle edge cases
    if len(np.unique(labels)) < 2:
        return 0.5  # Cannot compute AUC with single class

    fpr, tpr, _ = roc_curve(labels, preds)
    return float(sklearn_auc(fpr, tpr))


def calculate_tpr_at_fpr(
    preds: np.ndarray, labels: np.ndarray, target_fpr: float = 0.01
) -> Tuple[float, float]:
    """Calculate True Positive Rate at a specific False Positive Rate.

    Useful for evaluating model at specific operating points (e.g., TPR@FPR=1%).

    Args:
        preds: Prediction probabilities for being real (0-1).
        labels: Ground truth labels (0=spoof, 1=real).
        target_fpr: Target false positive rate (default: 0.01 = 1%).

    Returns:
        Tuple of (TPR, threshold) at the target FPR.
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    fpr, tpr, thresholds = roc_curve(labels, preds)

    # Find the index where FPR is closest to target
    idx = np.argmin(np.abs(fpr - target_fpr))

    return float(tpr[idx]), float(thresholds[idx])


class FASMetrics:
    """Accumulator class for Face Anti-Spoofing metrics.

    Collects predictions and labels during evaluation, then computes
    all FAS metrics at once.

    Example:
        >>> metrics = FASMetrics()
        >>> for batch in dataloader:
        ...     preds = model(batch['rgb'])
        ...     metrics.update(preds, batch['label'])
        >>> results = metrics.compute()
        >>> print(f"ACER: {results['acer']:.4f}")
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize metrics accumulator.

        Args:
            threshold: Default decision threshold for ACER/APCER/BPCER.
        """
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        """Reset accumulated predictions and labels."""
        self._preds: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []

    def update(
        self,
        preds: Union[np.ndarray, "torch.Tensor"],
        labels: Union[np.ndarray, "torch.Tensor"],
    ) -> None:
        """Update with new batch of predictions and labels.

        Args:
            preds: Prediction probabilities (can be torch.Tensor or np.ndarray).
            labels: Ground truth labels (can be torch.Tensor or np.ndarray).
        """
        # Convert to numpy if needed
        if hasattr(preds, "detach"):
            preds = preds.detach().cpu().numpy()
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().numpy()

        preds = np.asarray(preds).flatten()
        labels = np.asarray(labels).flatten()

        self._preds.append(preds)
        self._labels.append(labels)

    def compute(
        self, threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """Compute all FAS metrics.

        Args:
            threshold: Decision threshold. If None, uses the default or EER threshold.

        Returns:
            Dictionary containing all computed metrics:
            - apcer: Attack Presentation Classification Error Rate
            - bpcer: Bonafide Presentation Classification Error Rate
            - acer: Average Classification Error Rate
            - hter: Half Total Error Rate
            - eer: Equal Error Rate
            - eer_threshold: Threshold at EER
            - auc: Area Under ROC Curve
            - accuracy: Overall accuracy
        """
        if not self._preds:
            return {
                "apcer": 0.0,
                "bpcer": 0.0,
                "acer": 0.0,
                "hter": 0.0,
                "eer": 0.0,
                "eer_threshold": 0.5,
                "auc": 0.5,
                "accuracy": 0.0,
            }

        # Concatenate all batches
        preds = np.concatenate(self._preds)
        labels = np.concatenate(self._labels)

        # Calculate EER and its threshold
        eer, eer_threshold = calculate_eer(preds, labels)

        # Use provided threshold, or EER threshold, or default
        if threshold is None:
            threshold = self.threshold

        # Calculate all metrics
        apcer = calculate_apcer(preds, labels, threshold)
        bpcer = calculate_bpcer(preds, labels, threshold)
        acer = calculate_acer(preds, labels, threshold)
        hter = calculate_hter(preds, labels, threshold)
        auc_score = calculate_auc(preds, labels)

        # Calculate accuracy
        binary_preds = (preds >= threshold).astype(int)
        accuracy = np.mean(binary_preds == labels)

        return {
            "apcer": apcer,
            "bpcer": bpcer,
            "acer": acer,
            "hter": hter,
            "eer": eer,
            "eer_threshold": eer_threshold,
            "auc": auc_score,
            "accuracy": float(accuracy),
            "threshold": threshold,
            "num_samples": len(preds),
            "num_real": int(np.sum(labels == 1)),
            "num_spoof": int(np.sum(labels == 0)),
        }

    def compute_at_eer(self) -> Dict[str, float]:
        """Compute metrics using EER threshold.

        Returns:
            Dictionary containing metrics computed at EER threshold.
        """
        if not self._preds:
            return self.compute()

        preds = np.concatenate(self._preds)
        labels = np.concatenate(self._labels)

        _, eer_threshold = calculate_eer(preds, labels)

        return self.compute(threshold=eer_threshold)
