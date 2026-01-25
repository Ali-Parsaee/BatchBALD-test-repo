import numpy as np
from typing import Optional
from scipy.special import xlogy

from src.acquisition.batchbald import get_queryable_mask


class EntropyAcquisition:
    """
    Entropy-based acquisition function.

    Selects points with highest predictive entropy (averaged over ensemble).
    """

    def __init__(self):
        pass

    def compute_scores(
        self,
        predictions: np.ndarray,
        current_event: Optional[np.ndarray] = None,
        artificial_time: Optional[np.ndarray] = None,
        true_time: Optional[np.ndarray] = None,
        true_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute entropy scores.

        Args:
            predictions: (K, N, T) probability predictions
            current_event: (N,) event indicators
            artificial_time: (N,) artificial censoring times
            true_time: (N,) true times (before artificial censoring)
            true_event: (N,) true event indicators (before artificial censoring)

        Returns:
            scores: (N,) entropy scores (higher = more uncertain)
        """
        K, N, T = predictions.shape

        # Average predictions over ensemble
        mean_probs = predictions.mean(axis=0)  # (N, T)

        # Compute entropy: H(p) = -sum p log p
        entropy = -np.sum(xlogy(mean_probs, mean_probs), axis=1)  # (N,)

        # Set score to -inf for non-queryable points
        queryable_mask = get_queryable_mask(
            current_event, artificial_time, true_time, true_event
        )
        if queryable_mask is not None:
            entropy[~queryable_mask] = -np.inf

        return entropy

    def select_batch(
        self,
        predictions: np.ndarray,
        batch_size: int,
        current_event: Optional[np.ndarray] = None,
        artificial_time: Optional[np.ndarray] = None,
        true_time: Optional[np.ndarray] = None,
        true_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Select batch with highest entropy.

        Args:
            predictions: (K, N, T) predictions
            batch_size: Number to select
            current_event: (N,) event indicators
            artificial_time: (N,) artificial censoring times
            true_time: (N,) true times (before artificial censoring)
            true_event: (N,) true event indicators (before artificial censoring)

        Returns:
            selected_indices: (batch_size,) selected indices
        """
        scores = self.compute_scores(
            predictions, current_event, artificial_time, true_time, true_event
        )

        # Select top-k
        selected_indices = np.argsort(scores)[::-1][:batch_size]

        # Filter out invalid selections (-inf scores)
        valid_mask = scores[selected_indices] > -np.inf
        selected_indices = selected_indices[valid_mask]

        return selected_indices
