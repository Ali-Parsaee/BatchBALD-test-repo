import numpy as np
from typing import Optional

from src.acquisition.batchbald import get_queryable_mask


class VarianceAcquisition:
    """
    Variance-based acquisition function.

    Selects points with highest predictive variance across ensemble.
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
        Compute variance scores.

        Args:
            predictions: (K, N, T) probability predictions
            current_event: (N,) event indicators
            artificial_time: (N,) artificial censoring times
            true_time: (N,) true times (before artificial censoring)
            true_event: (N,) true event indicators (before artificial censoring)

        Returns:
            scores: (N,) variance scores (higher = more uncertain)
        """
        K, N, T = predictions.shape

        # Compute variance across ensemble for each bin
        variance = predictions.var(axis=0)  # (N, T)

        # Aggregate variance: use mean or max variance across bins
        # Here we use mean variance
        total_variance = variance.mean(axis=1)  # (N,)

        # Set score to -inf for non-queryable points
        queryable_mask = get_queryable_mask(
            current_event, artificial_time, true_time, true_event
        )
        if queryable_mask is not None:
            total_variance[~queryable_mask] = -np.inf

        return total_variance

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
        Select batch with highest variance.

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
