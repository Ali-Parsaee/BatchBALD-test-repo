"""
Improved BatchBALD variants based on systematic analysis.

Key insights from analysis:
1. Observable mass (ρ=+0.1209) is the best predictor
2. MI (ρ=+0.0966) is good
3. P(death in window) (ρ=+0.0923) helps
4. Combinations: MI * P(death) works well (ρ=+0.1145)
"""

import numpy as np
from typing import Optional
from scipy.special import xlogy


class ImprovedBatchBALD:
    """
    Improved BatchBALD using insights from systematic analysis.

    Strategy: Combine MI with observable mass and death probability.
    """

    def __init__(self, probe_depth: int, variant: str = 'observable_mass'):
        """
        Args:
            probe_depth: Oracle probe depth (k bins)
            variant: Which improvement to use:
                - 'observable_mass': Use observable mass as primary signal
                - 'mi_times_pdeath': MI * P(death in window)
                - 'mi_times_pdeath_sq': MI * P(death)^2
                - 'combined': MI * observable_mass
                - 'filtered': MI but filter low P(death) samples
        """
        self.probe_depth = probe_depth
        self.variant = variant

    def compute_observable_mass(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: np.ndarray
    ) -> np.ndarray:
        """
        Compute probability mass in observable window.

        For sample censored at c, observable window is [c+1, c+probe_depth].

        Args:
            predictions: (K, N, T) predictions
            current_time: (N,) censoring times
            current_event: (N,) event indicators

        Returns:
            observable_mass: (N,) probability mass in observable window
        """
        K, N, T = predictions.shape
        mean_preds = predictions.mean(axis=0)  # (N, T)

        observable_mass = np.zeros(N)

        for i in range(N):
            if current_event[i] == 1:
                continue  # Already uncensored

            c = current_time[i]
            max_obs = min(c + self.probe_depth, T - 1)

            if c + 1 <= max_obs:
                observable_mass[i] = mean_preds[i, c+1:max_obs+1].sum()

        return observable_mass

    def compute_p_death_window(
        self,
        oracle_probs: np.ndarray
    ) -> np.ndarray:
        """
        Compute P(death in window) from oracle probs.

        Args:
            oracle_probs: (K, N, k+1) oracle outcome probabilities

        Returns:
            p_death: (N,) probability of death in observable window
        """
        # Sum over death outcomes (all but last which is censored)
        mean_oracle_probs = oracle_probs.mean(axis=0)  # (N, k+1)
        p_death = mean_oracle_probs[:, :-1].sum(axis=1)  # (N,)

        return p_death

    def compute_mutual_information(
        self,
        oracle_probs: np.ndarray
    ) -> np.ndarray:
        """
        Standard MI computation.

        Args:
            oracle_probs: (K, N, k+1) oracle outcome probabilities

        Returns:
            mi: (N,) mutual information
        """
        # H(E[Y])
        mean_probs = oracle_probs.mean(axis=0)  # (N, k+1)
        H_expected = -np.sum(xlogy(mean_probs, mean_probs), axis=1)  # (N,)

        # E[H(Y|theta)]
        H_conditional = -np.sum(xlogy(oracle_probs, oracle_probs), axis=2)  # (K, N)
        E_H_conditional = H_conditional.mean(axis=0)  # (N,)

        mi = H_expected - E_H_conditional

        return mi

    def compute_scores(
        self,
        oracle_probs: np.ndarray,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute acquisition scores based on variant.

        Args:
            oracle_probs: (K, N, k+1) oracle outcome probabilities
            predictions: (K, N, T) model predictions
            current_time: (N,) censoring times
            current_event: (N,) event indicators

        Returns:
            scores: (N,) acquisition scores
        """
        N = oracle_probs.shape[1]

        # Compute components
        mi = self.compute_mutual_information(oracle_probs)
        p_death = self.compute_p_death_window(oracle_probs)
        obs_mass = self.compute_observable_mass(predictions, current_time, current_event)

        # Select variant
        if self.variant == 'observable_mass':
            scores = obs_mass

        elif self.variant == 'mi_times_pdeath':
            scores = mi * p_death

        elif self.variant == 'mi_times_pdeath_sq':
            scores = mi * (p_death ** 2)

        elif self.variant == 'combined':
            scores = mi * obs_mass

        elif self.variant == 'filtered':
            # MI but filter out low P(death) samples
            scores = mi.copy()
            scores[p_death < 0.1] = -np.inf

        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        # Filter already uncensored
        if current_event is not None:
            scores[current_event == 1] = -np.inf

        return scores

    def select_batch(
        self,
        oracle_probs: np.ndarray,
        predictions: np.ndarray,
        current_time: np.ndarray,
        batch_size: int,
        current_event: Optional[np.ndarray] = None,
        greedy: bool = True
    ) -> np.ndarray:
        """
        Select batch of samples.

        Args:
            oracle_probs: (K, N, k+1) oracle outcome probabilities
            predictions: (K, N, T) model predictions
            current_time: (N,) censoring times
            batch_size: Number of samples to select
            current_event: (N,) event indicators
            greedy: If True, use greedy selection (for batch diversity)
                   If False, just select top-k by score

        Returns:
            selected_indices: (batch_size,) selected indices
        """
        if not greedy:
            # Simple top-k selection
            scores = self.compute_scores(
                oracle_probs, predictions, current_time, current_event
            )
            selected = np.argsort(scores)[::-1][:batch_size]
            valid = scores[selected] > -np.inf
            return selected[valid]

        else:
            # Greedy batch selection (for batch diversity)
            N = oracle_probs.shape[1]
            selected = []
            remaining = set(range(N))

            # Filter out already uncensored
            if current_event is not None:
                for i in range(N):
                    if current_event[i] == 1:
                        remaining.discard(i)

            if len(remaining) == 0:
                return np.array([])

            for _ in range(batch_size):
                if len(remaining) == 0:
                    break

                best_score = -np.inf
                best_idx = None

                for idx in remaining:
                    if len(selected) == 0:
                        # First point: use score
                        score = self.compute_scores(
                            oracle_probs[:, [idx], :],
                            predictions[:, [idx], :],
                            current_time[[idx]],
                            current_event[[idx]] if current_event is not None else None
                        )[0]
                    else:
                        # Subsequent points: approximate joint score
                        # Simple approximation: sum of individual scores
                        candidate_batch = selected + [idx]
                        scores = self.compute_scores(
                            oracle_probs[:, candidate_batch, :],
                            predictions[:, candidate_batch, :],
                            current_time[candidate_batch],
                            current_event[candidate_batch] if current_event is not None else None
                        )
                        score = scores.sum()

                    if score > best_score:
                        best_score = score
                        best_idx = idx

                if best_idx is not None:
                    selected.append(best_idx)
                    remaining.discard(best_idx)

            return np.array(selected)


class IterativeBatchBALD:
    """
    Iterative batch selection: Instead of selecting batch_size at once,
    select smaller batches multiple times.

    Insight: BatchBALD works better for small batches (7/10 in top-10)
    but degrades for larger batches (25/50 in batch of 50).

    Solution: Select 5 batches of 10 instead of 1 batch of 50.
    """

    def __init__(self, base_acq_func, sub_batch_size: int = 10):
        """
        Args:
            base_acq_func: Base acquisition function to use
            sub_batch_size: Size of each sub-batch
        """
        self.base_acq_func = base_acq_func
        self.sub_batch_size = sub_batch_size

    def select_batch(
        self,
        oracle_probs: np.ndarray,
        predictions: np.ndarray,
        current_time: np.ndarray,
        batch_size: int,
        current_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Select batch iteratively.

        Args:
            oracle_probs: (K, N, k+1) oracle outcome probabilities
            predictions: (K, N, T) model predictions
            current_time: (N,) censoring times
            batch_size: Total number of samples to select
            current_event: (N,) event indicators

        Returns:
            selected_indices: (batch_size,) selected indices
        """
        N = oracle_probs.shape[1]
        all_selected = []
        current_event_copy = current_event.copy() if current_event is not None else np.zeros(N)

        n_iterations = (batch_size + self.sub_batch_size - 1) // self.sub_batch_size

        for iteration in range(n_iterations):
            # How many to select this round
            remaining_needed = batch_size - len(all_selected)
            this_batch_size = min(self.sub_batch_size, remaining_needed)

            if this_batch_size == 0:
                break

            # Select sub-batch - handle different acquisition function signatures
            if isinstance(self.base_acq_func, ImprovedBatchBALD):
                # ImprovedBatchBALD needs predictions and current_time
                selected = self.base_acq_func.select_batch(
                    oracle_probs,
                    predictions,
                    current_time,
                    this_batch_size,
                    current_event_copy,
                    greedy=True
                )
            else:
                # Original BatchBALD only needs oracle_probs
                selected = self.base_acq_func.select_batch(
                    oracle_probs,
                    this_batch_size,
                    current_event_copy
                )

            # Mark as selected (so they won't be selected again)
            for idx in selected:
                all_selected.append(idx)
                current_event_copy[idx] = 1  # Mark as "queried"

        return np.array(all_selected)
