"""
New C-BALD (Diversity-Enhanced) acquisition functions for survival analysis.

These variants combine C-BALD scoring with diversity-based batch selection.
Based on the formula: cbald_score = time_variance * (0.5 + 0.5 * death_prob)

Variants:
- CBaldDiverse: Hybrid C-BALD + diversity filtering (winner at budget=20)
- CBaldDiverseAdaptive: Adaptive ratio - exploit early, explore later (winner at budget=10)
- CBaldDiverseNoFilter: No pre-filtering - diversity from entire pool
- CBaldTwoStage: 2-stage selection (pure C-BALD first, then diversity)
"""

import numpy as np
from typing import Optional
from scipy.special import xlogy


def _compute_js_divergence(p1: np.ndarray, p2: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute Jensen-Shannon divergence between two probability distributions.

    Args:
        p1: First probability distribution
        p2: Second probability distribution
        eps: Small constant for numerical stability

    Returns:
        JS divergence value
    """
    p1 = np.clip(p1, eps, 1.0)
    p2 = np.clip(p2, eps, 1.0)

    # Normalize
    p1 = p1 / (p1.sum() + eps)
    p2 = p2 / (p2.sum() + eps)

    # JS divergence
    m = 0.5 * (p1 + p2)
    kl1 = np.sum(xlogy(p1, p1) - xlogy(p1, m))
    kl2 = np.sum(xlogy(p2, p2) - xlogy(p2, m))
    js = 0.5 * (kl1 + kl2)

    return js


class CBaldDiverse:
    """
    New C-BALD with diversity filtering.

    Strategy:
    1. Compute C-BALD scores (time variance weighted by death probability)
    2. Select top candidates by C-BALD score
    3. Greedily pick diverse subset using JS divergence

    Winner at budget=20.
    """

    def __init__(self, probe_depth: int, diversity_ratio: float = 0.3):
        """
        Args:
            probe_depth: Oracle probe depth
            diversity_ratio: Balance between C-BALD score and diversity (0-1)
                           0 = pure C-BALD, 1 = pure diversity
                           Default 0.3 balances both
        """
        self.probe_depth = probe_depth
        self.diversity_ratio = diversity_ratio

    def compute_cbald_scores(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: np.ndarray
    ) -> np.ndarray:
        """
        Compute base C-BALD scores.

        Formula: cbald_score = time_variance * (0.5 + 0.5 * death_prob)

        Args:
            predictions: (K, N, T) ensemble predictions
            current_time: (N,) current censoring times
            current_event: (N,) event indicators

        Returns:
            scores: (N,) C-BALD scores
        """
        K, N, T = predictions.shape

        # Compute expected survival time for each ensemble member
        time_bins = np.arange(T)
        expected_times = np.sum(predictions * time_bins, axis=2)  # (K, N)

        # Time variance (epistemic uncertainty)
        time_variance = np.var(expected_times, axis=0)  # (N,)

        # Death probability in observable window [c+1, c+probe_depth]
        avg_probs = predictions.mean(axis=0)  # (N, T)
        death_prob = np.zeros(N)

        for i in range(N):
            if current_event[i] == 1:
                # Already uncensored
                death_prob[i] = 1.0
            else:
                c = int(current_time[i])
                start = min(c + 1, T)
                end = min(c + self.probe_depth + 1, T)
                if start < end:
                    death_prob[i] = avg_probs[i, start:end].sum()

        # Combined C-BALD score
        cbald_scores = time_variance * (0.5 + 0.5 * death_prob)

        # Set to 0 for already uncensored
        cbald_scores[current_event == 1] = 0.0

        return cbald_scores

    def select_batch(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        batch_size: int,
        current_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Select batch using C-BALD + diversity.

        Args:
            predictions: (K, N, T) ensemble predictions
            current_time: (N,) current censoring times
            batch_size: Number of samples to select
            current_event: (N,) event indicators

        Returns:
            selected_indices: Selected sample indices
        """
        K, N, T = predictions.shape

        # Step 1: Compute C-BALD scores
        cbald_scores = self.compute_cbald_scores(predictions, current_time, current_event)

        # Step 2: Get top candidates (3x budget for diversity selection)
        top_k = min(batch_size * 3, N)
        top_indices = np.argsort(cbald_scores)[-top_k:][::-1]

        # If no diversity needed or pool too small, just return top samples
        if self.diversity_ratio == 0 or len(top_indices) <= batch_size:
            return top_indices[:batch_size]

        # Step 3: Get mean predictions for diversity computation
        avg_probs = predictions.mean(axis=0)[top_indices]  # (top_k, T)

        # Normalize predictions
        avg_probs = avg_probs / (avg_probs.sum(axis=1, keepdims=True) + 1e-12)

        # Step 4: Greedy diversity selection
        selected_local = []  # Indices in top_indices array
        selected_probs = []
        remaining = list(range(len(top_indices)))

        # Select first sample (highest C-BALD score)
        selected_local.append(0)
        remaining.remove(0)
        selected_probs.append(avg_probs[0])

        # Greedily select remaining samples
        while len(selected_local) < batch_size and len(remaining) > 0:
            best_idx = None
            best_combined_score = -np.inf

            for i in remaining:
                # C-BALD score component
                local_cbald_score = cbald_scores[top_indices[i]]

                # Diversity component: average JS divergence to selected
                js_divs = [_compute_js_divergence(avg_probs[i], sp) for sp in selected_probs]
                diversity_score = np.mean(js_divs)

                # Normalize and combine
                max_cbald = cbald_scores[top_indices].max()
                normalized_cbald = local_cbald_score / (max_cbald + 1e-12)
                normalized_diversity = diversity_score

                combined = ((1 - self.diversity_ratio) * normalized_cbald +
                           self.diversity_ratio * normalized_diversity)

                if combined > best_combined_score:
                    best_combined_score = combined
                    best_idx = i

            if best_idx is not None:
                selected_local.append(best_idx)
                remaining.remove(best_idx)
                selected_probs.append(avg_probs[best_idx])

        # Map back to original indices
        final_indices = top_indices[selected_local]

        return final_indices


class CBaldDiverseAdaptive:
    """
    New C-BALD with adaptive diversity ratio.

    Strategy: Exploit early (high C-BALD), explore later (high diversity)
    - Early selections: 90% C-BALD + 10% diversity
    - Later selections: 50% C-BALD + 50% diversity

    Winner at budget=10.
    """

    def __init__(self, probe_depth: int, start_ratio: float = 0.1, end_ratio: float = 0.5):
        """
        Args:
            probe_depth: Oracle probe depth
            start_ratio: Initial diversity ratio (default 0.1)
            end_ratio: Final diversity ratio (default 0.5)
        """
        self.probe_depth = probe_depth
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio

    def compute_cbald_scores(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: np.ndarray
    ) -> np.ndarray:
        """Compute base C-BALD scores (same as CBaldDiverse)."""
        K, N, T = predictions.shape

        time_bins = np.arange(T)
        expected_times = np.sum(predictions * time_bins, axis=2)
        time_variance = np.var(expected_times, axis=0)

        avg_probs = predictions.mean(axis=0)
        death_prob = np.zeros(N)

        for i in range(N):
            if current_event[i] == 1:
                death_prob[i] = 1.0
            else:
                c = int(current_time[i])
                start = min(c + 1, T)
                end = min(c + self.probe_depth + 1, T)
                if start < end:
                    death_prob[i] = avg_probs[i, start:end].sum()

        cbald_scores = time_variance * (0.5 + 0.5 * death_prob)
        cbald_scores[current_event == 1] = 0.0

        return cbald_scores

    def select_batch(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        batch_size: int,
        current_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Select batch with adaptive diversity ratio."""
        K, N, T = predictions.shape

        cbald_scores = self.compute_cbald_scores(predictions, current_time, current_event)

        top_k = min(batch_size * 3, N)
        top_indices = np.argsort(cbald_scores)[-top_k:][::-1]

        if len(top_indices) <= batch_size:
            return top_indices[:batch_size]

        avg_probs = predictions.mean(axis=0)[top_indices]
        avg_probs = avg_probs / (avg_probs.sum(axis=1, keepdims=True) + 1e-12)

        selected_local = []
        selected_probs = []
        remaining = list(range(len(top_indices)))

        # First sample
        selected_local.append(0)
        remaining.remove(0)
        selected_probs.append(avg_probs[0])

        # Adaptive selection
        while len(selected_local) < batch_size and len(remaining) > 0:
            # Compute adaptive diversity ratio
            progress = len(selected_local) / batch_size
            current_ratio = self.start_ratio + (self.end_ratio - self.start_ratio) * progress

            best_idx = None
            best_combined_score = -np.inf

            for i in remaining:
                local_cbald_score = cbald_scores[top_indices[i]]
                js_divs = [_compute_js_divergence(avg_probs[i], sp) for sp in selected_probs]
                diversity_score = np.mean(js_divs)

                max_cbald = cbald_scores[top_indices].max()
                normalized_cbald = local_cbald_score / (max_cbald + 1e-12)
                normalized_diversity = diversity_score

                combined = ((1 - current_ratio) * normalized_cbald +
                           current_ratio * normalized_diversity)

                if combined > best_combined_score:
                    best_combined_score = combined
                    best_idx = i

            if best_idx is not None:
                selected_local.append(best_idx)
                remaining.remove(best_idx)
                selected_probs.append(avg_probs[best_idx])

        return top_indices[selected_local]


class CBaldDiverseNoFilter:
    """
    New C-BALD with diversity but no pre-filtering.

    Strategy: Select diverse samples from entire pool (not just top C-BALD).
    More exploration, less C-BALD exploitation.
    """

    def __init__(self, probe_depth: int, diversity_ratio: float = 0.5):
        """
        Args:
            probe_depth: Oracle probe depth
            diversity_ratio: Balance between C-BALD and diversity
        """
        self.probe_depth = probe_depth
        self.diversity_ratio = diversity_ratio

    def compute_cbald_scores(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: np.ndarray
    ) -> np.ndarray:
        """Compute base C-BALD scores."""
        K, N, T = predictions.shape

        time_bins = np.arange(T)
        expected_times = np.sum(predictions * time_bins, axis=2)
        time_variance = np.var(expected_times, axis=0)

        avg_probs = predictions.mean(axis=0)
        death_prob = np.zeros(N)

        for i in range(N):
            if current_event[i] == 1:
                death_prob[i] = 1.0
            else:
                c = int(current_time[i])
                start = min(c + 1, T)
                end = min(c + self.probe_depth + 1, T)
                if start < end:
                    death_prob[i] = avg_probs[i, start:end].sum()

        cbald_scores = time_variance * (0.5 + 0.5 * death_prob)
        cbald_scores[current_event == 1] = 0.0

        return cbald_scores

    def select_batch(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        batch_size: int,
        current_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Select batch from entire pool without pre-filtering."""
        K, N, T = predictions.shape

        cbald_scores = self.compute_cbald_scores(predictions, current_time, current_event)

        # No pre-filtering - use entire pool
        avg_probs = predictions.mean(axis=0)
        avg_probs = avg_probs / (avg_probs.sum(axis=1, keepdims=True) + 1e-12)

        # Filter out already uncensored
        valid_mask = (current_event == 0) if current_event is not None else np.ones(N, dtype=bool)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return np.array([])

        if len(valid_indices) <= batch_size:
            return valid_indices

        selected = []
        selected_probs = []
        remaining = list(valid_indices)

        # First sample: highest C-BALD from valid samples
        valid_scores = cbald_scores[valid_indices]
        first_local = np.argmax(valid_scores)
        first_idx = valid_indices[first_local]

        selected.append(first_idx)
        remaining.remove(first_idx)
        selected_probs.append(avg_probs[first_idx])

        # Greedy selection with diversity
        while len(selected) < batch_size and len(remaining) > 0:
            best_idx = None
            best_combined_score = -np.inf

            for i in remaining:
                local_cbald_score = cbald_scores[i]
                js_divs = [_compute_js_divergence(avg_probs[i], sp) for sp in selected_probs]
                diversity_score = np.mean(js_divs)

                max_cbald = cbald_scores[valid_indices].max()
                normalized_cbald = local_cbald_score / (max_cbald + 1e-12)
                normalized_diversity = diversity_score

                combined = ((1 - self.diversity_ratio) * normalized_cbald +
                           self.diversity_ratio * normalized_diversity)

                if combined > best_combined_score:
                    best_combined_score = combined
                    best_idx = i

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
                selected_probs.append(avg_probs[best_idx])

        return np.array(selected)


class CBaldTwoStage:
    """
    New C-BALD with 2-stage selection.

    Strategy:
    - Stage 1: Pure C-BALD for first 50%
    - Stage 2: Diversity-focused for second 50%
    """

    def __init__(self, probe_depth: int, stage1_ratio: float = 0.5):
        """
        Args:
            probe_depth: Oracle probe depth
            stage1_ratio: Fraction of batch for stage 1 (default 0.5)
        """
        self.probe_depth = probe_depth
        self.stage1_ratio = stage1_ratio

    def compute_cbald_scores(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: np.ndarray
    ) -> np.ndarray:
        """Compute base C-BALD scores."""
        K, N, T = predictions.shape

        time_bins = np.arange(T)
        expected_times = np.sum(predictions * time_bins, axis=2)
        time_variance = np.var(expected_times, axis=0)

        avg_probs = predictions.mean(axis=0)
        death_prob = np.zeros(N)

        for i in range(N):
            if current_event[i] == 1:
                death_prob[i] = 1.0
            else:
                c = int(current_time[i])
                start = min(c + 1, T)
                end = min(c + self.probe_depth + 1, T)
                if start < end:
                    death_prob[i] = avg_probs[i, start:end].sum()

        cbald_scores = time_variance * (0.5 + 0.5 * death_prob)
        cbald_scores[current_event == 1] = 0.0

        return cbald_scores

    def select_batch(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        batch_size: int,
        current_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Select batch using 2-stage strategy."""
        K, N, T = predictions.shape

        cbald_scores = self.compute_cbald_scores(predictions, current_time, current_event)

        # Stage 1: Pure C-BALD selection
        stage1_size = int(batch_size * self.stage1_ratio)
        stage2_size = batch_size - stage1_size

        top_indices = np.argsort(cbald_scores)[::-1]
        stage1_selected = top_indices[:stage1_size].tolist()

        if stage2_size == 0:
            return np.array(stage1_selected)

        # Stage 2: Diversity-focused selection from remaining
        avg_probs = predictions.mean(axis=0)
        avg_probs = avg_probs / (avg_probs.sum(axis=1, keepdims=True) + 1e-12)

        # Get probabilities of stage 1 selections
        stage1_probs = [avg_probs[i] for i in stage1_selected]

        # Remaining candidates
        remaining = [i for i in range(N) if i not in stage1_selected]
        if current_event is not None:
            remaining = [i for i in remaining if current_event[i] == 0]

        if len(remaining) == 0:
            return np.array(stage1_selected)

        # Greedy diversity selection for stage 2
        stage2_selected = []

        while len(stage2_selected) < stage2_size and len(remaining) > 0:
            best_idx = None
            best_diversity = -np.inf

            for i in remaining:
                # Pure diversity: maximize JS divergence to stage 1
                js_divs = [_compute_js_divergence(avg_probs[i], sp) for sp in stage1_probs]
                avg_diversity = np.mean(js_divs)

                if avg_diversity > best_diversity:
                    best_diversity = avg_diversity
                    best_idx = i

            if best_idx is not None:
                stage2_selected.append(best_idx)
                remaining.remove(best_idx)
                stage1_probs.append(avg_probs[best_idx])

        return np.array(stage1_selected + stage2_selected)


class OldCBALD:
    """
    Old C-BALD: Simple top-k selection based on C-BALD scores.

    No diversity, no BatchBALD - just selects samples with highest scores.
    Formula: cbald_score = time_variance * (0.5 + 0.5 * death_prob)
    """

    def __init__(self, probe_depth: int):
        """
        Args:
            probe_depth: Oracle probe depth
        """
        self.probe_depth = probe_depth

    def compute_cbald_scores(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: np.ndarray
    ) -> np.ndarray:
        """Compute base C-BALD scores."""
        K, N, T = predictions.shape

        time_bins = np.arange(T)
        expected_times = np.sum(predictions * time_bins, axis=2)
        time_variance = np.var(expected_times, axis=0)

        avg_probs = predictions.mean(axis=0)
        death_prob = np.zeros(N)

        for i in range(N):
            if current_event[i] == 1:
                death_prob[i] = 1.0
            else:
                c = int(current_time[i])
                start = min(c + 1, T)
                end = min(c + self.probe_depth + 1, T)
                if start < end:
                    death_prob[i] = avg_probs[i, start:end].sum()

        cbald_scores = time_variance * (0.5 + 0.5 * death_prob)
        cbald_scores[current_event == 1] = 0.0

        return cbald_scores

    def select_batch(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        batch_size: int,
        current_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Select top-k samples by C-BALD score."""
        scores = self.compute_cbald_scores(predictions, current_time, current_event)

        # Simple top-k selection
        top_indices = np.argsort(scores)[::-1][:batch_size]

        return top_indices
