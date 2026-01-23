import numpy as np
from typing import Optional
from scipy.special import xlogy


class WeightedBatchBALD:
    """
    Weighted BatchBALD - weights death outcomes more than censored outcomes.

    Key insight: Not all oracle outcomes are equally informative!
    - Death revealed → very informative (new label)
    - Censoring extended → less informative (partial info)

    We weight the mutual information by expected informativeness.
    """

    def __init__(self, probe_depth: int, death_weight: float = 1.0, censor_weight: float = 0.3):
        """
        Args:
            probe_depth: Oracle probe depth (k bins)
            death_weight: Weight for death outcomes (default: 1.0)
            censor_weight: Weight for censored outcome (default: 0.3)
        """
        self.probe_depth = probe_depth
        self.death_weight = death_weight
        self.censor_weight = censor_weight

        # Create weight vector: [death_weight, death_weight, ..., censor_weight]
        self.outcome_weights = np.ones(probe_depth + 1)
        self.outcome_weights[-1] = censor_weight  # Last outcome is censored

    def compute_weighted_entropy(
        self,
        probs: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute weighted entropy: H_w(Y) = -sum w_i * p_i * log(p_i)

        Args:
            probs: Probability distribution (can be 1D, 2D, or 3D)
            weights: Outcome weights (default: use self.outcome_weights)

        Returns:
            Weighted entropy
        """
        if weights is None:
            weights = self.outcome_weights

        # Broadcast weights to match probs shape
        if probs.ndim == 1:  # (k+1,)
            w = weights
        elif probs.ndim == 2:  # (N, k+1)
            w = weights[np.newaxis, :]
        elif probs.ndim == 3:  # (K, N, k+1)
            w = weights[np.newaxis, np.newaxis, :]
        else:
            raise ValueError(f"Unexpected probs shape: {probs.shape}")

        # Weighted entropy: -sum w * p * log(p)
        weighted_h = -np.sum(w * xlogy(probs, probs), axis=-1)

        return weighted_h

    def compute_conditional_entropy(
        self,
        oracle_probs: np.ndarray
    ) -> np.ndarray:
        """
        Compute weighted H(Y_oracle | theta) for each ensemble member.

        Args:
            oracle_probs: (K, N, k+1) probabilities

        Returns:
            conditional_entropy: (K, N) weighted entropy
        """
        return self.compute_weighted_entropy(oracle_probs)  # (K, N)

    def compute_entropy_of_expected(
        self,
        oracle_probs: np.ndarray
    ) -> np.ndarray:
        """
        Compute weighted H(E[Y_oracle]).

        Args:
            oracle_probs: (K, N, k+1) probabilities

        Returns:
            entropy_expected: (N,) weighted entropy of mean predictions
        """
        mean_probs = oracle_probs.mean(axis=0)  # (N, k+1)
        return self.compute_weighted_entropy(mean_probs)  # (N,)

    def compute_mutual_information(
        self,
        oracle_probs: np.ndarray
    ) -> np.ndarray:
        """
        Compute weighted I(Y_oracle; theta) = H_w(E[Y]) - E[H_w(Y|theta)].

        This emphasizes disagreement about DEATH outcomes over censored outcomes.

        Args:
            oracle_probs: (K, N, k+1) probabilities

        Returns:
            mutual_info: (N,) weighted mutual information
        """
        # Weighted H(E[Y_oracle])
        entropy_expected = self.compute_entropy_of_expected(oracle_probs)  # (N,)

        # E[Weighted H(Y_oracle | theta)]
        conditional_entropy = self.compute_conditional_entropy(oracle_probs)  # (K, N)
        expected_conditional_entropy = conditional_entropy.mean(axis=0)  # (N,)

        # Weighted mutual information
        mutual_info = entropy_expected - expected_conditional_entropy

        return mutual_info

    def select_batch(
        self,
        oracle_probs: np.ndarray,
        batch_size: int,
        current_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Select batch using greedy approximation.

        Args:
            oracle_probs: (K, N, k+1) oracle outcome probabilities
            batch_size: Number of samples to select
            current_event: (N,) event indicators (to filter already uncensored)

        Returns:
            selected_indices: (batch_size,) indices of selected samples
        """
        K, N, num_outcomes = oracle_probs.shape
        selected = []
        remaining = set(range(N))

        # Filter out already uncensored samples
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
                # Compute joint MI if we add this point
                if len(selected) == 0:
                    # First point: just use weighted MI
                    score = self.compute_mutual_information(
                        oracle_probs[:, [idx], :]
                    )[0]
                else:
                    # Joint MI of selected + candidate
                    candidate_batch = selected + [idx]
                    score = self._compute_joint_mutual_info(
                        oracle_probs[:, candidate_batch, :]
                    )

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.discard(best_idx)

        return np.array(selected)

    def _compute_joint_mutual_info(
        self,
        oracle_probs_batch: np.ndarray
    ) -> float:
        """
        Compute joint weighted mutual information for a batch.

        Approximates as sum of individual weighted MIs (assumes independence).

        Args:
            oracle_probs_batch: (K, batch_size, k+1)

        Returns:
            joint_mi: Joint weighted mutual information
        """
        K, batch_size, num_outcomes = oracle_probs_batch.shape

        # Approximate joint entropy (assumes independence)
        entropy_expected = self.compute_entropy_of_expected(
            oracle_probs_batch
        ).sum()

        # E[H({Y_i} | theta)] ≈ E[sum H(Y_i | theta)]
        conditional_entropy = self.compute_conditional_entropy(
            oracle_probs_batch
        )  # (K, batch_size)
        expected_conditional_entropy = conditional_entropy.sum(axis=1).mean()

        joint_mi = entropy_expected - expected_conditional_entropy

        return joint_mi

    def compute_scores(
        self,
        oracle_probs: np.ndarray,
        current_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute acquisition scores (weighted MI) for all instances.

        Args:
            oracle_probs: (K, N, k+1)
            current_event: (N,) event indicators

        Returns:
            scores: (N,) acquisition scores
        """
        scores = self.compute_mutual_information(oracle_probs)

        # Set score to -inf for already uncensored
        if current_event is not None:
            scores[current_event == 1] = -np.inf

        return scores
