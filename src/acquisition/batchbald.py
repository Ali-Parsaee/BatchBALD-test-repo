import numpy as np
from typing import Optional
from scipy.special import xlogy


class SurvivalBatchBALD:
    """
    BatchBALD acquisition function for survival analysis with probe depth.

    Computes I(Y_oracle; theta | X, D) where Y_oracle is what the oracle reveals.
    """

    def __init__(self, probe_depth: int):
        """
        Args:
            probe_depth: Oracle probe depth (k bins)
        """
        self.probe_depth = probe_depth

    def compute_conditional_entropy(
        self,
        oracle_probs: np.ndarray
    ) -> np.ndarray:
        """
        Compute H(Y_oracle | theta) for each ensemble member.

        H(Y_oracle | theta_k) = -sum_c p(c|theta_k) log p(c|theta_k)

        Args:
            oracle_probs: (K, N, k+1) probabilities over oracle outcomes

        Returns:
            conditional_entropy: (K, N) entropy for each instance and ensemble member
        """
        K, N, _ = oracle_probs.shape

        # Compute entropy: -sum p log p
        # Use xlogy to handle 0 * log(0) = 0
        entropy = -np.sum(xlogy(oracle_probs, oracle_probs), axis=2)  # (K, N)

        return entropy

    def compute_entropy_of_expected(
        self,
        oracle_probs: np.ndarray
    ) -> np.ndarray:
        """
        Compute H(E[Y_oracle]) = H(mean over theta).

        Args:
            oracle_probs: (K, N, k+1) probabilities

        Returns:
            entropy_expected: (N,) entropy of the mean predictions
        """
        # Average over ensemble
        mean_probs = oracle_probs.mean(axis=0)  # (N, k+1)

        # Compute entropy
        entropy = -np.sum(xlogy(mean_probs, mean_probs), axis=1)  # (N,)

        return entropy

    def compute_mutual_information(
        self,
        oracle_probs: np.ndarray
    ) -> np.ndarray:
        """
        Compute I(Y_oracle; theta) = H(E[Y_oracle]) - E[H(Y_oracle | theta)].

        This is the information gained about model parameters from oracle query.

        Args:
            oracle_probs: (K, N, k+1) probabilities

        Returns:
            mutual_info: (N,) mutual information for each instance
        """
        # H(E[Y_oracle])
        entropy_expected = self.compute_entropy_of_expected(oracle_probs)  # (N,)

        # E[H(Y_oracle | theta)]
        conditional_entropy = self.compute_conditional_entropy(oracle_probs)  # (K, N)
        expected_conditional_entropy = conditional_entropy.mean(axis=0)  # (N,)

        # Mutual information
        mutual_info = entropy_expected - expected_conditional_entropy

        return mutual_info

    def select_batch(
        self,
        oracle_probs: np.ndarray,
        batch_size: int,
        current_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Select batch using greedy BatchBALD approximation.

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
                # Compute joint entropy if we add this point
                if len(selected) == 0:
                    # First point: just use mutual information
                    score = self.compute_mutual_information(
                        oracle_probs[:, [idx], :]
                    )[0]
                else:
                    # Joint entropy of selected + candidate
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
        Compute joint mutual information for a batch.

        I({Y_i}; theta) = H(E[{Y_i}]) - E[H({Y_i} | theta)]

        For computational efficiency, we approximate the joint distribution
        assuming conditional independence given theta:
        H({Y_i} | theta) ≈ sum H(Y_i | theta)

        Args:
            oracle_probs_batch: (K, batch_size, k+1)

        Returns:
            joint_mi: Joint mutual information
        """
        K, batch_size, num_outcomes = oracle_probs_batch.shape

        # Approximate joint entropy (assumes independence)
        # H(E[{Y_i}]) ≈ sum H(E[Y_i])
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
        Compute acquisition scores (mutual information) for all instances.

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
