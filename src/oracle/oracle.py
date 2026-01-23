import numpy as np
from typing import Tuple


class Oracle:
    """
    Oracle for survival data with probe depth constraint.

    Given a censored instance and probe depth k, the oracle reveals
    information up to k bins beyond the current censoring time.
    """

    def __init__(
        self,
        true_time: np.ndarray,
        true_event: np.ndarray,
        probe_depth: int
    ):
        """
        Initialize oracle with ground truth data.

        Args:
            true_time: True survival times (ground truth)
            true_event: True event indicators (ground truth)
            probe_depth: Number of bins the oracle can reveal beyond censoring time
        """
        self.true_time = true_time
        self.true_event = true_event
        self.probe_depth = probe_depth

    def query(
        self,
        indices: np.ndarray,
        current_time: np.ndarray,
        current_event: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query oracle for selected indices with probe depth constraint.

        Args:
            indices: Indices of samples to query
            current_time: Current observed censoring times
            current_event: Current event indicators

        Returns:
            updated_time: Updated times after oracle query
            updated_event: Updated event indicators after oracle query

        Logic:
        - If censored at bin c with probe depth k:
          - True death at c+3, k=5: Reveal death at c+3 (event=1, time=c+3)
          - True death at c+7, k=5: Remain censored at c+k (event=0, time=c+5)
          - Already censored at c+2, k=5: Remain censored at c+2 (event=0, time=c+2)
        """
        updated_time = current_time.copy()
        updated_event = current_event.copy()

        for idx in indices:
            c = current_time[idx]  # Current censoring time
            true_t = self.true_time[idx]  # True time (ground truth)
            true_e = self.true_event[idx]  # True event (ground truth)

            # Maximum observable time with probe depth
            max_observable_time = c + self.probe_depth

            if true_e == 1:  # True event is death
                if true_t <= max_observable_time:
                    # Death occurs within probe window - reveal it
                    updated_time[idx] = true_t
                    updated_event[idx] = 1
                else:
                    # Death occurs beyond probe window - censored at max_observable_time
                    updated_time[idx] = max_observable_time
                    updated_event[idx] = 0
            else:  # True event is censoring
                # Already censored - just update to min of true censoring and max observable
                updated_time[idx] = min(true_t, max_observable_time)
                updated_event[idx] = 0

        return updated_time, updated_event

    def get_oracle_outcome_probs(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: np.ndarray
    ) -> np.ndarray:
        """
        Get oracle outcome probabilities for BatchBALD.

        For a censored instance at bin c with probe depth k, the oracle can reveal:
        - Death in bin c+1, c+2, ..., c+k (k outcomes)
        - Censored at c+k (1 outcome for T > c+k)

        This computes the probability distribution over these k+1 outcomes.

        Args:
            predictions: Model predictions (K, N, T) - probabilities
            current_time: Current censoring times (N,)
            current_event: Current event indicators (N,)

        Returns:
            oracle_probs: (N, k+1) probabilities over oracle outcomes
                         First k columns: P(death in c+1), ..., P(death in c+k)
                         Last column: P(censored at c+k)
        """
        K, N, T = predictions.shape
        # Average over ensemble
        mean_probs = predictions.mean(axis=0)  # (N, T)

        oracle_probs_list = []

        for i in range(N):
            if current_event[i] == 1:
                # Already uncensored - no oracle query needed
                # Return uniform (this shouldn't be queried anyway)
                oracle_probs_list.append(np.ones(self.probe_depth + 1) / (self.probe_depth + 1))
                continue

            c = current_time[i]  # Current censoring time
            probs = mean_probs[i].copy()  # (T,)

            # Step 1: Renormalize - can't have died before censoring time c
            probs[:c+1] = 0
            probs = probs / (probs.sum() + 1e-8)

            # Step 2: Compute oracle outcome probabilities
            oracle_probs = np.zeros(self.probe_depth + 1)

            for j in range(self.probe_depth):
                bin_idx = c + 1 + j  # Bins c+1, c+2, ..., c+k
                if bin_idx < T:
                    oracle_probs[j] = probs[bin_idx]

            # Last outcome: P(T > c+k) = sum of probs for bins > c+k
            max_observable = c + self.probe_depth
            if max_observable + 1 < T:
                oracle_probs[-1] = probs[max_observable + 1:].sum()

            # Renormalize
            oracle_probs = oracle_probs / (oracle_probs.sum() + 1e-8)

            oracle_probs_list.append(oracle_probs)

        return np.array(oracle_probs_list)  # (N, k+1)

    def get_oracle_outcome_probs_ensemble(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: np.ndarray
    ) -> np.ndarray:
        """
        Get oracle outcome probabilities for each ensemble member.

        Args:
            predictions: Model predictions (K, N, T)
            current_time: Current censoring times (N,)
            current_event: Current event indicators (N,)

        Returns:
            oracle_probs: (K, N, k+1) probabilities over oracle outcomes for each ensemble member
        """
        K, N, T = predictions.shape
        oracle_probs_all = np.zeros((K, N, self.probe_depth + 1))

        for k in range(K):
            # Get oracle probs for this ensemble member
            probs_k = predictions[k:k+1]  # (1, N, T)
            oracle_probs_k = self.get_oracle_outcome_probs(
                probs_k,
                current_time,
                current_event
            )
            oracle_probs_all[k] = oracle_probs_k

        return oracle_probs_all
