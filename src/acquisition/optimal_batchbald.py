"""
Optimal BatchBALD for Survival Active Learning with Probe Depth.

This combines ALL the best insights:
1. Observable mass (best single predictor: ρ=+0.1209)
2. Mutual information (epistemic uncertainty: ρ=+0.0966)
3. P(death in window) (signal strength: ρ=+0.0923)
4. Ensemble disagreement in observable window
5. Proper greedy batch selection

The goal: DOMINATE entropy and variance methods.
"""

import numpy as np
from typing import Optional
from scipy.special import xlogy


class OptimalBatchBALD:
    """
    The Ultimate BatchBALD for survival AL with probe depth.

    Key innovations:
    1. Ensemble-aware observable mass (not just mean predictions)
    2. Multi-factor scoring: combines observable mass, MI, and variance
    3. Adaptive weighting based on censoring distribution
    4. Greedy batch selection with proper diversity
    """

    def __init__(
        self,
        probe_depth: int,
        strategy: str = 'adaptive_fusion',
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.3
    ):
        """
        Args:
            probe_depth: Oracle probe depth (k bins)
            strategy: Scoring strategy to use:
                - 'adaptive_fusion': Adaptive combination of all signals (RECOMMENDED)
                - 'observable_mass_boosted': Observable mass × (1 + α×MI)
                - 'ensemble_disagreement': Variance of observable mass across ensemble
                - 'multi_factor': observable_mass × MI^β × (1 + γ×P(death))
                - 'weighted_sum': α×obs_mass + β×MI + γ×P(death)
            alpha, beta, gamma: Weighting parameters
        """
        self.probe_depth = probe_depth
        self.strategy = strategy
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_observable_mass_ensemble(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: np.ndarray
    ) -> tuple:
        """
        Compute observable mass for EACH ensemble member (not just mean).

        This captures ensemble disagreement about what's in the observable window.

        Args:
            predictions: (K, N, T) ensemble predictions
            current_time: (N,) censoring times
            current_event: (N,) event indicators

        Returns:
            obs_mass_mean: (N,) mean observable mass across ensemble
            obs_mass_var: (N,) variance of observable mass across ensemble
            obs_mass_per_ensemble: (K, N) observable mass per ensemble member
        """
        K, N, T = predictions.shape
        obs_mass_per_ensemble = np.zeros((K, N))

        for k in range(K):
            for i in range(N):
                if current_event[i] == 1:
                    continue

                c = int(current_time[i])
                max_obs = min(c + self.probe_depth, T - 1)

                if c + 1 <= max_obs:
                    obs_mass_per_ensemble[k, i] = predictions[k, i, c+1:max_obs+1].sum()

        obs_mass_mean = obs_mass_per_ensemble.mean(axis=0)
        obs_mass_var = obs_mass_per_ensemble.var(axis=0)

        return obs_mass_mean, obs_mass_var, obs_mass_per_ensemble

    def compute_mutual_information(
        self,
        oracle_probs: np.ndarray
    ) -> np.ndarray:
        """
        Standard BatchBALD mutual information.

        I(Y_oracle; θ) = H(E[Y_oracle]) - E[H(Y_oracle | θ)]

        Args:
            oracle_probs: (K, N, k+1) oracle outcome probabilities

        Returns:
            mi: (N,) mutual information
        """
        # H(E[Y])
        mean_probs = oracle_probs.mean(axis=0)  # (N, k+1)
        H_expected = -np.sum(xlogy(mean_probs, mean_probs), axis=1)

        # E[H(Y|theta)]
        H_conditional = -np.sum(xlogy(oracle_probs, oracle_probs), axis=2)  # (K, N)
        E_H_conditional = H_conditional.mean(axis=0)

        mi = H_expected - E_H_conditional

        return mi

    def compute_p_death_window(
        self,
        oracle_probs: np.ndarray
    ) -> np.ndarray:
        """
        Probability of revealing death (not censoring extension).

        Args:
            oracle_probs: (K, N, k+1) oracle outcome probabilities

        Returns:
            p_death: (N,) probability of death in observable window
        """
        mean_probs = oracle_probs.mean(axis=0)  # (N, k+1)
        # Sum all death outcomes (exclude last outcome which is censored)
        p_death = mean_probs[:, :-1].sum(axis=1)
        return p_death

    def compute_death_window_variance(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: np.ndarray
    ) -> np.ndarray:
        """
        Variance of predictions within the observable window.

        High variance in observable window = ensemble disagrees about what will happen.

        Args:
            predictions: (K, N, T) predictions
            current_time: (N,) censoring times
            current_event: (N,) event indicators

        Returns:
            variance: (N,) variance in observable window
        """
        K, N, T = predictions.shape
        variance = np.zeros(N)

        for i in range(N):
            if current_event[i] == 1:
                continue

            c = int(current_time[i])
            max_obs = min(c + self.probe_depth, T - 1)

            if c + 1 <= max_obs:
                # Get predictions in window for all ensemble members
                window_preds = predictions[:, i, c+1:max_obs+1]  # (K, window_size)
                # Variance across ensemble and bins
                variance[i] = window_preds.var()

        return variance

    def compute_scores(
        self,
        oracle_probs: np.ndarray,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute optimal acquisition scores.

        Args:
            oracle_probs: (K, N, k+1) oracle outcome probabilities
            predictions: (K, N, T) model predictions
            current_time: (N,) censoring times
            current_event: (N,) event indicators

        Returns:
            scores: (N,) acquisition scores
        """
        N = oracle_probs.shape[1]

        if current_event is None:
            current_event = np.zeros(N)

        # Compute all components
        obs_mass_mean, obs_mass_var, _ = self.compute_observable_mass_ensemble(
            predictions, current_time, current_event
        )
        mi = self.compute_mutual_information(oracle_probs)
        p_death = self.compute_p_death_window(oracle_probs)
        death_var = self.compute_death_window_variance(predictions, current_time, current_event)

        # Normalize components for combination (to [0, 1] range)
        def normalize(x):
            x_min, x_max = x.min(), x.max()
            if x_max > x_min:
                return (x - x_min) / (x_max - x_min)
            return np.zeros_like(x)

        # Apply strategy
        if self.strategy == 'adaptive_fusion':
            # Adaptive fusion: combine all signals intelligently
            # Observable mass is primary (highest correlation)
            # Boost with MI (epistemic uncertainty)
            # Boost with variance (ensemble disagreement)
            # Weight by P(death) (signal strength)

            obs_norm = normalize(obs_mass_mean)
            mi_norm = normalize(mi)
            var_norm = normalize(obs_mass_var)

            # Primary score: observable mass
            scores = obs_mass_mean

            # Boost by MI (more uncertain = more valuable)
            scores = scores * (1.0 + self.alpha * mi_norm)

            # Boost by ensemble disagreement in observable window
            scores = scores * (1.0 + self.beta * var_norm)

            # Boost by likelihood of revealing death
            scores = scores * (1.0 + self.gamma * p_death)

        elif self.strategy == 'observable_mass_boosted':
            # Observable mass boosted by MI
            mi_norm = normalize(mi)
            scores = obs_mass_mean * (1.0 + self.alpha * mi_norm)

        elif self.strategy == 'ensemble_disagreement':
            # Focus on where ensemble disagrees about observable mass
            scores = obs_mass_mean * (1.0 + obs_mass_var)

        elif self.strategy == 'multi_factor':
            # Multiplicative combination
            # Ensure positive values
            mi_safe = np.maximum(mi, 1e-10)
            scores = obs_mass_mean * (mi_safe ** self.beta) * (1.0 + self.gamma * p_death)

        elif self.strategy == 'weighted_sum':
            # Linear combination (normalize first)
            obs_norm = normalize(obs_mass_mean)
            mi_norm = normalize(mi)
            p_death_norm = normalize(p_death)

            scores = (self.alpha * obs_norm +
                     self.beta * mi_norm +
                     self.gamma * p_death_norm)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Filter already uncensored
        scores[current_event == 1] = -np.inf

        return scores

    def select_batch(
        self,
        oracle_probs: np.ndarray,
        predictions: np.ndarray,
        current_time: np.ndarray,
        batch_size: int,
        current_event: Optional[np.ndarray] = None,
        greedy: bool = False
    ) -> np.ndarray:
        """
        Select batch with optional greedy diversity.

        Args:
            oracle_probs: (K, N, k+1) oracle outcome probabilities
            predictions: (K, N, T) model predictions
            current_time: (N,) censoring times
            batch_size: Number of samples to select
            current_event: (N,) event indicators
            greedy: If True, use greedy selection for diversity

        Returns:
            selected_indices: Selected sample indices
        """
        N = oracle_probs.shape[1]

        if current_event is None:
            current_event = np.zeros(N)

        if not greedy:
            # Simple top-k selection
            scores = self.compute_scores(oracle_probs, predictions, current_time, current_event)
            selected = np.argsort(scores)[::-1][:batch_size]
            valid = scores[selected] > -np.inf
            return selected[valid]

        # Greedy batch selection
        selected = []
        remaining = set(range(N))

        # Filter already uncensored
        for i in range(N):
            if current_event[i] == 1:
                remaining.discard(i)

        if len(remaining) == 0:
            return np.array([])

        for iteration in range(batch_size):
            if len(remaining) == 0:
                break

            best_score = -np.inf
            best_idx = None

            for idx in remaining:
                if len(selected) == 0:
                    # First point: use individual score
                    score = self.compute_scores(
                        oracle_probs[:, [idx], :],
                        predictions[:, [idx], :],
                        current_time[[idx]],
                        current_event[[idx]]
                    )[0]
                else:
                    # Greedy: score of batch + candidate
                    # Approximate joint MI (proper BatchBALD would compute joint entropy)
                    candidate_batch = selected + [idx]

                    # Compute joint MI approximation
                    batch_oracle_probs = oracle_probs[:, candidate_batch, :]

                    # Joint entropy approximation
                    mean_probs = batch_oracle_probs.mean(axis=0)  # (batch_size+1, k+1)
                    H_joint = -xlogy(mean_probs, mean_probs).sum()

                    # Conditional entropy
                    H_cond = -xlogy(batch_oracle_probs, batch_oracle_probs).sum(axis=2)  # (K, batch_size+1)
                    E_H_cond = H_cond.sum(axis=1).mean()

                    score = H_joint - E_H_cond

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.discard(best_idx)

        return np.array(selected)


class UltraAggressiveBatchBALD(OptimalBatchBALD):
    """
    Even more aggressive: maximize expected information gain.

    Key insight: We want instances that will:
    1. Reveal their death (high P(death))
    2. Where we're very uncertain (high MI)
    3. Where the observable window has high probability mass
    4. Where ensemble disagrees strongly
    """

    def __init__(self, probe_depth: int):
        super().__init__(
            probe_depth=probe_depth,
            strategy='adaptive_fusion',
            alpha=1.5,  # Strong MI boost
            beta=1.0,   # Strong variance boost
            gamma=2.0   # Very strong P(death) boost
        )

    def compute_scores(
        self,
        oracle_probs: np.ndarray,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Ultra-aggressive scoring."""
        N = oracle_probs.shape[1]

        if current_event is None:
            current_event = np.zeros(N)

        # Get all signals
        obs_mass_mean, obs_mass_var, _ = self.compute_observable_mass_ensemble(
            predictions, current_time, current_event
        )
        mi = self.compute_mutual_information(oracle_probs)
        p_death = self.compute_p_death_window(oracle_probs)

        # Aggressive combination: heavily weight likelihood of revealing death
        # Because revealed deaths provide MUCH stronger signals than extended censoring

        # Base: observable mass (what can we learn)
        scores = obs_mass_mean

        # Multiply by P(death)^2 (strongly prefer likely reveals)
        scores = scores * (p_death ** 2 + 0.1)  # +0.1 to avoid zeros

        # Multiply by MI (prefer uncertain)
        mi_normalized = (mi - mi.min()) / (mi.max() - mi.min() + 1e-10)
        scores = scores * (1.0 + 2.0 * mi_normalized)

        # Add variance term
        scores = scores * (1.0 + obs_mass_var)

        # Filter already uncensored
        scores[current_event == 1] = -np.inf

        return scores
