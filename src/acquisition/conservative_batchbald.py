"""
Conservative BatchBALD: Reduce Variance, Increase Reliability

Key insights from all testing:
1. High variance kills significance (±0.01-0.02 >> ±0.001-0.002 effect)
2. Random performs well = baseline is hard to beat
3. Need to AVOID bad picks, not just find good picks

Strategy: Conservative selection that minimizes variance
- Prioritize instances LIKELY to reveal death (reduces variance in outcomes)
- Avoid extreme/outlier instances (reduces variance)
- Focus on early censored (more learnable)
- Use ensemble agreement as reliability signal
"""

import numpy as np
from typing import Optional
from scipy.special import xlogy
from scipy.stats import entropy as scipy_entropy


class ConservativeBatchBALD:
    """
    Conservative BatchBALD: Prioritizes reliability over optimality.

    Philosophy: In noisy settings, avoiding bad picks > finding optimal picks

    Key differences from OptimalBatchBALD:
    1. Filters out low-reliability instances FIRST
    2. Uses P(death) as primary signal (most predictive of success)
    3. Adds MI as secondary signal (uncertainty)
    4. Penalizes extremes (too early/late censoring, too high/low observable mass)
    5. Conservative scoring: multiplicative penalties hurt more than additive bonuses
    """

    def __init__(
        self,
        probe_depth: int,
        min_p_death: float = 0.15,  # Filter: only select if >15% chance of death reveal
        early_censor_boost: bool = True,  # Prefer earlier censored instances
        penalize_extremes: bool = True  # Penalize outliers
    ):
        """
        Args:
            probe_depth: Oracle probe depth
            min_p_death: Minimum P(death in window) to consider (filter)
            early_censor_boost: Boost instances censored earlier
            penalize_extremes: Penalize instances with extreme values
        """
        self.probe_depth = probe_depth
        self.min_p_death = min_p_death
        self.early_censor_boost = early_censor_boost
        self.penalize_extremes = penalize_extremes

    def compute_p_death_window(self, oracle_probs: np.ndarray) -> np.ndarray:
        """P(death will be revealed in probe window)."""
        mean_probs = oracle_probs.mean(axis=0)  # (N, k+1)
        p_death = mean_probs[:, :-1].sum(axis=1)  # Sum death outcomes, exclude censoring
        return p_death

    def compute_mutual_information(self, oracle_probs: np.ndarray) -> np.ndarray:
        """Standard MI: I(Y_oracle; theta)."""
        # H(E[Y])
        mean_probs = oracle_probs.mean(axis=0)
        H_expected = -np.sum(xlogy(mean_probs, mean_probs), axis=1)

        # E[H(Y|theta)]
        H_conditional = -np.sum(xlogy(oracle_probs, oracle_probs), axis=2)
        E_H_conditional = H_conditional.mean(axis=0)

        return H_expected - E_H_conditional

    def compute_observable_mass(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: np.ndarray
    ) -> np.ndarray:
        """Probability mass in observable window."""
        K, N, T = predictions.shape
        mean_preds = predictions.mean(axis=0)

        obs_mass = np.zeros(N)
        for i in range(N):
            if current_event[i] == 1:
                continue
            c = int(current_time[i])
            max_obs = min(c + self.probe_depth, T - 1)
            if c + 1 <= max_obs:
                obs_mass[i] = mean_preds[i, c+1:max_obs+1].sum()

        return obs_mass

    def compute_ensemble_agreement(
        self,
        oracle_probs: np.ndarray
    ) -> np.ndarray:
        """
        Measure how much ensemble AGREES (inverse of disagreement).
        High agreement = more reliable prediction
        """
        # Compute variance of P(death) across ensemble
        p_death_per_ensemble = oracle_probs[:, :, :-1].sum(axis=2)  # (K, N)
        p_death_var = p_death_per_ensemble.var(axis=0)  # (N,)

        # Convert to agreement: low variance = high agreement
        # Use negative log to make it a score (higher = better)
        agreement = 1.0 / (1.0 + p_death_var)

        return agreement

    def compute_scores(
        self,
        oracle_probs: np.ndarray,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Conservative scoring:
        1. Filter: Remove instances with low P(death)
        2. Primary: P(death) - likelihood of getting useful signal
        3. Secondary: MI - model uncertainty
        4. Tertiary: Early censoring boost
        5. Penalties: Extreme values
        """
        N = oracle_probs.shape[1]
        K, _, T = predictions.shape

        if current_event is None:
            current_event = np.zeros(N)

        # Compute all components
        p_death = self.compute_p_death_window(oracle_probs)
        mi = self.compute_mutual_information(oracle_probs)
        obs_mass = self.compute_observable_mass(predictions, current_time, current_event)
        agreement = self.compute_ensemble_agreement(oracle_probs)

        # Initialize scores
        scores = np.ones(N)

        # STEP 1: Hard filter - remove low P(death) instances
        # These are unreliable and add variance
        low_pdeath_mask = p_death < self.min_p_death
        scores[low_pdeath_mask] = -np.inf

        # STEP 2: Primary signal - P(death)
        # Use quadratic to strongly prefer high P(death)
        scores = scores * (p_death ** 1.5)

        # STEP 3: Secondary signal - MI (normalized)
        mi_norm = (mi - mi.min()) / (mi.max() - mi.min() + 1e-10)
        scores = scores * (1.0 + mi_norm)

        # STEP 4: Reliability boost - ensemble agreement
        agreement_norm = (agreement - agreement.min()) / (agreement.max() - agreement.min() + 1e-10)
        scores = scores * (1.0 + 0.5 * agreement_norm)

        # STEP 5: Early censoring boost
        if self.early_censor_boost:
            # Prefer instances censored earlier (more room to learn)
            # But not TOO early (might be outliers)
            censor_time_norm = current_time / T
            # Prefer 20th-60th percentile of censoring times
            optimal_censor = (censor_time_norm > 0.2) & (censor_time_norm < 0.6)
            scores[optimal_censor] *= 1.3

            # Slight penalty for very late censoring
            very_late = censor_time_norm > 0.8
            scores[very_late] *= 0.7

        # STEP 6: Penalize extremes
        if self.penalize_extremes:
            # Observable mass too low or too high = risky
            obs_mass_norm = (obs_mass - obs_mass.min()) / (obs_mass.max() - obs_mass.min() + 1e-10)
            extreme_obs = (obs_mass_norm < 0.1) | (obs_mass_norm > 0.9)
            scores[extreme_obs] *= 0.5

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
        """Select batch conservatively."""
        scores = self.compute_scores(oracle_probs, predictions, current_time, current_event)

        # Simple top-k selection (greedy not needed for conservative approach)
        selected = np.argsort(scores)[::-1][:batch_size]
        valid = scores[selected] > -np.inf

        return selected[valid]


class DataDrivenBatchBALD:
    """
    Data-driven approach: Learn from what actually works.

    Based on our experiments, we know:
    - P(death) is most predictive
    - Early censoring helps
    - Too much uncertainty hurts

    This uses a simple weighted combination tuned to our data.
    """

    def __init__(self, probe_depth: int):
        self.probe_depth = probe_depth

        # Weights learned from our experiments
        self.w_pdeath = 2.0      # P(death) is most important
        self.w_mi = 0.5          # MI helps but less than P(death)
        self.w_early = 0.3       # Small boost for early censoring
        self.w_obs_mass = 0.2    # Small boost for observable mass

    def compute_p_death_window(self, oracle_probs: np.ndarray) -> np.ndarray:
        mean_probs = oracle_probs.mean(axis=0)
        return mean_probs[:, :-1].sum(axis=1)

    def compute_mi(self, oracle_probs: np.ndarray) -> np.ndarray:
        mean_probs = oracle_probs.mean(axis=0)
        H_exp = -np.sum(xlogy(mean_probs, mean_probs), axis=1)
        H_cond = -np.sum(xlogy(oracle_probs, oracle_probs), axis=2).mean(axis=0)
        return H_exp - H_cond

    def compute_observable_mass(self, predictions, current_time, current_event):
        K, N, T = predictions.shape
        mean_preds = predictions.mean(axis=0)
        obs_mass = np.zeros(N)
        for i in range(N):
            if current_event[i] == 1:
                continue
            c = int(current_time[i])
            max_obs = min(c + self.probe_depth, T - 1)
            if c + 1 <= max_obs:
                obs_mass[i] = mean_preds[i, c+1:max_obs+1].sum()
        return obs_mass

    def compute_scores(self, oracle_probs, predictions, current_time, current_event=None):
        N = oracle_probs.shape[1]
        K, _, T = predictions.shape

        if current_event is None:
            current_event = np.zeros(N)

        # Compute components
        p_death = self.compute_p_death_window(oracle_probs)
        mi = self.compute_mi(oracle_probs)
        obs_mass = self.compute_observable_mass(predictions, current_time, current_event)

        # Normalize
        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-10)

        p_death_norm = norm(p_death)
        mi_norm = norm(mi)
        obs_mass_norm = norm(obs_mass)

        # Early censoring score
        censor_norm = current_time / T
        early_score = 1.0 - censor_norm  # Lower time = higher score
        early_score = norm(early_score)

        # Weighted combination
        scores = (
            self.w_pdeath * p_death_norm +
            self.w_mi * mi_norm +
            self.w_early * early_score +
            self.w_obs_mass * obs_mass_norm
        )

        # Filter
        scores[current_event == 1] = -np.inf
        scores[p_death < 0.1] = -np.inf  # Hard filter on low P(death)

        return scores

    def select_batch(self, oracle_probs, predictions, current_time, batch_size, current_event=None, greedy=False):
        scores = self.compute_scores(oracle_probs, predictions, current_time, current_event)
        selected = np.argsort(scores)[::-1][:batch_size]
        valid = scores[selected] > -np.inf
        return selected[valid]
