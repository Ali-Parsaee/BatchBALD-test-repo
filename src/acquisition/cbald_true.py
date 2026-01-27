"""
True C-BALD (Censored-BALD) acquisition function for survival analysis.

Implements the paper-correct C-BALD formulation:
C-BALD(x) = I((y,l);θ|x) = I(y;θ|l,x) + I(l;θ|x)

Where:
- I(l;θ|x): Mutual information between censoring indicator and model
- I(y;θ|l,x): Mutual information between outcome and model, conditioned on censoring status
"""

import numpy as np
from typing import Optional
from scipy.special import xlogy


def _entropy_categorical(p, eps=1e-12):
    """
    Compute entropy of categorical distribution.

    Args:
        p: [..., C] probabilities, sum over last dim = 1
        eps: Small constant for numerical stability

    Returns:
        entropy: [...] entropy values
    """
    p = np.clip(p, eps, 1.0)
    return -np.sum(xlogy(p, p), axis=-1)


def _entropy_bernoulli(q, eps=1e-12):
    """
    Compute entropy of Bernoulli distribution.

    Args:
        q: [...] probabilities in (0,1)
        eps: Small constant for numerical stability

    Returns:
        entropy: [...] entropy values
    """
    q = np.clip(q, eps, 1.0 - eps)
    return -(xlogy(q, q) + xlogy(1 - q, 1 - q))


class TrueCBALD:
    """
    True C-BALD acquisition function implementing the paper-correct formulation.

    C-BALD decomposes the mutual information into:
    1. I(l;θ|x): Information about whether observation is censored
    2. I(y;θ|l,x): Information about the outcome given censoring status
    """

    def __init__(self, probe_depth: int, eps: float = 1e-12):
        """
        Args:
            probe_depth: Oracle probe depth (k bins)
            eps: Small constant for numerical stability
        """
        self.probe_depth = probe_depth
        self.eps = eps

    def _compute_lambda_ensemble(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: np.ndarray
    ) -> np.ndarray:
        """
        Compute lambda_ens: p(l=1|x,θ) for each ensemble member.

        l=1 means "uncensored" (death observed), l=0 means "censored".
        This is the probability that death occurs within the observable window.

        Args:
            predictions: (K, N, T) predicted survival probabilities
            current_time: (N,) current censoring times
            current_event: (N,) current event indicators

        Returns:
            lambda_ens: (K, N) probability of being uncensored for each ensemble member
        """
        K, N, T = predictions.shape
        lambda_ens = np.zeros((K, N))

        for i in range(N):
            if current_event[i] == 1:
                # Already uncensored
                lambda_ens[:, i] = 1.0
                continue

            c = int(current_time[i])
            max_observable = min(c + self.probe_depth, T - 1)

            # Probability of death within observable window [c+1, max_observable]
            # Sum of death probabilities in this range
            if c + 1 < T:
                # Can't have died before c+1 (already censored at c)
                probs = predictions[:, i, :].copy()  # (K, T)

                # Zero out probabilities before current censoring time
                probs[:, :c+1] = 0

                # Renormalize
                probs = probs / (probs.sum(axis=1, keepdims=True) + self.eps)

                # Sum probabilities in observable window
                if c + 1 <= max_observable:
                    lambda_ens[:, i] = probs[:, c+1:max_observable+1].sum(axis=1)

        return lambda_ens

    def _estimate_z_idx(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray
    ) -> np.ndarray:
        """
        Estimate censoring threshold z_idx as proxy for observed value.

        Uses expected bin index under mean predictive distribution.

        Args:
            predictions: (K, N, T) predicted probabilities
            current_time: (N,) current censoring times

        Returns:
            z_idx: (N,) estimated bin indices
        """
        K, N, T = predictions.shape
        mean_probs = predictions.mean(axis=0)  # (N, T)

        z_idx = np.zeros(N, dtype=int)

        for i in range(N):
            c = int(current_time[i])
            probs = mean_probs[i].copy()

            # Zero out probabilities before censoring time
            probs[:c+1] = 0
            probs = probs / (probs.sum() + self.eps)

            # Expected bin index
            bin_ids = np.arange(T)
            exp_bin = (probs * bin_ids).sum()
            z_idx[i] = int(np.clip(np.round(exp_bin), c, T - 1))

        return z_idx

    def compute_cbald_score(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: np.ndarray,
        z_idx: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute C-BALD scores.

        C-BALD(x) = I(l;θ|x) + I(y;θ|l,x)

        Args:
            predictions: (K, N, T) predicted probabilities from ensemble
            current_time: (N,) current censoring times
            current_event: (N,) current event indicators
            z_idx: (N,) optional censoring threshold indices

        Returns:
            cbald_scores: (N,) C-BALD acquisition scores
        """
        K, N, T = predictions.shape

        # Normalize predictions
        probs_ens = predictions.copy()
        probs_ens = np.clip(probs_ens, self.eps, 1.0)
        probs_ens = probs_ens / probs_ens.sum(axis=2, keepdims=True)

        # Compute lambda_ens: p(l=1|x,θ) where l=1 means uncensored
        lambda_ens = self._compute_lambda_ensemble(
            probs_ens, current_time, current_event
        )  # (K, N)

        # Estimate z_idx if not provided
        if z_idx is None:
            z_idx = self._estimate_z_idx(probs_ens, current_time)  # (N,)

        # ---- Term A: I(l;θ|x) = H(E[p(l|x,θ)]) - E[H(p(l|x,θ))]
        q_bar = lambda_ens.mean(axis=0)  # (N,)
        H_qbar = _entropy_bernoulli(q_bar, eps=self.eps)  # (N,)
        H_q_theta = _entropy_bernoulli(lambda_ens, eps=self.eps)  # (K, N)
        I_l = H_qbar - H_q_theta.mean(axis=0)  # (N,)

        # ---- Term B: I(y;θ|l,x) ≈ (1-qbar)*I_y_unc + qbar*I_y_cens

        # B1) Uncensored label y entropy MI (standard BALD for categorical)
        p_bar = probs_ens.mean(axis=0)  # (N, T)
        H_pbar = _entropy_categorical(p_bar, eps=self.eps)  # (N,)
        H_p_theta = _entropy_categorical(probs_ens, eps=self.eps)  # (K, N)
        I_y_unc = H_pbar - H_p_theta.mean(axis=0)  # (N,)

        # B2) Censored observation MI using survival mass beyond z
        # S = sum_{t >= z} p_t (analog of 1 - Φ(z) in paper)
        # Create mask for t >= z_idx
        t_indices = np.arange(T).reshape(1, 1, T)  # (1, 1, T)
        z_indices = z_idx.reshape(1, N, 1)  # (1, N, 1)
        mask = (t_indices >= z_indices).astype(float)  # (1, N, T)

        # Survival mass for each ensemble member
        S_theta = (probs_ens * mask).sum(axis=2)  # (K, N)
        S_theta = np.clip(S_theta, self.eps, 1.0)
        S_bar = S_theta.mean(axis=0)  # (N,)

        # Entropy of censored observation ~ -log(S)
        H_Sbar = -np.log(np.clip(S_bar, self.eps, 1.0))  # (N,)
        H_Stheta = -np.log(S_theta)  # (K, N)
        I_y_cens = H_Sbar - H_Stheta.mean(axis=0)  # (N,)

        # Blend by expected censoring probability
        # q_bar = P(uncensored), so weight uncensored by q_bar
        I_y_given_l = q_bar * I_y_unc + (1.0 - q_bar) * I_y_cens  # (N,)

        # Total C-BALD score
        cbald = I_l + I_y_given_l  # (N,)

        return cbald

    def compute_scores(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        current_event: np.ndarray
    ) -> np.ndarray:
        """
        Compute acquisition scores for all instances.

        Args:
            predictions: (K, N, T) ensemble predictions
            current_time: (N,) current censoring times
            current_event: (N,) event indicators

        Returns:
            scores: (N,) C-BALD acquisition scores
        """
        scores = self.compute_cbald_score(
            predictions, current_time, current_event
        )

        # Set score to -inf for already uncensored
        scores[current_event == 1] = -np.inf

        return scores

    def select_batch(
        self,
        predictions: np.ndarray,
        current_time: np.ndarray,
        batch_size: int,
        current_event: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Select batch greedily based on C-BALD scores.

        Note: This is a simple greedy selection. For better results,
        could implement joint selection like BatchBALD.

        Args:
            predictions: (K, N, T) ensemble predictions
            current_time: (N,) current censoring times
            batch_size: Number of samples to select
            current_event: (N,) event indicators

        Returns:
            selected_indices: (batch_size,) indices of selected samples
        """
        K, N, T = predictions.shape

        # Compute scores
        scores = self.compute_scores(predictions, current_time, current_event)

        # Select top-k
        # Filter out already selected (score = -inf)
        valid_mask = np.isfinite(scores)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return np.array([])

        # Get scores for valid indices
        valid_scores = scores[valid_indices]

        # Sort by score (descending)
        sorted_idx = np.argsort(-valid_scores)

        # Select top batch_size
        n_select = min(batch_size, len(sorted_idx))
        selected = valid_indices[sorted_idx[:n_select]]

        return selected
