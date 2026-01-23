import numpy as np
from typing import Tuple


def concordance_index(
    predictions: np.ndarray,
    time: np.ndarray,
    event: np.ndarray
) -> float:
    """
    Compute concordance index (C-index) for survival predictions.

    C-index measures the fraction of pairs where the model correctly
    orders the survival times.

    Args:
        predictions: (N, T) predicted probabilities or (K, N, T) ensemble predictions
        time: (N,) observed times
        event: (N,) event indicators (1=death, 0=censored)

    Returns:
        c_index: Concordance index (0.5 = random, 1.0 = perfect)
    """
    if predictions.ndim == 3:
        # Average over ensemble
        predictions = predictions.mean(axis=0)

    N = len(time)

    # Compute predicted survival time (expected value)
    # E[T] = sum_t t * P(T=t)
    time_bins = np.arange(predictions.shape[1])
    predicted_time = (predictions * time_bins).sum(axis=1)

    concordant = 0
    permissible = 0

    for i in range(N):
        for j in range(i + 1, N):
            # Only compare if at least one event occurred
            if event[i] == 0 and event[j] == 0:
                continue

            # Determine which time is shorter
            if time[i] < time[j] and event[i] == 1:
                # i died first
                permissible += 1
                if predicted_time[i] < predicted_time[j]:
                    concordant += 1
                elif predicted_time[i] == predicted_time[j]:
                    concordant += 0.5

            elif time[j] < time[i] and event[j] == 1:
                # j died first
                permissible += 1
                if predicted_time[j] < predicted_time[i]:
                    concordant += 1
                elif predicted_time[i] == predicted_time[j]:
                    concordant += 0.5

            elif time[i] == time[j] and event[i] == 1 and event[j] == 1:
                # Both died at same time
                permissible += 1
                concordant += 0.5

    if permissible == 0:
        return 0.5

    return concordant / permissible


def brier_score(
    predictions: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    evaluation_times: np.ndarray = None
) -> float:
    """
    Compute Brier score for survival predictions.

    Brier score measures calibration of survival probabilities.

    Args:
        predictions: (N, T) or (K, N, T) survival probabilities
        time: (N,) observed times
        event: (N,) event indicators
        evaluation_times: Time points to evaluate (default: all unique times)

    Returns:
        brier_score: Mean Brier score across time points (lower is better)
    """
    if predictions.ndim == 3:
        # Average over ensemble
        predictions = predictions.mean(axis=0)

    N, T = predictions.shape

    if evaluation_times is None:
        # Use all time bins
        evaluation_times = np.arange(T)

    brier_scores = []

    for t in evaluation_times:
        if t >= T:
            continue

        # Compute survival probability at time t
        # S(t) = P(T > t) = sum of probs for bins > t
        if t + 1 < T:
            survival_prob = predictions[:, t+1:].sum(axis=1)
        else:
            survival_prob = np.zeros(N)

        # True survival status at time t
        true_survival = (time > t).astype(float)

        # For censored samples before time t, we can't evaluate
        # Use inverse probability weighting (simplified version)
        weights = np.ones(N)
        for i in range(N):
            if event[i] == 0 and time[i] <= t:
                weights[i] = 0  # Exclude censored samples before evaluation time

        # Brier score: mean squared error
        if weights.sum() > 0:
            bs = np.sum(weights * (survival_prob - true_survival) ** 2) / weights.sum()
            brier_scores.append(bs)

    if len(brier_scores) == 0:
        return 0.0

    return np.mean(brier_scores)


def integrated_brier_score(
    predictions: np.ndarray,
    time: np.ndarray,
    event: np.ndarray
) -> float:
    """
    Compute integrated Brier score over all time points.

    Args:
        predictions: (N, T) or (K, N, T) survival probabilities
        time: (N,) observed times
        event: (N,) event indicators

    Returns:
        ibs: Integrated Brier score
    """
    if predictions.ndim == 3:
        T = predictions.shape[2]
    else:
        T = predictions.shape[1]

    evaluation_times = np.arange(T)
    return brier_score(predictions, time, event, evaluation_times)
