import numpy as np
from typing import Tuple, Dict


def generate_survival_data(
    n_samples: int,
    n_features: int,
    n_time_bins: int,
    censoring_rate: float = 0.3,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic survival data with time bins.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_time_bins: Number of discrete time bins
        censoring_rate: Proportion of censored samples
        random_state: Random seed

    Returns:
        Dictionary with:
            - X: Features (n_samples, n_features)
            - time: Survival time bin (n_samples,)
            - event: Event indicator (1=death, 0=censored) (n_samples,)
    """
    np.random.seed(random_state)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate survival times based on features
    # Use a simple linear combination for the hazard
    coefficients = np.random.randn(n_features) * 0.5
    risk_score = X @ coefficients

    # Convert to survival times (higher risk = shorter survival)
    # Use exponential distribution with rate depending on risk
    base_rate = 0.1
    rates = base_rate * np.exp(risk_score)
    survival_times_continuous = np.random.exponential(1.0 / rates)

    # Discretize into bins (bins are [0, 1, 2, ..., n_time_bins-1])
    # Normalize to fit in time bins
    max_time = np.percentile(survival_times_continuous, 95)
    survival_times_continuous = np.clip(survival_times_continuous, 0, max_time)
    time = np.floor(survival_times_continuous / max_time * (n_time_bins - 1)).astype(int)
    time = np.clip(time, 0, n_time_bins - 1)

    # Generate censoring
    event = np.ones(n_samples, dtype=int)
    n_censored = int(n_samples * censoring_rate)
    censored_idx = np.random.choice(n_samples, n_censored, replace=False)
    event[censored_idx] = 0

    # For censored samples, censoring time should be before true death time
    for idx in censored_idx:
        if time[idx] > 0:
            time[idx] = np.random.randint(0, time[idx] + 1)

    return {
        'X': X,
        'time': time,
        'event': event
    }


def artificially_censor(
    time: np.ndarray,
    event: np.ndarray,
    proportion: float = 0.5,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Artificially censor a proportion of the training data.

    For uncensored samples: censor them with artificial_event=0 at random time < true time
    For already censored samples: further censor them with artificial_event=0

    Args:
        time: Original time bins
        event: Original event indicators
        proportion: Proportion to artificially censor
        random_state: Random seed

    Returns:
        artificial_time: New censoring times
        artificial_event: New event indicators (0 for artificially censored)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(time)
    n_to_censor = int(n_samples * proportion)

    # Randomly select indices to artificially censor
    idx_to_censor = np.random.choice(n_samples, n_to_censor, replace=False)

    artificial_time = time.copy()
    artificial_event = event.copy()

    for idx in idx_to_censor:
        original_time = time[idx]
        if original_time > 0:
            # Censor at random time between 0 and original time
            artificial_time[idx] = np.random.randint(0, original_time)
        else:
            artificial_time[idx] = 0
        # Mark as censored
        artificial_event[idx] = 0

    return artificial_time, artificial_event


def split_data(
    data: Dict[str, np.ndarray],
    train_size: float = 0.7,
    random_state: int = 42
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Split data into train and test sets."""
    np.random.seed(random_state)
    n_samples = len(data['X'])
    n_train = int(n_samples * train_size)

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_data = {
        'X': data['X'][train_idx],
        'time': data['time'][train_idx],
        'event': data['event'][train_idx]
    }

    test_data = {
        'X': data['X'][test_idx],
        'time': data['time'][test_idx],
        'event': data['event'][test_idx]
    }

    return train_data, test_data
