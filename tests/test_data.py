import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.synthetic import generate_survival_data, artificially_censor, split_data


def test_generate_survival_data():
    """Test synthetic data generation."""
    print("\n=== Testing synthetic data generation ===")

    data = generate_survival_data(
        n_samples=100,
        n_features=5,
        n_time_bins=10,
        censoring_rate=0.3,
        random_state=42
    )

    assert 'X' in data
    assert 'time' in data
    assert 'event' in data

    assert data['X'].shape == (100, 5)
    assert data['time'].shape == (100,)
    assert data['event'].shape == (100,)

    # Check time bins are valid
    assert np.all(data['time'] >= 0)
    assert np.all(data['time'] < 10)

    # Check censoring rate is approximately correct
    censoring_actual = (data['event'] == 0).mean()
    print(f"Censoring rate: {censoring_actual:.2f} (target: 0.30)")
    assert 0.2 < censoring_actual < 0.4  # Allow some variance

    print("✓ Data generation test passed")


def test_artificial_censoring():
    """Test artificial censoring."""
    print("\n=== Testing artificial censoring ===")

    time = np.array([5, 8, 3, 9, 2])
    event = np.array([1, 1, 0, 1, 1])

    artificial_time, artificial_event = artificially_censor(
        time, event, proportion=0.6, random_state=42
    )

    # Check that artificial times are <= original times
    assert np.all(artificial_time <= time)

    # Check that artificially censored samples have event=0
    censored_mask = artificial_event == 0
    n_censored = censored_mask.sum()
    print(f"Censored {n_censored}/5 samples")

    # Should have censored ~3 samples (60% of 5)
    assert 2 <= n_censored <= 4

    print("✓ Artificial censoring test passed")


def test_split_data():
    """Test train/test split."""
    print("\n=== Testing data split ===")

    data = generate_survival_data(
        n_samples=100,
        n_features=5,
        n_time_bins=10,
        random_state=42
    )

    train_data, test_data = split_data(data, train_size=0.7, random_state=42)

    assert len(train_data['X']) == 70
    assert len(test_data['X']) == 30

    # Check no overlap
    train_set = set([tuple(x) for x in train_data['X']])
    test_set = set([tuple(x) for x in test_data['X']])
    assert len(train_set.intersection(test_set)) == 0

    print("✓ Data split test passed")


if __name__ == '__main__':
    test_generate_survival_data()
    test_artificial_censoring()
    test_split_data()
    print("\n=== All data tests passed! ===\n")
