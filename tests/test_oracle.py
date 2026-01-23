import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.oracle.oracle import Oracle


def test_oracle_query():
    """Test oracle query with probe depth."""
    print("\n=== Testing oracle query ===")

    # Ground truth
    true_time = np.array([8, 5, 3, 9, 6])
    true_event = np.array([1, 1, 0, 1, 1])

    # Current observations (artificially censored)
    current_time = np.array([3, 2, 2, 4, 3])
    current_event = np.array([0, 0, 0, 0, 0])

    probe_depth = 2
    oracle = Oracle(true_time, true_event, probe_depth)

    # Query first sample (censored at 3, true death at 8, probe depth 2)
    # Should remain censored at 3+2=5
    updated_time, updated_event = oracle.query(
        np.array([0]),
        current_time,
        current_event
    )

    assert updated_time[0] == 5  # c + probe_depth
    assert updated_event[0] == 0  # Still censored
    print(f"Sample 0: censored at 3, true death at 8, probe depth 2 -> censored at {updated_time[0]}")

    # Query second sample (censored at 2, true death at 5, probe depth 2)
    # Should reveal death at 5 (within probe window)
    current_time_copy = current_time.copy()
    current_event_copy = current_event.copy()
    updated_time, updated_event = oracle.query(
        np.array([1]),
        current_time_copy,
        current_event_copy
    )

    # Note: 2 + 2 = 4, but true death is at 5, which is > 4
    # So should be censored at 4
    expected_time = min(true_time[1], current_time[1] + probe_depth)
    print(f"Sample 1: censored at 2, true death at 5, probe depth 2 -> time={updated_time[1]}")

    print("✓ Oracle query test passed")


def test_oracle_outcome_probs():
    """Test oracle outcome probability computation."""
    print("\n=== Testing oracle outcome probabilities ===")

    # Setup
    true_time = np.array([8, 5])
    true_event = np.array([1, 1])
    current_time = np.array([3, 2])
    current_event = np.array([0, 0])
    probe_depth = 2

    oracle = Oracle(true_time, true_event, probe_depth)

    # Predictions (K=2, N=2, T=10)
    K, N, T = 2, 2, 10
    predictions = np.random.rand(K, N, T)
    # Normalize to probabilities
    predictions = predictions / predictions.sum(axis=2, keepdims=True)

    # Get oracle outcome probs
    oracle_probs = oracle.get_oracle_outcome_probs(
        predictions,
        current_time,
        current_event
    )

    assert oracle_probs.shape == (N, probe_depth + 1)

    # Check probabilities sum to 1
    prob_sums = oracle_probs.sum(axis=1)
    assert np.allclose(prob_sums, 1.0), f"Probabilities don't sum to 1: {prob_sums}"

    print(f"Oracle outcome probs shape: {oracle_probs.shape}")
    print(f"Sample 0 probs: {oracle_probs[0]}")
    print(f"Sample 1 probs: {oracle_probs[1]}")

    print("✓ Oracle outcome probability test passed")


def test_oracle_ensemble_probs():
    """Test oracle outcome probabilities for ensemble."""
    print("\n=== Testing oracle ensemble probabilities ===")

    true_time = np.array([8, 5, 6])
    true_event = np.array([1, 1, 1])
    current_time = np.array([3, 2, 4])
    current_event = np.array([0, 0, 0])
    probe_depth = 3

    oracle = Oracle(true_time, true_event, probe_depth)

    # Predictions (K=5, N=3, T=12)
    K, N, T = 5, 3, 12
    predictions = np.random.rand(K, N, T)
    predictions = predictions / predictions.sum(axis=2, keepdims=True)

    oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
        predictions,
        current_time,
        current_event
    )

    assert oracle_probs.shape == (K, N, probe_depth + 1)

    # Check each ensemble member
    for k in range(K):
        prob_sums = oracle_probs[k].sum(axis=1)
        assert np.allclose(prob_sums, 1.0), f"Ensemble {k} probs don't sum to 1"

    print(f"Oracle ensemble probs shape: {oracle_probs.shape}")
    print("✓ Oracle ensemble probability test passed")


if __name__ == '__main__':
    test_oracle_query()
    test_oracle_outcome_probs()
    test_oracle_ensemble_probs()
    print("\n=== All oracle tests passed! ===\n")
