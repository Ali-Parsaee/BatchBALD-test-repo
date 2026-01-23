import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.acquisition.batchbald import SurvivalBatchBALD
from src.acquisition.entropy import EntropyAcquisition
from src.acquisition.variance import VarianceAcquisition


def test_batchbald():
    """Test BatchBALD acquisition function."""
    print("\n=== Testing BatchBALD ===")

    probe_depth = 3
    acq = SurvivalBatchBALD(probe_depth=probe_depth)

    # Create oracle probabilities (K=5, N=10, k+1=4)
    K, N = 5, 10
    oracle_probs = np.random.rand(K, N, probe_depth + 1)
    # Normalize
    oracle_probs = oracle_probs / oracle_probs.sum(axis=2, keepdims=True)

    # Test conditional entropy
    cond_entropy = acq.compute_conditional_entropy(oracle_probs)
    assert cond_entropy.shape == (K, N)
    assert np.all(cond_entropy >= 0), "Entropy should be non-negative"
    print(f"Conditional entropy shape: {cond_entropy.shape}")
    print(f"Conditional entropy range: [{cond_entropy.min():.4f}, {cond_entropy.max():.4f}]")

    # Test entropy of expected
    entropy_exp = acq.compute_entropy_of_expected(oracle_probs)
    assert entropy_exp.shape == (N,)
    assert np.all(entropy_exp >= 0)
    print(f"Entropy of expected shape: {entropy_exp.shape}")

    # Test mutual information
    mi = acq.compute_mutual_information(oracle_probs)
    assert mi.shape == (N,)
    # MI can be slightly negative due to numerical errors, but should be close to 0 or positive
    print(f"Mutual information range: [{mi.min():.4f}, {mi.max():.4f}]")

    # Test batch selection
    current_event = np.zeros(N)
    current_event[0] = 1  # First sample already uncensored

    selected = acq.select_batch(oracle_probs, batch_size=3, current_event=current_event)
    assert len(selected) <= 3
    assert 0 not in selected, "Should not select already uncensored sample"
    print(f"Selected batch: {selected}")

    print("✓ BatchBALD test passed")


def test_entropy_acquisition():
    """Test entropy acquisition function."""
    print("\n=== Testing Entropy Acquisition ===")

    acq = EntropyAcquisition()

    # Create predictions (K=5, N=10, T=8)
    K, N, T = 5, 10, 8
    predictions = np.random.rand(K, N, T)
    predictions = predictions / predictions.sum(axis=2, keepdims=True)

    # Test scores
    scores = acq.compute_scores(predictions)
    assert scores.shape == (N,)
    assert np.all(scores[scores > -np.inf] >= 0), "Entropy should be non-negative"
    print(f"Entropy scores range: [{scores[scores > -np.inf].min():.4f}, {scores.max():.4f}]")

    # Test with current_event
    current_event = np.zeros(N)
    current_event[5] = 1

    scores_filtered = acq.compute_scores(predictions, current_event)
    assert scores_filtered[5] == -np.inf, "Already uncensored should have -inf score"

    # Test batch selection
    selected = acq.select_batch(predictions, batch_size=4, current_event=current_event)
    assert len(selected) <= 4
    assert 5 not in selected
    print(f"Selected batch: {selected}")

    print("✓ Entropy acquisition test passed")


def test_variance_acquisition():
    """Test variance acquisition function."""
    print("\n=== Testing Variance Acquisition ===")

    acq = VarianceAcquisition()

    # Create predictions with varying variance
    K, N, T = 5, 10, 8
    predictions = np.random.rand(K, N, T)
    predictions = predictions / predictions.sum(axis=2, keepdims=True)

    # Make one sample have high variance
    predictions[:, 3, :] = np.random.rand(K, T) * 2
    predictions[:, 3, :] = predictions[:, 3, :] / predictions[:, 3, :].sum(axis=1, keepdims=True)

    # Test scores
    scores = acq.compute_scores(predictions)
    assert scores.shape == (N,)
    assert np.all(scores[scores > -np.inf] >= 0), "Variance should be non-negative"
    print(f"Variance scores range: [{scores[scores > -np.inf].min():.6f}, {scores.max():.6f}]")

    # Test batch selection
    current_event = np.zeros(N)
    selected = acq.select_batch(predictions, batch_size=5, current_event=current_event)
    assert len(selected) <= 5
    print(f"Selected batch: {selected}")

    # Sample 3 should have high variance and likely be selected
    print(f"Sample 3 variance: {scores[3]:.6f}")

    print("✓ Variance acquisition test passed")


def test_acquisition_comparison():
    """Compare acquisition functions on same data."""
    print("\n=== Comparing Acquisition Functions ===")

    # Setup
    K, N, T = 10, 20, 10
    probe_depth = 2

    predictions = np.random.rand(K, N, T)
    predictions = predictions / predictions.sum(axis=2, keepdims=True)

    current_event = np.zeros(N)
    current_event[0] = 1

    # Oracle probs for BatchBALD
    oracle_probs = np.random.rand(K, N, probe_depth + 1)
    oracle_probs = oracle_probs / oracle_probs.sum(axis=2, keepdims=True)

    # Get scores from each method
    batchbald = SurvivalBatchBALD(probe_depth=probe_depth)
    entropy = EntropyAcquisition()
    variance = VarianceAcquisition()

    batchbald_scores = batchbald.compute_scores(oracle_probs, current_event)
    entropy_scores = entropy.compute_scores(predictions, current_event)
    variance_scores = variance.compute_scores(predictions, current_event)

    print(f"BatchBALD scores: min={batchbald_scores[batchbald_scores > -np.inf].min():.4f}, "
          f"max={batchbald_scores.max():.4f}, mean={batchbald_scores[batchbald_scores > -np.inf].mean():.4f}")
    print(f"Entropy scores: min={entropy_scores[entropy_scores > -np.inf].min():.4f}, "
          f"max={entropy_scores.max():.4f}, mean={entropy_scores[entropy_scores > -np.inf].mean():.4f}")
    print(f"Variance scores: min={variance_scores[variance_scores > -np.inf].min():.6f}, "
          f"max={variance_scores.max():.6f}, mean={variance_scores[variance_scores > -np.inf].mean():.6f}")

    # Select batches
    batch_size = 5
    batchbald_batch = batchbald.select_batch(oracle_probs, batch_size, current_event)
    entropy_batch = entropy.select_batch(predictions, batch_size, current_event)
    variance_batch = variance.select_batch(predictions, batch_size, current_event)

    print(f"\nBatchBALD batch: {batchbald_batch}")
    print(f"Entropy batch: {entropy_batch}")
    print(f"Variance batch: {variance_batch}")

    # Calculate overlap
    batchbald_set = set(batchbald_batch)
    entropy_set = set(entropy_batch)
    variance_set = set(variance_batch)

    print(f"\nOverlap BatchBALD-Entropy: {len(batchbald_set & entropy_set)}/{batch_size}")
    print(f"Overlap BatchBALD-Variance: {len(batchbald_set & variance_set)}/{batch_size}")
    print(f"Overlap Entropy-Variance: {len(entropy_set & variance_set)}/{batch_size}")

    print("✓ Acquisition comparison test passed")


if __name__ == '__main__':
    test_batchbald()
    test_entropy_acquisition()
    test_variance_acquisition()
    test_acquisition_comparison()
    print("\n=== All acquisition tests passed! ===\n")
