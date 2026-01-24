"""
Quick test to verify OptimalBatchBALD works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.acquisition.optimal_batchbald import OptimalBatchBALD, UltraAggressiveBatchBALD


def test_optimal_batchbald():
    """Test OptimalBatchBALD basic functionality."""
    print("\n" + "="*80)
    print("Testing OptimalBatchBALD")
    print("="*80 + "\n")

    # Create synthetic data
    K, N, T = 5, 100, 20
    probe_depth = 3

    # Random predictions
    np.random.seed(42)
    predictions = np.random.rand(K, N, T)
    predictions = predictions / predictions.sum(axis=2, keepdims=True)

    # Random censoring
    current_time = np.random.randint(0, T-5, size=N)
    current_event = np.zeros(N)
    current_event[:10] = 1  # First 10 are uncensored

    # Create oracle probs (simplified)
    oracle_probs = np.random.rand(K, N, probe_depth + 1)
    oracle_probs = oracle_probs / oracle_probs.sum(axis=2, keepdims=True)

    # Test each strategy
    strategies = [
        'adaptive_fusion',
        'observable_mass_boosted',
        'ensemble_disagreement',
        'multi_factor',
        'weighted_sum'
    ]

    print("Testing all strategies:")
    for strategy in strategies:
        print(f"\n  Testing strategy: {strategy}")
        acq = OptimalBatchBALD(probe_depth=probe_depth, strategy=strategy)

        # Compute scores
        scores = acq.compute_scores(oracle_probs, predictions, current_time, current_event)

        # Check outputs
        assert scores.shape == (N,), f"Expected shape ({N},), got {scores.shape}"
        assert np.all(scores[:10] == -np.inf), "Uncensored samples should have -inf score"
        assert np.any(scores[10:] > -np.inf), "Should have some valid scores"

        print(f"    ✓ Scores shape: {scores.shape}")
        print(f"    ✓ Valid scores: {(scores > -np.inf).sum()}/{N}")
        print(f"    ✓ Score range: [{scores[scores > -np.inf].min():.4f}, {scores[scores > -np.inf].max():.4f}]")

        # Test batch selection
        batch_size = 10
        selected = acq.select_batch(
            oracle_probs,
            predictions,
            current_time,
            batch_size,
            current_event,
            use_greedy=False
        )

        assert len(selected) <= batch_size, f"Selected {len(selected)} > {batch_size}"
        assert np.all(current_event[selected] == 0), "Should only select censored samples"

        print(f"    ✓ Batch selection: {len(selected)} samples")

    print("\n" + "="*80)
    print("Testing UltraAggressiveBatchBALD")
    print("="*80 + "\n")

    acq_ultra = UltraAggressiveBatchBALD(probe_depth=probe_depth)
    scores_ultra = acq_ultra.compute_scores(oracle_probs, predictions, current_time, current_event)

    print(f"  ✓ Ultra-aggressive scores computed")
    print(f"  ✓ Valid scores: {(scores_ultra > -np.inf).sum()}/{N}")
    print(f"  ✓ Score range: [{scores_ultra[scores_ultra > -np.inf].min():.4f}, {scores_ultra[scores_ultra > -np.inf].max():.4f}]")

    selected_ultra = acq_ultra.select_batch(
        oracle_probs,
        predictions,
        current_time,
        batch_size=10,
        current_event=current_event,
        use_greedy=False
    )

    print(f"  ✓ Batch selection: {len(selected_ultra)} samples")

    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80 + "\n")


if __name__ == '__main__':
    test_optimal_batchbald()
