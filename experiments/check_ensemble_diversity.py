"""
Check ensemble diversity to diagnose BatchBALD issues.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.synthetic import generate_survival_data, artificially_censor, split_data
from src.models.survival_model import BayesianSurvivalModel


def check_diversity():
    """Check if ensemble members are diverse enough."""

    print("\n" + "="*80)
    print("ENSEMBLE DIVERSITY CHECK")
    print("="*80 + "\n")

    # Generate data
    data = generate_survival_data(
        n_samples=200,
        n_features=5,
        n_time_bins=8,
        censoring_rate=0.3,
        random_state=42
    )

    train_data, test_data = split_data(data, train_size=0.7, random_state=42)

    artificial_time, artificial_event = artificially_censor(
        train_data['time'],
        train_data['event'],
        proportion=0.5,
        random_state=42
    )

    # Train ensemble
    print("Training ensemble...")
    model = BayesianSurvivalModel(
        n_features=5,
        n_time_bins=8,
        n_ensemble=5,  # Use 5 members
        hidden_size=32
    )
    model.fit(
        train_data['X'],
        artificial_time,
        artificial_event,
        epochs=30,
        batch_size=32,
        verbose=False
    )

    # Get predictions
    train_preds = model.predict_proba(train_data['X'])  # (K, N, T)
    K, N, T = train_preds.shape

    print(f"Ensemble size: {K}")
    print(f"Samples: {N}")
    print(f"Time bins: {T}\n")

    # Compute pairwise disagreement between ensemble members
    print("Pairwise disagreement between ensemble members:")
    print("(Average absolute difference in probabilities)\n")

    disagreements = []
    for i in range(K):
        for j in range(i+1, K):
            # Average absolute difference
            diff = np.abs(train_preds[i] - train_preds[j]).mean()
            disagreements.append(diff)
            print(f"Member {i} vs Member {j}: {diff:.4f}")

    avg_disagreement = np.mean(disagreements)
    print(f"\nAverage disagreement: {avg_disagreement:.4f}")

    if avg_disagreement < 0.05:
        print("⚠️  WARNING: Very low diversity! Ensemble members are too similar.")
    elif avg_disagreement < 0.10:
        print("⚠️  Low diversity. This may reduce BatchBALD effectiveness.")
    else:
        print("✓ Good diversity.")

    # Check variance across ensemble for each sample
    print("\n" + "-"*80)
    print("Per-sample variance across ensemble:")
    print("-"*80 + "\n")

    # Variance in predicted class (argmax)
    predicted_classes = train_preds.argmax(axis=2)  # (K, N)

    # For each sample, count how many different classes ensemble predicts
    unique_predictions = []
    for n in range(min(N, 20)):  # Check first 20 samples
        unique = len(set(predicted_classes[:, n]))
        unique_predictions.append(unique)

    avg_unique = np.mean(unique_predictions)
    print(f"Samples checked: {len(unique_predictions)}")
    print(f"Average unique predictions per sample: {avg_unique:.2f}/{K}")

    if avg_unique < 1.5:
        print("⚠️  WARNING: Ensemble members almost always agree on the same class!")
    elif avg_unique < 2.5:
        print("⚠️  Low prediction diversity.")
    else:
        print("✓ Good prediction diversity.")

    # Compute entropy of ensemble predictions
    print("\n" + "-"*80)
    print("Ensemble uncertainty (mutual information):")
    print("-"*80 + "\n")

    # Average predictions
    mean_preds = train_preds.mean(axis=0)  # (N, T)

    # Entropy of mean
    entropy_mean = -(mean_preds * np.log(mean_preds + 1e-8)).sum(axis=1)

    # Mean of entropies
    entropies_individual = -(train_preds * np.log(train_preds + 1e-8)).sum(axis=2)  # (K, N)
    mean_entropy = entropies_individual.mean(axis=0)  # (N,)

    # Mutual information
    mutual_info = entropy_mean - mean_entropy

    print(f"Mutual information statistics:")
    print(f"  Min: {mutual_info.min():.4f}")
    print(f"  Max: {mutual_info.max():.4f}")
    print(f"  Mean: {mutual_info.mean():.4f}")
    print(f"  Std: {mutual_info.std():.4f}")

    if mutual_info.mean() < 0.01:
        print("\n⚠️  CRITICAL: Very low mutual information!")
        print("   This means BatchBALD has almost no signal to work with.")
        print("   Possible causes:")
        print("   - Ensemble members are too similar (low diversity)")
        print("   - Model is overconfident")
        print("   - Need more ensemble members or stronger regularization")
    elif mutual_info.mean() < 0.05:
        print("\n⚠️  Low mutual information. BatchBALD may struggle.")
    else:
        print("\n✓ Reasonable mutual information for BatchBALD.")


if __name__ == '__main__':
    check_diversity()
    print("\n" + "="*80)
    print("Diversity check complete!")
    print("="*80)
