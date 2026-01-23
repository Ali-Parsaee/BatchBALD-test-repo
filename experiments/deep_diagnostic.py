"""
Deep diagnostic to find why BatchBALD isn't winning.

Investigates:
1. Are oracle probabilities computed correctly?
2. Are mutual information scores meaningful?
3. Are we selecting different/better samples than entropy?
4. Is there a bug in the implementation?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.synthetic import generate_survival_data, artificially_censor, split_data
from src.models.survival_model import BayesianSurvivalModel
from src.oracle.oracle import Oracle
from src.acquisition.batchbald import SurvivalBatchBALD
from src.acquisition.entropy import EntropyAcquisition
from src.evaluation.metrics import concordance_index


def deep_diagnostic():
    """Deep dive into what's happening."""

    print("\n" + "="*100)
    print("DEEP DIAGNOSTIC: Why Isn't BatchBALD Winning?")
    print("="*100 + "\n")

    # Generate data
    np.random.seed(42)
    data = generate_survival_data(
        n_samples=500,
        n_features=10,
        n_time_bins=15,
        censoring_rate=0.3,
        random_state=42
    )

    train_data, test_data = split_data(data, train_size=0.7, random_state=42)
    true_train_time = train_data['time'].copy()
    true_train_event = train_data['event'].copy()

    artificial_time, artificial_event = artificially_censor(
        train_data['time'], train_data['event'],
        proportion=0.5, random_state=42
    )

    print(f"Dataset: {len(train_data['X'])} train, {len(test_data['X'])} test")
    print(f"Censored: {(artificial_event == 0).sum()}/{len(artificial_event)}\n")

    # Train model
    print("Training model...")
    model = BayesianSurvivalModel(
        n_features=10, n_time_bins=15, n_ensemble=5, hidden_size=64
    )
    model.fit(
        train_data['X'], artificial_time, artificial_event,
        epochs=50, batch_size=32, verbose=False
    )

    # Get predictions
    train_preds = model.predict_proba(train_data['X'])  # (K, N, T)
    K, N, T = train_preds.shape

    print(f"Model predictions: K={K}, N={N}, T={T}\n")

    # Check ensemble diversity
    print("="*100)
    print("1. ENSEMBLE DIVERSITY CHECK")
    print("="*100 + "\n")

    disagreement = []
    for i in range(K):
        for j in range(i+1, K):
            diff = np.abs(train_preds[i] - train_preds[j]).mean()
            disagreement.append(diff)

    avg_disagreement = np.mean(disagreement)
    print(f"Average ensemble disagreement: {avg_disagreement:.4f}")

    # Mutual information
    mean_preds = train_preds.mean(axis=0)
    entropy_mean = -(mean_preds * np.log(mean_preds + 1e-8)).sum(axis=1)
    entropies_individual = -(train_preds * np.log(train_preds + 1e-8)).sum(axis=2)
    mean_entropy = entropies_individual.mean(axis=0)
    mutual_info = entropy_mean - mean_entropy

    print(f"Mutual information: min={mutual_info.min():.4f}, max={mutual_info.max():.4f}, mean={mutual_info.mean():.4f}")

    if mutual_info.mean() < 0.1:
        print("⚠️  WARNING: Low mutual information! BatchBALD has weak signal.\n")
    else:
        print("✓ Good mutual information.\n")

    # Oracle setup
    probe_depth = 5
    batch_size = 50
    oracle = Oracle(true_train_time, true_train_event, probe_depth)

    # Get oracle probabilities
    print("="*100)
    print("2. ORACLE PROBABILITY ANALYSIS")
    print("="*100 + "\n")

    oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
        train_preds, artificial_time, artificial_event
    )  # (K, N, k+1)

    print(f"Oracle probs shape: {oracle_probs.shape}")

    # Check if oracle probs are reasonable
    mean_oracle_probs = oracle_probs.mean(axis=0)  # (N, k+1)

    # For censored samples, check distribution
    censored_mask = artificial_event == 0
    censored_indices = np.where(censored_mask)[0][:10]  # First 10 censored

    print("\nSample oracle outcome probabilities (first 10 censored samples):")
    print(f"{'Idx':<6} {'C.Time':<8} {'P(death in window)':<20} {'P(beyond window)':<20}")
    print("-"*80)

    for idx in censored_indices:
        c_time = artificial_time[idx]
        oracle_prob = mean_oracle_probs[idx]

        # P(death within window) = sum of first k probabilities
        p_death_window = oracle_prob[:-1].sum()
        p_beyond_window = oracle_prob[-1]

        print(f"{idx:<6} {c_time:<8} {p_death_window:.4f}               {p_beyond_window:.4f}")

    # Are oracle probs well-distributed or all concentrated?
    max_probs = mean_oracle_probs.max(axis=1)
    print(f"\nMax probability per sample: mean={max_probs.mean():.4f}, std={max_probs.std():.4f}")

    if max_probs.mean() > 0.9:
        print("⚠️  WARNING: Oracle probabilities are too peaked! Model is overconfident.\n")
    else:
        print("✓ Oracle probabilities are reasonably distributed.\n")

    # Compute acquisition scores
    print("="*100)
    print("3. ACQUISITION SCORE COMPARISON")
    print("="*100 + "\n")

    batchbald = SurvivalBatchBALD(probe_depth=probe_depth)
    entropy_acq = EntropyAcquisition()

    batchbald_scores = batchbald.compute_scores(oracle_probs, artificial_event)
    entropy_scores = entropy_acq.compute_scores(train_preds, artificial_event)

    # Filter to censored samples
    batchbald_scores_censored = batchbald_scores[censored_mask]
    entropy_scores_censored = entropy_scores[censored_mask]

    print(f"BatchBALD scores: min={batchbald_scores_censored.min():.4f}, "
          f"max={batchbald_scores_censored.max():.4f}, mean={batchbald_scores_censored.mean():.4f}")
    print(f"Entropy scores:   min={entropy_scores_censored.min():.4f}, "
          f"max={entropy_scores_censored.max():.4f}, mean={entropy_scores_censored.mean():.4f}")

    # Normalize scores to [0, 1] for comparison
    batchbald_norm = (batchbald_scores_censored - batchbald_scores_censored.min()) / \
                     (batchbald_scores_censored.max() - batchbald_scores_censored.min() + 1e-8)
    entropy_norm = (entropy_scores_censored - entropy_scores_censored.min()) / \
                   (entropy_scores_censored.max() - entropy_scores_censored.min() + 1e-8)

    # Correlation between scores
    correlation = np.corrcoef(batchbald_norm, entropy_norm)[0, 1]
    print(f"\nCorrelation between BatchBALD and Entropy scores: {correlation:.4f}")

    if correlation > 0.9:
        print("⚠️  WARNING: BatchBALD and Entropy are selecting nearly the same samples!")
        print("   This means BatchBALD isn't providing unique information.\n")
    else:
        print("✓ BatchBALD and Entropy select different samples.\n")

    # Select batches
    print("="*100)
    print("4. BATCH SELECTION ANALYSIS")
    print("="*100 + "\n")

    batchbald_batch = batchbald.select_batch(oracle_probs, batch_size, artificial_event)
    entropy_batch = entropy_acq.select_batch(train_preds, batch_size, artificial_event)

    print(f"BatchBALD selected: {len(batchbald_batch)} samples")
    print(f"Entropy selected: {len(entropy_batch)} samples")

    # Overlap
    overlap = len(set(batchbald_batch) & set(entropy_batch))
    print(f"Overlap: {overlap}/{batch_size} ({overlap/batch_size*100:.1f}%)")

    if overlap / batch_size > 0.8:
        print("⚠️  WARNING: >80% overlap! BatchBALD selecting nearly same samples as Entropy.\n")
    else:
        print("✓ Different selections.\n")

    # What gets revealed?
    print("="*100)
    print("5. ORACLE REVELATION ANALYSIS")
    print("="*100 + "\n")

    def analyze_selection(selected, name):
        updated_time, updated_event = oracle.query(
            selected, artificial_time.copy(), artificial_event.copy()
        )

        n_revealed = ((updated_event[selected] - artificial_event[selected]) == 1).sum()
        n_extended = ((updated_time[selected] - artificial_time[selected]) > 0).sum()

        # True informativeness
        true_deaths_in_window = 0
        true_deaths_beyond = 0

        for idx in selected:
            c = artificial_time[idx]
            max_obs = c + probe_depth

            if true_train_event[idx] == 1:
                if true_train_time[idx] <= max_obs:
                    true_deaths_in_window += 1
                else:
                    true_deaths_beyond += 1

        print(f"{name}:")
        print(f"  Deaths revealed: {n_revealed}/{len(selected)} ({n_revealed/len(selected)*100:.1f}%)")
        print(f"  Censoring extended: {n_extended}/{len(selected)}")
        print(f"  True deaths in window: {true_deaths_in_window}")
        print(f"  True deaths beyond window: {true_deaths_beyond}")

        return n_revealed, true_deaths_in_window

    bald_revealed, bald_in_window = analyze_selection(batchbald_batch, "BatchBALD")
    print()
    ent_revealed, ent_in_window = analyze_selection(entropy_batch, "Entropy")
    print()

    if bald_in_window <= ent_in_window:
        print("❌ PROBLEM FOUND: BatchBALD is NOT selecting more informative samples!")
        print("   It should select samples where deaths are likely within probe window.\n")
    else:
        print("✓ BatchBALD selects more informative samples.\n")

    # Score vs true informativeness
    print("="*100)
    print("6. SCORE CALIBRATION")
    print("="*100 + "\n")

    # For each censored sample, compute true informativeness
    informativeness = np.zeros(N)

    for idx in np.where(censored_mask)[0]:
        c = artificial_time[idx]
        max_obs = c + probe_depth

        if true_train_event[idx] == 1 and true_train_time[idx] <= max_obs:
            informativeness[idx] = 1.0  # Death revealed
        elif true_train_time[idx] > c:
            informativeness[idx] = 0.3  # Censoring extended
        else:
            informativeness[idx] = 0.0  # No new info

    # Correlation with true informativeness
    from scipy.stats import spearmanr

    info_censored = informativeness[censored_mask]

    corr_bald = spearmanr(batchbald_scores_censored, info_censored)
    corr_ent = spearmanr(entropy_scores_censored, info_censored)

    print(f"Correlation with TRUE informativeness:")
    print(f"  BatchBALD: ρ={corr_bald.correlation:.4f}, p={corr_bald.pvalue:.4f}")
    print(f"  Entropy:   ρ={corr_ent.correlation:.4f}, p={corr_ent.pvalue:.4f}")

    if corr_bald.correlation < 0.1:
        print("\n❌ CRITICAL PROBLEM: BatchBALD scores have NO correlation with true informativeness!")
        print("   This means the mutual information calculation is not working as expected.\n")
    elif corr_bald.correlation < corr_ent.correlation:
        print(f"\n❌ PROBLEM: BatchBALD (ρ={corr_bald.correlation:.4f}) has LOWER correlation than Entropy (ρ={corr_ent.correlation:.4f})")
        print("   BatchBALD should be better at identifying informative samples.\n")
    else:
        print("\n✓ BatchBALD has higher correlation with true informativeness.\n")

    # Final diagnosis
    print("="*100)
    print("DIAGNOSIS SUMMARY")
    print("="*100 + "\n")

    issues = []

    if avg_disagreement < 0.1:
        issues.append("• Low ensemble diversity (disagreement < 0.1)")

    if mutual_info.mean() < 0.1:
        issues.append("• Low mutual information (< 0.1)")

    if max_probs.mean() > 0.9:
        issues.append("• Model is overconfident (oracle probs too peaked)")

    if correlation > 0.9:
        issues.append("• BatchBALD and Entropy selecting nearly identical samples")

    if overlap / batch_size > 0.8:
        issues.append(f"• High batch overlap ({overlap/batch_size*100:.1f}%)")

    if bald_in_window <= ent_in_window:
        issues.append("• BatchBALD not finding more informative samples than Entropy")

    if corr_bald.correlation < 0.1:
        issues.append("• BatchBALD scores uncorrelated with true informativeness")

    if corr_bald.correlation < corr_ent.correlation:
        issues.append("• BatchBALD worse calibrated than Entropy")

    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(issue)
        print("\n🔍 ROOT CAUSE: " + issues[0])
    else:
        print("✓ No obvious implementation issues found.")
        print("\n🤔 BatchBALD may simply not provide enough advantage in this setting.")
        print("   Possible reasons:")
        print("   • Entropy is a very strong baseline for this problem")
        print("   • Limited information from probe depth constraint")
        print("   • Neural network training noise dominates signal")
        print("   • Need much larger datasets for advantage to emerge")

    print("\n" + "="*100 + "\n")


if __name__ == '__main__':
    deep_diagnostic()
