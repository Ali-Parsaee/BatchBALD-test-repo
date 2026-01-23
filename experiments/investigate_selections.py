"""
Investigate WHAT BatchBALD is selecting and WHY it's wrong.
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


def investigate():
    """Detailed investigation of what gets selected."""

    np.random.seed(42)

    # Generate data
    data = generate_survival_data(
        n_samples=500, n_features=10, n_time_bins=15,
        censoring_rate=0.3, random_state=42
    )

    train_data, test_data = split_data(data, train_size=0.7, random_state=42)
    true_train_time = train_data['time'].copy()
    true_train_event = train_data['event'].copy()

    artificial_time, artificial_event = artificially_censor(
        train_data['time'], train_data['event'],
        proportion=0.5, random_state=42
    )

    # Train model
    print("Training model...\n")
    model = BayesianSurvivalModel(
        n_features=10, n_time_bins=15, n_ensemble=5, hidden_size=64
    )
    model.fit(
        train_data['X'], artificial_time, artificial_event,
        epochs=50, batch_size=32, verbose=False
    )

    train_preds = model.predict_proba(train_data['X'])

    # Setup
    probe_depth = 5
    oracle = Oracle(true_train_time, true_train_event, probe_depth)
    oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
        train_preds, artificial_time, artificial_event
    )

    # Compute scores
    batchbald = SurvivalBatchBALD(probe_depth=probe_depth)
    entropy_acq = EntropyAcquisition()

    batchbald_scores = batchbald.compute_scores(oracle_probs, artificial_event)
    entropy_scores = entropy_acq.compute_scores(train_preds, artificial_event)

    # Get top samples by each method
    censored_mask = artificial_event == 0
    censored_indices = np.where(censored_mask)[0]

    # Top 10 by BatchBALD
    bald_ranking = censored_indices[np.argsort(batchbald_scores[censored_mask])[::-1]][:10]

    # Top 10 by Entropy
    ent_ranking = censored_indices[np.argsort(entropy_scores[censored_mask])[::-1]][:10]

    print("="*100)
    print("TOP 10 SAMPLES BY EACH METHOD")
    print("="*100 + "\n")

    print("BatchBALD Top 10:")
    print(f"{'Idx':<6} {'Cens@':<7} {'True':<7} {'Event':<7} {'Reveal?':<10} {'BALD Score':<12} {'Ent Score':<12} {'Oracle Probs'}")
    print("-"*100)

    for idx in bald_ranking:
        c_time = artificial_time[idx]
        true_time = true_train_time[idx]
        true_event = true_train_event[idx]

        # Will it be revealed?
        max_obs = c_time + probe_depth
        if true_event == 1 and true_time <= max_obs:
            reveal = "DEATH✓"
        elif true_time > c_time:
            reveal = "Extended"
        else:
            reveal = "No"

        bald_score = batchbald_scores[idx]
        ent_score = entropy_scores[idx]

        # Oracle probs (averaged across ensemble)
        oracle_prob = oracle_probs.mean(axis=0)[idx]
        oracle_str = f"[{oracle_prob[0]:.2f}, {oracle_prob[1]:.2f}, ..., {oracle_prob[-1]:.2f}]"

        print(f"{idx:<6} {c_time:<7} {true_time:<7} {true_event:<7} {reveal:<10} {bald_score:.4f}      {ent_score:.4f}      {oracle_str}")

    print("\n" + "="*100 + "\n")
    print("Entropy Top 10:")
    print(f"{'Idx':<6} {'Cens@':<7} {'True':<7} {'Event':<7} {'Reveal?':<10} {'BALD Score':<12} {'Ent Score':<12} {'Oracle Probs'}")
    print("-"*100)

    for idx in ent_ranking:
        c_time = artificial_time[idx]
        true_time = true_train_time[idx]
        true_event = true_train_event[idx]

        max_obs = c_time + probe_depth
        if true_event == 1 and true_time <= max_obs:
            reveal = "DEATH✓"
        elif true_time > c_time:
            reveal = "Extended"
        else:
            reveal = "No"

        bald_score = batchbald_scores[idx]
        ent_score = entropy_scores[idx]

        oracle_prob = oracle_probs.mean(axis=0)[idx]
        oracle_str = f"[{oracle_prob[0]:.2f}, {oracle_prob[1]:.2f}, ..., {oracle_prob[-1]:.2f}]"

        print(f"{idx:<6} {c_time:<7} {true_time:<7} {true_event:<7} {reveal:<10} {bald_score:.4f}      {ent_score:.4f}      {oracle_str}")

    # Count reveals
    print("\n" + "="*100)
    print("COMPARISON")
    print("="*100 + "\n")

    def count_reveals(indices):
        deaths = 0
        extended = 0
        nothing = 0

        for idx in indices:
            c_time = artificial_time[idx]
            max_obs = c_time + probe_depth

            if true_train_event[idx] == 1 and true_train_time[idx] <= max_obs:
                deaths += 1
            elif true_train_time[idx] > c_time:
                extended += 1
            else:
                nothing += 1

        return deaths, extended, nothing

    bald_deaths, bald_ext, bald_nothing = count_reveals(bald_ranking)
    ent_deaths, ent_ext, ent_nothing = count_reveals(ent_ranking)

    print(f"BatchBALD top 10: {bald_deaths} deaths revealed, {bald_ext} extended, {bald_nothing} no info")
    print(f"Entropy top 10:   {ent_deaths} deaths revealed, {ent_ext} extended, {ent_nothing} no info")

    print("\n" + "="*100)
    print("ANALYSIS: Why is BatchBALD selecting these samples?")
    print("="*100 + "\n")

    # Look at ensemble disagreement for top BatchBALD samples
    print("Ensemble disagreement for BatchBALD top 10:\n")

    for i, idx in enumerate(bald_ranking[:5]):
        oracle_prob_ensemble = oracle_probs[:, idx, :]  # (K, k+1)

        # Compute variance across ensemble
        variance = oracle_prob_ensemble.var(axis=0)  # (k+1,)

        print(f"Sample {idx} (rank {i+1}):")
        print(f"  Oracle probs per ensemble member:")
        for k in range(oracle_probs.shape[0]):
            print(f"    Member {k}: {oracle_prob_ensemble[k]}")
        print(f"  Variance: {variance}")
        print(f"  Mean variance: {variance.mean():.4f}")

        # True outcome
        c_time = artificial_time[idx]
        max_obs = c_time + probe_depth
        if true_train_event[idx] == 1 and true_train_time[idx] <= max_obs:
            print(f"  TRUE OUTCOME: Death will be REVEALED at bin {true_train_time[idx]}")
        else:
            print(f"  TRUE OUTCOME: No death revealed (censored)")
        print()

    print("="*100)
    print("HYPOTHESIS TEST")
    print("="*100 + "\n")

    # Hypothesis: BatchBALD is selecting samples with high ensemble disagreement
    # but that disagreement is NOT about whether death will be revealed

    # For each sample, compute:
    # 1. Ensemble disagreement about "death in window" vs "censored beyond window"
    # 2. vs total MI

    print("Does ensemble disagreement correlate with informative revelations?\n")

    # Compute "death probability within window" for each ensemble member
    censored_idxs = np.where(censored_mask)[0]

    death_in_window_probs = np.zeros((oracle_probs.shape[0], len(censored_idxs)))

    for i, idx in enumerate(censored_idxs):
        # Sum of probs for death in bins c+1 to c+k
        death_in_window_probs[:, i] = oracle_probs[:, idx, :-1].sum(axis=1)

    # Variance in "death in window" probability across ensemble
    death_in_window_variance = death_in_window_probs.var(axis=0)

    # True informativeness
    true_info = np.zeros(len(censored_idxs))
    for i, idx in enumerate(censored_idxs):
        c_time = artificial_time[idx]
        max_obs = c_time + probe_depth
        if true_train_event[idx] == 1 and true_train_time[idx] <= max_obs:
            true_info[i] = 1.0

    # Correlation
    from scipy.stats import spearmanr

    corr_death_var = spearmanr(death_in_window_variance, true_info)
    corr_mi = spearmanr(batchbald_scores[censored_mask], true_info)

    print(f"Correlation of 'death-in-window variance' with true reveals: ρ={corr_death_var.correlation:.4f}, p={corr_death_var.pvalue:.4f}")
    print(f"Correlation of 'mutual information' with true reveals: ρ={corr_mi.correlation:.4f}, p={corr_mi.pvalue:.4f}")

    if corr_death_var.correlation > corr_mi.correlation:
        print(f"\n💡 INSIGHT: Disagreement about 'death in window' is MORE predictive than total MI!")
        print(f"   This suggests BatchBALD should focus on the death outcomes, not all outcomes equally.\n")

    print("="*100 + "\n")


if __name__ == '__main__':
    investigate()
