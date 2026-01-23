"""
Debug BatchBALD to understand why it's inconsistent.
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


def debug_selection(random_state=42):
    """Debug what BatchBALD vs Entropy are selecting."""

    print(f"\n{'='*80}")
    print("DEBUGGING ACQUISITION FUNCTION SELECTIONS")
    print(f"{'='*80}\n")

    # Generate data
    data = generate_survival_data(
        n_samples=200,
        n_features=5,
        n_time_bins=8,
        censoring_rate=0.3,
        random_state=random_state
    )

    train_data, test_data = split_data(data, train_size=0.7, random_state=random_state)

    true_train_time = train_data['time'].copy()
    true_train_event = train_data['event'].copy()

    artificial_time, artificial_event = artificially_censor(
        train_data['time'],
        train_data['event'],
        proportion=0.5,
        random_state=random_state
    )

    # Train model
    print("Training model...")
    model = BayesianSurvivalModel(
        n_features=5,
        n_time_bins=8,
        n_ensemble=3,
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

    # Oracle
    probe_depth = 2
    oracle = Oracle(true_train_time, true_train_event, probe_depth)

    # Oracle probs for BatchBALD
    oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
        train_preds, artificial_time, artificial_event
    )

    # Compute scores
    batchbald = SurvivalBatchBALD(probe_depth=probe_depth)
    entropy_acq = EntropyAcquisition()

    batchbald_scores = batchbald.compute_scores(oracle_probs, artificial_event)
    entropy_scores = entropy_acq.compute_scores(train_preds, artificial_event)

    # Filter out already uncensored
    censored_mask = artificial_event == 0
    censored_indices = np.where(censored_mask)[0]

    print(f"Total samples: {len(train_data['X'])}")
    print(f"Censored samples available for query: {len(censored_indices)}")

    # Analyze score distributions
    batchbald_scores_valid = batchbald_scores[censored_mask]
    entropy_scores_valid = entropy_scores[censored_mask]

    print(f"\nBatchBALD scores: min={batchbald_scores_valid.min():.4f}, "
          f"max={batchbald_scores_valid.max():.4f}, mean={batchbald_scores_valid.mean():.4f}")
    print(f"Entropy scores: min={entropy_scores_valid.min():.4f}, "
          f"max={entropy_scores_valid.max():.4f}, mean={entropy_scores_valid.mean():.4f}")

    # Select batches
    batch_size = 10
    batchbald_batch = batchbald.select_batch(oracle_probs, batch_size, artificial_event)
    entropy_batch = entropy_acq.select_batch(train_preds, batch_size, artificial_event)

    print(f"\nBatchBALD selected: {batchbald_batch}")
    print(f"Entropy selected: {entropy_batch}")

    # Analyze what will actually be revealed
    print(f"\n{'='*80}")
    print("ORACLE REVELATION ANALYSIS")
    print(f"{'='*80}\n")

    for method_name, selected in [('BatchBALD', batchbald_batch), ('Entropy', entropy_batch)]:
        updated_time, updated_event = oracle.query(
            selected, artificial_time.copy(), artificial_event.copy()
        )

        n_revealed_deaths = ((updated_event[selected] - artificial_event[selected]) == 1).sum()
        n_extended_censor = ((updated_time[selected] - artificial_time[selected]) > 0).sum()

        print(f"{method_name}:")
        print(f"  Deaths revealed: {n_revealed_deaths}/{len(selected)}")
        print(f"  Censoring extended: {n_extended_censor}/{len(selected)}")

        # Analyze true outcomes
        true_deaths_in_window = 0
        true_deaths_beyond_window = 0

        for idx in selected:
            c = artificial_time[idx]
            max_obs = c + probe_depth

            if true_train_event[idx] == 1:  # True death
                if true_train_time[idx] <= max_obs:
                    true_deaths_in_window += 1
                else:
                    true_deaths_beyond_window += 1

        print(f"  True deaths within probe window: {true_deaths_in_window}")
        print(f"  True deaths beyond probe window: {true_deaths_beyond_window}")
        print()

    # Check correlation between scores and true informativeness
    print(f"{'='*80}")
    print("SCORE VS TRUE INFORMATIVENESS")
    print(f"{'='*80}\n")

    # For each censored sample, compute "true informativeness"
    # = whether revealing it would actually give new information
    informativeness = np.zeros(len(train_data['X']))

    for idx in censored_indices:
        c = artificial_time[idx]
        max_obs = c + probe_depth

        # Most informative: death within window
        if true_train_event[idx] == 1 and true_train_time[idx] <= max_obs:
            informativeness[idx] = 1.0
        # Somewhat informative: censoring extension
        elif true_train_time[idx] > c:
            informativeness[idx] = 0.5
        else:
            informativeness[idx] = 0.0

    # Correlation
    from scipy.stats import spearmanr

    info_valid = informativeness[censored_mask]

    corr_batchbald = spearmanr(batchbald_scores_valid, info_valid)
    corr_entropy = spearmanr(entropy_scores_valid, info_valid)

    print(f"Correlation with true informativeness:")
    print(f"  BatchBALD: ρ={corr_batchbald.correlation:.3f}, p={corr_batchbald.pvalue:.3f}")
    print(f"  Entropy: ρ={corr_entropy.correlation:.3f}, p={corr_entropy.pvalue:.3f}")


if __name__ == '__main__':
    debug_selection(random_state=42)
    print("\n" + "="*80)
    print("Debug complete!")
    print("="*80)
