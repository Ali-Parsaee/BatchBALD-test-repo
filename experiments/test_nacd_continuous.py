"""
Test on NACD with continuous times for training, discrete bins for active learning.

Model trains on continuous times but outputs discrete bins.
Oracle and acquisition functions work with discrete bins.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel

from src.oracle.oracle import Oracle
from src.models.survival_model import BayesianSurvivalModel
from src.evaluation.metrics import concordance_index


def load_nacd():
    """Load NACD dataset."""
    df = pd.read_csv('data/MIMIC/NACD/NACD_Full.csv')
    df = df.rename(columns={'SURVIVAL': 'time', 'CENSORED': 'event'})
    print(f"Loaded NACD: {df.shape}")
    print(f"Event rate: {df['event'].mean():.2%}")
    print(f"Time range: [{df['time'].min():.1f}, {df['time'].max():.1f}]")
    return df


def split_data(df, train_frac=0.7, random_state=42):
    """Split data."""
    np.random.seed(random_state)
    n = len(df)
    indices = np.random.permutation(n)
    n_train = int(n * train_frac)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def time_to_bin(continuous_times, bin_edges):
    """Map continuous times to discrete bins."""
    bins = np.digitize(continuous_times, bin_edges[1:])
    bins = np.clip(bins, 0, len(bin_edges) - 2)
    return bins


def artificially_censor_continuous(time_continuous, time_binned, event, proportion=0.5, random_state=None):
    """
    Artificially censor continuous times.

    Returns:
        artificial_time_cont: Censored continuous times
        artificial_time_bin: Censored binned times
        artificial_event: Censored events
    """
    if random_state is not None:
        np.random.seed(random_state)

    n = len(time_continuous)
    n_to_censor = int(n * proportion)

    # Select indices to censor (only uncensored events)
    uncensored_idx = np.where(event == 1)[0]
    if len(uncensored_idx) == 0:
        return time_continuous.copy(), time_binned.copy(), event.copy()

    n_to_censor = min(n_to_censor, len(uncensored_idx))
    idx_to_censor = np.random.choice(uncensored_idx, size=n_to_censor, replace=False)

    artificial_time_cont = time_continuous.copy()
    artificial_time_bin = time_binned.copy()
    artificial_event = event.copy()

    for idx in idx_to_censor:
        # Censor at fraction of original time
        original_time = time_continuous[idx]
        original_bin = time_binned[idx]

        if original_bin > 0:
            # Censor at random bin before the true bin
            new_bin = np.random.randint(0, max(1, original_bin))
            # Estimate continuous time for that bin (use original time as reference)
            if original_bin > 0:
                new_time_cont = original_time * (new_bin / original_bin)
            else:
                new_time_cont = original_time * 0.5

            artificial_time_cont[idx] = new_time_cont
            artificial_time_bin[idx] = new_bin
        else:
            artificial_time_cont[idx] = original_time * 0.5
            artificial_time_bin[idx] = 0

        artificial_event[idx] = 0

    return artificial_time_cont, artificial_time_bin, artificial_event


def test_strategy(strategy_name, score_func, data, probe_depth=3, batch_size=30, n_runs=3):
    """Test single strategy."""

    results = []

    for run in range(n_runs):
        # Split data
        train_df, test_df = split_data(data, train_frac=0.7, random_state=42 + run)

        feature_cols = [col for col in train_df.columns if col not in ['time', 'event']]
        X_train = train_df[feature_cols].values.astype(np.float32)
        time_train_cont = train_df['time'].values  # CONTINUOUS
        event_train = train_df['event'].values

        X_test = test_df[feature_cols].values.astype(np.float32)
        time_test_cont = test_df['time'].values  # CONTINUOUS
        event_test = test_df['event'].values

        # Create bin edges from ALL data
        all_times = np.concatenate([time_train_cont, time_test_cont])
        n_time_bins = min(20, int(np.sqrt(len(time_train_cont))))
        bin_edges = np.quantile(all_times[all_times > 0], np.linspace(0, 1, n_time_bins + 1))

        # Convert to bins for oracle/AL logic
        time_train_bin = time_to_bin(time_train_cont, bin_edges)
        time_test_bin = time_to_bin(time_test_cont, bin_edges)

        print(f"\n    Run {run+1}: Train={len(X_train)}, Test={len(X_test)}, Features={X_train.shape[1]}, Bins={n_time_bins}")

        # Store true labels
        true_train_time_cont = time_train_cont.copy()
        true_train_time_bin = time_train_bin.copy()
        true_train_event = event_train.copy()

        # Artificially censor
        artificial_time_cont, artificial_time_bin, artificial_event = artificially_censor_continuous(
            time_train_cont, time_train_bin, event_train, proportion=0.5, random_state=42 + run
        )

        # Train initial model with CONTINUOUS times
        model_init = BayesianSurvivalModel(
            n_features=X_train.shape[1],
            n_time_bins=n_time_bins,
            n_ensemble=5,
            hidden_size=64
        )
        model_init.fit(X_train, artificial_time_cont, artificial_event, epochs=30, batch_size=32, verbose=False)

        # Test performance
        test_preds_init = model_init.predict_proba(X_test)
        initial_c_index = concordance_index(test_preds_init, time_test_bin, event_test)
        print(f"      Initial C-index: {initial_c_index:.4f}")

        # Get ensemble predictions for acquisition (DISCRETE BINS)
        train_preds = model_init.predict_proba(X_train)  # (K, N, T_bins)

        # Oracle works with BINNED times
        oracle = Oracle(true_train_time_bin, true_train_event, probe_depth)
        oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
            train_preds, artificial_time_bin, artificial_event
        )

        # Compute scores with BINNED times
        scores = score_func(train_preds, oracle_probs, artificial_time_bin, artificial_event, X_train)

        # Filter already uncensored
        scores[artificial_event == 1] = -np.inf

        # Select batch
        valid_mask = scores > -np.inf
        n_valid = valid_mask.sum()
        batch_size_use = min(batch_size, n_valid)

        if batch_size_use == 0:
            print(f"      No valid samples!")
            results.append({'improvement': 0.0, 'n_revealed': 0})
            continue

        selected = np.argsort(scores)[-batch_size_use:]

        # Query oracle with BINNED times
        updated_time_bin, updated_event = oracle.query(selected, artificial_time_bin, artificial_event)
        n_revealed = ((updated_event - artificial_event) > 0).sum()

        # Convert updated BINNED times back to approximate CONTINUOUS times for training
        # For newly revealed deaths, use a random time within the bin
        updated_time_cont = artificial_time_cont.copy()
        max_time = bin_edges[-1]  # Maximum valid time

        for idx in selected:
            if updated_event[idx] == 1 and artificial_event[idx] == 0:
                # Death revealed - estimate continuous time
                revealed_bin = int(updated_time_bin[idx])
                if revealed_bin < len(bin_edges) - 1:
                    # Random time within the bin
                    bin_start = bin_edges[revealed_bin]
                    bin_end = bin_edges[revealed_bin + 1]
                    updated_time_cont[idx] = np.random.uniform(bin_start, bin_end)
                else:
                    # Last bin - use edge
                    updated_time_cont[idx] = min(bin_edges[revealed_bin], max_time * 0.99)
            elif updated_event[idx] == 0 and updated_time_bin[idx] != artificial_time_bin[idx]:
                # Censoring extended - use bin midpoint
                new_bin = int(updated_time_bin[idx])
                if new_bin < len(bin_edges) - 1:
                    updated_time_cont[idx] = (bin_edges[new_bin] + bin_edges[new_bin + 1]) / 2
                else:
                    updated_time_cont[idx] = min(bin_edges[new_bin], max_time * 0.99)

        # Ensure all times are within valid range
        updated_time_cont = np.clip(updated_time_cont, 0, max_time * 0.99)

        print(f"      Selected {batch_size_use}, revealed {n_revealed} events")

        # Retrain with CONTINUOUS times
        model_updated = BayesianSurvivalModel(
            n_features=X_train.shape[1],
            n_time_bins=n_time_bins,
            n_ensemble=5,
            hidden_size=64
        )
        model_updated.fit(X_train, updated_time_cont, updated_event, epochs=30, batch_size=32, verbose=False)

        # Evaluate
        test_preds_updated = model_updated.predict_proba(X_test)
        updated_c_index = concordance_index(test_preds_updated, time_test_bin, event_test)
        improvement = updated_c_index - initial_c_index

        print(f"      Updated C-index: {updated_c_index:.4f} (Δ={improvement:+.4f})")

        results.append({'improvement': improvement, 'n_revealed': n_revealed})

    return results


def main():
    print("\n" + "="*100)
    print("NACD TEST: Continuous Training, Discrete Active Learning")
    print("="*100 + "\n")

    data = load_nacd()

    # Define strategies
    def entropy_strategy(preds, oracle_probs, time_bin, event, X):
        """Entropy on predictions."""
        mean_preds = preds.mean(axis=0)
        return -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=1)

    def variance_strategy(preds, oracle_probs, time_bin, event, X):
        """Variance."""
        return preds.var(axis=0).sum(axis=1)

    def batchbald_mi(preds, oracle_probs, time_bin, event, X):
        """BatchBALD MI."""
        mean_probs = oracle_probs.mean(axis=0)
        H_exp = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
        H_cond = -np.sum(oracle_probs * np.log(oracle_probs + 1e-10), axis=2)
        return H_exp - H_cond.mean(axis=0)

    def random_strategy(preds, oracle_probs, time_bin, event, X):
        """Random."""
        return np.random.rand(len(time_bin))

    strategies = {
        'Random': random_strategy,
        'Entropy': entropy_strategy,
        'Variance': variance_strategy,
        'BatchBALD MI': batchbald_mi,
    }

    n_runs = 3
    all_results = {}

    for strategy_name, score_func in strategies.items():
        print(f"\n{'='*100}")
        print(f"Testing: {strategy_name}")
        print(f"{'='*100}")

        try:
            results = test_strategy(strategy_name, score_func, data, probe_depth=3, batch_size=30, n_runs=n_runs)
            all_results[strategy_name] = results

            improvements = [r['improvement'] for r in results]
            print(f"\n  Summary: Δ={np.mean(improvements):+.4f}±{np.std(improvements):.4f}, Wins={sum(1 for x in improvements if x > 0)}/{n_runs}")

        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Results
    print(f"\n\n{'='*100}")
    print("RESULTS")
    print(f"{'='*100}\n")

    print(f"{'Strategy':<20} {'Mean Δ':<18} {'Wins':<10} {'Avg Reveals'}")
    print("-" * 70)

    ranked = []
    for name, results in all_results.items():
        imps = [r['improvement'] for r in results]
        revs = [r['n_revealed'] for r in results]
        ranked.append({
            'name': name,
            'mean': np.mean(imps),
            'std': np.std(imps),
            'wins': sum(1 for x in imps if x > 0),
            'reveals': np.mean(revs)
        })

    ranked.sort(key=lambda x: x['mean'], reverse=True)

    for rank, r in enumerate(ranked, 1):
        marker = "🏆" if rank == 1 else " "
        print(f"{marker} {r['name']:<20} {r['mean']:+.4f}±{r['std']:.4f}     {r['wins']}/{n_runs}      {r['reveals']:.1f}")

    print(f"\n{'='*100}\n")


if __name__ == '__main__':
    main()
