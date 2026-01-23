"""
Test active learning strategies on real NACD dataset with user's Bayesian MTLR model.

This uses the actual data and model from the user's codebase to test what works
in practice for survival active learning with probe depth constraints.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ttest_rel
from datetime import datetime

# Import user's code
try:
    from Making_and_getting_datasets import Get_Dataset
    from model import BayesMtlr, mtlr_nll
    print("Successfully imported user's modules")
except Exception as e:
    print(f"Error importing user modules: {e}")
    sys.exit(1)


def artificially_censor(time, event, proportion=0.5, random_state=None):
    """Artificially censor a proportion of samples."""
    if random_state is not None:
        np.random.seed(random_state)

    artificial_time = time.copy()
    artificial_event = event.copy()

    # Select samples to censor
    uncensored_idx = np.where(event == 1)[0]
    n_to_censor = int(len(uncensored_idx) * proportion)
    idx_to_censor = np.random.choice(uncensored_idx, size=n_to_censor, replace=False)

    # Censor at random time < original_time
    for idx in idx_to_censor:
        original_time = time[idx]
        if original_time > 0:
            artificial_time[idx] = np.random.randint(0, max(1, int(original_time)))
        artificial_event[idx] = 0

    return artificial_time, artificial_event


class SimpleOracle:
    """Oracle with probe depth constraint."""

    def __init__(self, true_time, true_event, probe_depth):
        self.true_time = true_time
        self.true_event = true_event
        self.probe_depth = probe_depth

    def query(self, indices, current_time, current_event):
        """Query oracle for selected indices."""
        updated_time = current_time.copy()
        updated_event = current_event.copy()

        for idx in indices:
            c = current_time[idx]
            max_observable = c + self.probe_depth

            if self.true_event[idx] == 1:  # True death
                if self.true_time[idx] <= max_observable:
                    # Reveal death
                    updated_time[idx] = self.true_time[idx]
                    updated_event[idx] = 1
                else:
                    # Extend censoring to probe limit
                    updated_time[idx] = max_observable
                    updated_event[idx] = 0
            else:
                # Extend censoring
                updated_time[idx] = min(self.true_time[idx], max_observable)
                updated_event[idx] = 0

        return updated_time, updated_event


def make_time_bins_simple(times, num_bins=None):
    """Create time bins for MTLR."""
    if num_bins is None:
        num_bins = min(15, int(np.sqrt(len(times))))

    bins = np.quantile(times[times > 0], np.linspace(0, 1, num_bins + 1))
    bins = np.unique(bins)
    return bins[1:]  # Exclude 0


def encode_survival(time, event, bins):
    """Encode survival data for MTLR."""
    n_samples = len(time)
    n_bins = len(bins)
    y = np.zeros((n_samples, n_bins))

    for i in range(n_samples):
        if event[i] == 1:
            # Death - one-hot at death time
            bin_idx = np.searchsorted(bins, time[i])
            if bin_idx < n_bins:
                y[i, bin_idx] = 1
        else:
            # Censored - all bins after censoring time
            bin_idx = np.searchsorted(bins, time[i])
            if bin_idx < n_bins:
                y[i, bin_idx:] = 1

    return y


def compute_c_index_simple(predictions, time, event):
    """Simple concordance index."""
    n = len(time)
    concordant = 0
    total = 0

    # Use mean survival time from predictions
    mean_survival = np.sum(predictions * np.arange(predictions.shape[1]), axis=1)

    for i in range(n):
        if event[i] == 0:
            continue
        for j in range(n):
            if time[j] > time[i]:
                total += 1
                if mean_survival[i] < mean_survival[j]:
                    concordant += 1
                elif mean_survival[i] == mean_survival[j]:
                    concordant += 0.5

    return concordant / total if total > 0 else 0.5


def train_model(model, X_train, y_train, config, epochs=50):
    """Train Bayesian MTLR model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.train()

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train)

    dataset_size = len(X_train)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Use sample_elbo for Bayesian training
        loss, log_prior, log_var_post, nll = model.sample_elbo(
            X_tensor, y_tensor, dataset_size, see1=0.01
        )

        loss.backward()
        optimizer.step()

    return model


def test_strategy(strategy_name, score_func, data, args, n_runs=3, probe_depth=3, batch_size=30):
    """Test a single strategy on NACD data."""

    results = []

    for run in range(n_runs):
        # Split data
        n_total = len(data)
        n_train = int(0.7 * n_total)
        n_test = n_total - n_train

        np.random.seed(42 + run)
        indices = np.random.permutation(n_total)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        # Get train/test split
        X_cols = [col for col in data.columns if col not in ['time', 'event']]
        X_train = data.iloc[train_idx][X_cols].values
        time_train = data.iloc[train_idx]['time'].values
        event_train = data.iloc[train_idx]['event'].values

        X_test = data.iloc[test_idx][X_cols].values
        time_test = data.iloc[test_idx]['time'].values
        event_test = data.iloc[test_idx]['event'].values

        # Artificially censor training data
        true_train_time = time_train.copy()
        true_train_event = event_train.copy()

        artificial_time, artificial_event = artificially_censor(
            time_train, event_train, proportion=0.5, random_state=42 + run
        )

        # Create time bins
        bins = make_time_bins_simple(time_train, num_bins=15)

        # Encode survival for MTLR
        y_train_init = encode_survival(artificial_time, artificial_event, bins)
        y_test = encode_survival(time_test, event_test, bins)

        # Train initial model
        args.n_time_bins = len(bins)
        model_init = BayesMtlr(in_features=X_train.shape[1], num_time_bins=len(bins), config=args)
        model_init = train_model(model_init, X_train, y_train_init, args, epochs=30)

        # Predict on test set
        model_init.eval()
        with torch.no_grad():
            test_preds_init = model_init(torch.FloatTensor(X_test), sample=True, n_samples=10)
            test_preds_init = torch.softmax(test_preds_init.mean(0), dim=1).numpy()

        initial_c_index = compute_c_index_simple(test_preds_init, time_test, event_test)

        # Get train predictions for acquisition
        model_init.eval()
        with torch.no_grad():
            # Get ensemble predictions
            train_preds_ensemble = []
            for _ in range(5):
                pred = model_init(torch.FloatTensor(X_train), sample=True, n_samples=1)
                pred = torch.softmax(pred.squeeze(0), dim=1).numpy()
                train_preds_ensemble.append(pred)
            train_preds = np.array(train_preds_ensemble)  # (K, N, T)

        # Compute scores with strategy
        scores = score_func(train_preds, artificial_time, artificial_event, X_train, bins)

        # Filter out already uncensored
        scores[artificial_event == 1] = -np.inf

        # Select top batch
        valid_scores = scores[scores > -np.inf]
        if len(valid_scores) < batch_size:
            batch_size_use = len(valid_scores)
        else:
            batch_size_use = batch_size

        if batch_size_use == 0:
            results.append({'improvement': 0.0, 'n_revealed': 0})
            continue

        selected = np.argsort(scores)[-batch_size_use:]

        # Query oracle
        oracle = SimpleOracle(true_train_time, true_train_event, probe_depth)
        updated_time, updated_event = oracle.query(selected, artificial_time, artificial_event)
        n_revealed = ((updated_event - artificial_event) > 0).sum()

        # Retrain with updated labels
        y_train_updated = encode_survival(updated_time, updated_event, bins)
        model_updated = BayesMtlr(in_features=X_train.shape[1], num_time_bins=len(bins), config=args)
        model_updated = train_model(model_updated, X_train, y_train_updated, args, epochs=30)

        # Evaluate
        model_updated.eval()
        with torch.no_grad():
            test_preds_updated = model_updated(torch.FloatTensor(X_test), sample=True, n_samples=10)
            test_preds_updated = torch.softmax(test_preds_updated.mean(0), dim=1).numpy()

        updated_c_index = compute_c_index_simple(test_preds_updated, time_test, event_test)

        results.append({
            'improvement': updated_c_index - initial_c_index,
            'n_revealed': n_revealed
        })

    return results


def main():
    print("\n" + "="*100)
    print("TESTING ON REAL NACD DATA")
    print("="*100 + "\n")

    # Load NACD data
    print("Loading NACD dataset...")
    try:
        data, args = Get_Dataset('NACD')
        print(f"Loaded NACD: {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
    except Exception as e:
        print(f"Error loading NACD: {e}")
        import traceback
        traceback.print_exc()
        return

    # Define simple strategies
    def entropy_strategy(preds, time, event, X, bins):
        """Standard entropy."""
        mean_preds = preds.mean(axis=0)
        entropy = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=1)
        return entropy

    def variance_strategy(preds, time, event, X, bins):
        """Variance across ensemble."""
        variance = preds.var(axis=0).mean(axis=1)
        return variance

    def random_strategy(preds, time, event, X, bins):
        """Random selection."""
        return np.random.rand(len(time))

    strategies = {
        'Random': random_strategy,
        'Entropy': entropy_strategy,
        'Variance': variance_strategy,
    }

    n_runs = 3
    print(f"\nTesting {len(strategies)} strategies with {n_runs} runs each...")
    print("Settings: probe_depth=3, batch_size=30\n")

    all_results = {}

    for strategy_name, score_func in strategies.items():
        print(f"  Testing {strategy_name}...", end=" ", flush=True)
        try:
            results = test_strategy(strategy_name, score_func, data, args, n_runs=n_runs)
            all_results[strategy_name] = results

            improvements = [r['improvement'] for r in results]
            mean_imp = np.mean(improvements)
            std_imp = np.std(improvements)
            wins = sum(1 for imp in improvements if imp > 0)

            print(f"Δ={mean_imp:+.4f}±{std_imp:.4f}, Wins={wins}/{n_runs}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Compare results
    print(f"\n{'='*100}")
    print("RESULTS")
    print(f"{'='*100}\n")

    print(f"{'Strategy':<20} {'Mean Δ':<15} {'Wins':<10} {'Avg Reveals':<15}")
    print("-" * 70)

    for strategy_name, results in all_results.items():
        improvements = [r['improvement'] for r in results]
        reveals = [r['n_revealed'] for r in results]
        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        wins = sum(1 for imp in improvements if imp > 0)
        avg_reveals = np.mean(reveals)

        print(f"{strategy_name:<20} {mean_imp:+.4f}±{std_imp:.4f}   {wins}/{n_runs}      {avg_reveals:.1f}")

    print(f"\n{'='*100}\n")


if __name__ == '__main__':
    main()
