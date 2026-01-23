"""
Simplified test on NACD real data - test what strategies work in practice.

Loads NACD CSV directly without complex dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel

# Our own oracle and model
from src.oracle.oracle import Oracle
from src.models.survival_model import BayesianSurvivalModel
from src.evaluation.metrics import concordance_index
from src.data.synthetic import artificially_censor


def load_nacd_simple():
    """Load NACD dataset directly."""
    df = pd.read_csv('data/MIMIC/NACD/NACD_Full.csv')

    # Rename columns
    df = df.rename(columns={'SURVIVAL': 'time', 'CENSORED': 'event'})

    print(f"Loaded NACD: {df.shape}")
    print(f"Event rate: {df['event'].mean():.2%}")
    print(f"Time range: [{df['time'].min():.1f}, {df['time'].max():.1f}]")

    return df


def split_data(df, train_frac=0.7, random_state=42):
    """Split data into train/test."""
    np.random.seed(random_state)
    n = len(df)
    indices = np.random.permutation(n)

    n_train = int(n * train_frac)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def test_strategy(strategy_name, score_func, data, probe_depth=3, batch_size=30, n_runs=3):
    """Test single strategy on NACD."""

    results = []

    for run in range(n_runs):
        # Split data
        train_df, test_df = split_data(data, train_frac=0.7, random_state=42 + run)

        # Prepare features and labels
        feature_cols = [col for col in train_df.columns if col not in ['time', 'event']]
        X_train = train_df[feature_cols].values.astype(np.float32)
        time_train = train_df['time'].values
        event_train = train_df['event'].values

        X_test = test_df[feature_cols].values.astype(np.float32)
        time_test = test_df['time'].values
        event_test = test_df['event'].values

        # Determine number of time bins
        n_time_bins = min(20, int(np.sqrt(len(time_train))))

        # Discretize times into bins
        all_times = np.concatenate([time_train, time_test])
        unique_times = np.unique(all_times[all_times > 0])
        if len(unique_times) > n_time_bins:
            bin_edges = np.quantile(unique_times, np.linspace(0, 1, n_time_bins + 1))
        else:
            bin_edges = np.concatenate([[0], unique_times, [unique_times[-1] + 1]])
            n_time_bins = len(unique_times)

        # Map continuous times to discrete bins
        time_train_binned = np.digitize(time_train, bin_edges[1:])
        time_test_binned = np.digitize(time_test, bin_edges[1:])

        # Clip to valid range
        time_train_binned = np.clip(time_train_binned, 0, n_time_bins - 1)
        time_test_binned = np.clip(time_test_binned, 0, n_time_bins - 1)

        # Store true labels
        true_train_time = time_train_binned.copy()
        true_train_event = event_train.copy()

        # Artificially censor
        artificial_time, artificial_event = artificially_censor(
            time_train_binned, event_train, proportion=0.5, random_state=42 + run
        )

        print(f"\n    Run {run+1}: Train={len(X_train)}, Test={len(X_test)}, Features={X_train.shape[1]}, Bins={n_time_bins}")

        # Train initial model
        model_init = BayesianSurvivalModel(
            n_features=X_train.shape[1],
            n_time_bins=n_time_bins,
            n_ensemble=5,
            hidden_size=64
        )
        model_init.fit(X_train, artificial_time, artificial_event, epochs=30, batch_size=32, verbose=False)

        # Initial test performance
        test_preds_init = model_init.predict_proba(X_test)
        initial_c_index = concordance_index(test_preds_init, time_test_binned, event_test)
        print(f"      Initial C-index: {initial_c_index:.4f}")

        # Get ensemble predictions for acquisition
        train_preds = model_init.predict_proba(X_train)  # (K, N, T)

        # Oracle
        oracle = Oracle(true_train_time, true_train_event, probe_depth)
        oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
            train_preds, artificial_time, artificial_event
        )

        # Compute scores
        scores = score_func(train_preds, oracle_probs, artificial_time, artificial_event, X_train)

        # Filter already uncensored
        scores[artificial_event == 1] = -np.inf

        # Select batch
        valid_mask = scores > -np.inf
        n_valid = valid_mask.sum()

        if n_valid < batch_size:
            batch_size_use = n_valid
        else:
            batch_size_use = batch_size

        if batch_size_use == 0:
            print(f"      No valid samples to query!")
            results.append({'improvement': 0.0, 'n_revealed': 0})
            continue

        selected = np.argsort(scores)[-batch_size_use:]

        # Query oracle
        updated_time, updated_event = oracle.query(selected, artificial_time, artificial_event)
        n_revealed = ((updated_event - artificial_event) > 0).sum()
        print(f"      Selected {batch_size_use} samples, revealed {n_revealed} events")

        # Retrain
        model_updated = BayesianSurvivalModel(
            n_features=X_train.shape[1],
            n_time_bins=n_time_bins,
            n_ensemble=5,
            hidden_size=64
        )
        model_updated.fit(X_train, updated_time, updated_event, epochs=30, batch_size=32, verbose=False)

        # Evaluate
        test_preds_updated = model_updated.predict_proba(X_test)
        updated_c_index = concordance_index(test_preds_updated, time_test_binned, event_test)
        improvement = updated_c_index - initial_c_index

        print(f"      Updated C-index: {updated_c_index:.4f} (Δ={improvement:+.4f})")

        results.append({
            'improvement': improvement,
            'n_revealed': n_revealed
        })

    return results


def main():
    print("\n" + "="*100)
    print("TESTING ON REAL NACD DATA - What Strategies Work?")
    print("="*100 + "\n")

    # Load data
    data = load_nacd_simple()

    # Define strategies
    def entropy_strategy(preds, oracle_probs, time, event, X):
        """Standard entropy on predictions."""
        mean_preds = preds.mean(axis=0)
        entropy = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=1)
        return entropy

    def variance_strategy(preds, oracle_probs, time, event, X):
        """Variance across ensemble."""
        variance = preds.var(axis=0).sum(axis=1)
        return variance

    def batchbald_mi_strategy(preds, oracle_probs, time, event, X):
        """BatchBALD mutual information."""
        mean_probs = oracle_probs.mean(axis=0)
        H_expected = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

        H_conditional = -np.sum(oracle_probs * np.log(oracle_probs + 1e-10), axis=2)
        E_H_conditional = H_conditional.mean(axis=0)

        return H_expected - E_H_conditional

    def random_strategy(preds, oracle_probs, time, event, X):
        """Random selection."""
        return np.random.rand(len(time))

    strategies = {
        'Random': random_strategy,
        'Entropy': entropy_strategy,
        'Variance': variance_strategy,
        'BatchBALD MI': batchbald_mi_strategy,
    }

    n_runs = 3
    probe_depth = 3
    batch_size = 30

    print(f"\nTesting {len(strategies)} strategies:")
    print(f"  Runs: {n_runs}")
    print(f"  Probe depth: {probe_depth}")
    print(f"  Batch size: {batch_size}")

    all_results = {}

    for strategy_name, score_func in strategies.items():
        print(f"\n{'='*100}")
        print(f"Testing: {strategy_name}")
        print(f"{'='*100}")

        try:
            results = test_strategy(strategy_name, score_func, data, probe_depth, batch_size, n_runs)
            all_results[strategy_name] = results

            improvements = [r['improvement'] for r in results]
            mean_imp = np.mean(improvements)
            std_imp = np.std(improvements)
            wins = sum(1 for imp in improvements if imp > 0)

            print(f"\n  Summary: Δ={mean_imp:+.4f}±{std_imp:.4f}, Wins={wins}/{n_runs}")

        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Final comparison
    print(f"\n\n{'='*100}")
    print("FINAL RESULTS")
    print(f"{'='*100}\n")

    print(f"{'Strategy':<20} {'Mean Δ':<18} {'Wins':<10} {'Avg Reveals':<15}")
    print("-" * 70)

    ranked = []
    for strategy_name, results in all_results.items():
        improvements = [r['improvement'] for r in results]
        reveals = [r['n_revealed'] for r in results]
        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        wins = sum(1 for imp in improvements if imp > 0)
        avg_reveals = np.mean(reveals)

        ranked.append({
            'name': strategy_name,
            'mean': mean_imp,
            'std': std_imp,
            'wins': wins,
            'reveals': avg_reveals
        })

    # Sort by mean improvement
    ranked.sort(key=lambda x: x['mean'], reverse=True)

    for rank, r in enumerate(ranked, 1):
        marker = "🏆" if rank == 1 else " "
        print(f"{marker} {r['name']:<20} {r['mean']:+.4f}±{r['std']:.4f}     {r['wins']}/{n_runs}      {r['reveals']:.1f}")

    # Statistical test
    if 'Entropy' in all_results:
        print(f"\n{'='*100}")
        print("Statistical Tests vs Entropy Baseline")
        print(f"{'='*100}\n")

        entropy_imps = [r['improvement'] for r in all_results['Entropy']]

        for strategy_name, results in all_results.items():
            if strategy_name == 'Entropy':
                continue

            strategy_imps = [r['improvement'] for r in results]
            t_stat, p_value = ttest_rel(strategy_imps, entropy_imps)

            diff = np.mean(strategy_imps) - np.mean(entropy_imps)
            sig = "✓" if p_value < 0.05 else " "

            print(f"{sig} {strategy_name:<20} Δ={diff:+.4f}, p={p_value:.4f}")

    print(f"\n{'='*100}\n")


if __name__ == '__main__':
    main()
