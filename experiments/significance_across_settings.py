"""
Test statistical significance across different probe depths and batch sizes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel

from src.acquisition.optimal_batchbald import OptimalBatchBALD
from src.oracle.oracle import Oracle
from src.models.survival_model import BayesianSurvivalModel
from src.evaluation.metrics import concordance_index
from src.data.synthetic import artificially_censor


def load_nacd_data():
    """Load NACD dataset."""
    df = pd.read_csv('data/MIMIC/NACD/NACD_Full.csv')
    df = df.rename(columns={'SURVIVAL': 'time', 'CENSORED': 'event'})
    return df


def split_data(df, train_frac=0.7, random_state=42):
    """Split into train/test."""
    np.random.seed(random_state)
    n = len(df)
    indices = np.random.permutation(n)
    n_train = int(n * train_frac)
    return df.iloc[indices[:n_train]].reset_index(drop=True), \
           df.iloc[indices[n_train:]].reset_index(drop=True)


def discretize_times(time_train, time_test, n_bins=20):
    """Discretize continuous times into bins."""
    all_times = np.concatenate([time_train, time_test])
    unique_times = np.unique(all_times[all_times > 0])

    if len(unique_times) > n_bins:
        bin_edges = np.quantile(unique_times, np.linspace(0, 1, n_bins + 1))
    else:
        bin_edges = np.concatenate([[0], unique_times, [unique_times[-1] + 1]])
        n_bins = len(unique_times)

    time_train_binned = np.digitize(time_train, bin_edges[1:])
    time_test_binned = np.digitize(time_test, bin_edges[1:])

    time_train_binned = np.clip(time_train_binned, 0, n_bins - 1)
    time_test_binned = np.clip(time_test_binned, 0, n_bins - 1)

    return time_train_binned, time_test_binned, n_bins


def run_single_trial(
    data,
    strategy_name,
    acq_func,
    probe_depth,
    batch_size,
    random_seed
):
    """Run a single trial."""
    # Split data
    train_df, test_df = split_data(data, random_state=random_seed)

    # Prepare features
    feature_cols = [col for col in train_df.columns if col not in ['time', 'event']]
    X_train = train_df[feature_cols].values.astype(np.float32)
    time_train = train_df['time'].values
    event_train = train_df['event'].values

    X_test = test_df[feature_cols].values.astype(np.float32)
    time_test = test_df['time'].values
    event_test = test_df['event'].values

    # Discretize times
    time_train_binned, time_test_binned, n_bins = discretize_times(
        time_train, time_test, n_bins=20
    )

    # Store ground truth
    true_train_time = time_train_binned.copy()
    true_train_event = event_train.copy()

    # Artificially censor
    artificial_time, artificial_event = artificially_censor(
        time_train_binned, event_train, proportion=0.5, random_state=random_seed
    )

    # Train initial model
    model_init = BayesianSurvivalModel(
        n_features=X_train.shape[1],
        n_time_bins=n_bins,
        n_ensemble=5,
        hidden_size=64
    )
    model_init.fit(X_train, artificial_time, artificial_event, epochs=20, batch_size=32, verbose=False)

    # Initial test performance
    test_preds_init = model_init.predict_proba(X_test)
    initial_c_index = concordance_index(test_preds_init, time_test_binned, event_test)

    # Get ensemble predictions
    train_preds = model_init.predict_proba(X_train)

    # Oracle setup
    oracle = Oracle(true_train_time, true_train_event, probe_depth)
    oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
        train_preds, artificial_time, artificial_event
    )

    # Select batch based on strategy
    if strategy_name == 'Entropy':
        mean_preds = train_preds.mean(axis=0)
        scores = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=1)
        scores[artificial_event == 1] = -np.inf
        selected = np.argsort(scores)[-batch_size:]
        valid = scores[selected] > -np.inf
        selected = selected[valid]
    else:
        # OptimalBatchBALD
        selected = acq_func.select_batch(
            oracle_probs,
            train_preds,
            artificial_time,
            batch_size,
            artificial_event,
            greedy=False
        )

    if len(selected) == 0:
        return {'improvement': 0.0, 'n_revealed': 0}

    # Query oracle
    updated_time, updated_event = oracle.query(selected, artificial_time, artificial_event)
    n_revealed = ((updated_event - artificial_event) > 0).sum()

    # Retrain
    model_updated = BayesianSurvivalModel(
        n_features=X_train.shape[1],
        n_time_bins=n_bins,
        n_ensemble=5,
        hidden_size=64
    )
    model_updated.fit(X_train, updated_time, updated_event, epochs=20, batch_size=32, verbose=False)

    # Evaluate
    test_preds_updated = model_updated.predict_proba(X_test)
    updated_c_index = concordance_index(test_preds_updated, time_test_binned, event_test)
    improvement = updated_c_index - initial_c_index

    return {
        'improvement': improvement,
        'n_revealed': n_revealed,
        'initial_c': initial_c_index,
        'final_c': updated_c_index
    }


def test_setting(data, probe_depth, batch_size, n_runs=10):
    """Test a single setting."""
    print(f"\n  Testing probe_depth={probe_depth}, batch_size={batch_size}...", flush=True)

    optimal_acq = OptimalBatchBALD(probe_depth=probe_depth, strategy='adaptive_fusion')

    optimal_results = []
    entropy_results = []

    for run in range(n_runs):
        # Run Optimal
        result_opt = run_single_trial(
            data, 'Optimal', optimal_acq, probe_depth, batch_size, 42 + run
        )
        optimal_results.append(result_opt)

        # Run Entropy
        result_ent = run_single_trial(
            data, 'Entropy', None, probe_depth, batch_size, 42 + run
        )
        entropy_results.append(result_ent)

        print(f"    Run {run+1}/{n_runs}: Optimal Δ={result_opt['improvement']:+.4f}, Entropy Δ={result_ent['improvement']:+.4f}", flush=True)

    # Compute statistics
    optimal_imps = [r['improvement'] for r in optimal_results]
    entropy_imps = [r['improvement'] for r in entropy_results]

    mean_opt = np.mean(optimal_imps)
    mean_ent = np.mean(entropy_imps)
    mean_diff = mean_opt - mean_ent

    # Statistical test
    t_stat, p_val = ttest_rel(optimal_imps, entropy_imps)

    significant = "✓ YES" if p_val < 0.05 else "  No"

    return {
        'probe_depth': probe_depth,
        'batch_size': batch_size,
        'optimal_mean': mean_opt,
        'entropy_mean': mean_ent,
        'difference': mean_diff,
        'p_value': p_val,
        'significant': p_val < 0.05,
        'optimal_wins': sum(1 for imp in optimal_imps if imp > 0),
        'entropy_wins': sum(1 for imp in entropy_imps if imp > 0),
        'n_runs': n_runs
    }


def main():
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE ACROSS SETTINGS")
    print("="*80 + "\n")

    # Load data
    print("Loading NACD dataset...")
    data = load_nacd_data()
    print(f"Loaded: {data.shape}, Event rate: {data['event'].mean():.2%}\n")

    # Test different settings
    settings = [
        # Vary probe depth (keep batch=30)
        (2, 30),
        (3, 30),
        (5, 30),

        # Vary batch size (keep probe=3)
        (3, 20),
        (3, 40),
        (3, 50),
    ]

    n_runs = 10

    print(f"Testing {len(settings)} different settings with {n_runs} runs each")
    print("="*80)

    results = []

    for probe_depth, batch_size in settings:
        result = test_setting(data, probe_depth, batch_size, n_runs=n_runs)
        results.append(result)

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: Statistical Significance Across Settings")
    print("="*80 + "\n")

    print(f"{'Probe':<7} {'Batch':<7} {'Optimal Δ':<12} {'Entropy Δ':<12} {'Difference':<12} {'p-value':<10} {'Sig?'}")
    print("-" * 80)

    for r in results:
        sig_marker = "✓" if r['significant'] else " "
        print(f"{r['probe_depth']:<7} {r['batch_size']:<7} "
              f"{r['optimal_mean']:+.4f}      {r['entropy_mean']:+.4f}      "
              f"{r['difference']:+.4f}      {r['p_value']:.4f}     {sig_marker}")

    # Check if any are significant
    n_significant = sum(1 for r in results if r['significant'])

    print("\n" + "="*80)
    print(f"SIGNIFICANT RESULTS: {n_significant}/{len(settings)} settings")
    print("="*80 + "\n")

    if n_significant > 0:
        print("✓ Found statistically significant improvements in:")
        for r in results:
            if r['significant']:
                print(f"  - Probe depth={r['probe_depth']}, Batch size={r['batch_size']}: "
                      f"Δ={r['difference']:+.4f}, p={r['p_value']:.4f}")
    else:
        print("✗ No statistically significant improvements found (all p > 0.05)")
        print("\nBest setting:")
        best = max(results, key=lambda x: x['difference'])
        print(f"  - Probe depth={best['probe_depth']}, Batch size={best['batch_size']}: "
              f"Δ={best['difference']:+.4f}, p={best['p_value']:.4f}")

    print("\n" + "="*80 + "\n")

    return results


if __name__ == '__main__':
    results = main()
