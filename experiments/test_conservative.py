"""
Test Conservative BatchBALD methods for statistical significance.

Focus: Reduce variance, increase reliability, achieve significance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel

from src.acquisition.conservative_batchbald import ConservativeBatchBALD, DataDrivenBatchBALD
from src.acquisition.optimal_batchbald import OptimalBatchBALD
from src.oracle.oracle import Oracle
from src.models.survival_model import BayesianSurvivalModel
from src.evaluation.metrics import concordance_index
from src.data.synthetic import artificially_censor


def load_nacd_data():
    df = pd.read_csv('data/MIMIC/NACD/NACD_Full.csv')
    df = df.rename(columns={'SURVIVAL': 'time', 'CENSORED': 'event'})
    return df


def split_data(df, train_frac=0.7, random_state=42):
    np.random.seed(random_state)
    n = len(df)
    indices = np.random.permutation(n)
    n_train = int(n * train_frac)
    return df.iloc[indices[:n_train]].reset_index(drop=True), \
           df.iloc[indices[n_train:]].reset_index(drop=True)


def discretize_times(time_train, time_test, n_bins=20):
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


def test_method(method_name, acq_func, data, probe_depth, batch_size, n_runs, random_seeds):
    """Test a single method."""
    results = []

    for run_idx, seed in enumerate(random_seeds):
        # Split
        train_df, test_df = split_data(data, random_state=seed)

        # Features
        feature_cols = [col for col in train_df.columns if col not in ['time', 'event']]
        X_train = train_df[feature_cols].values.astype(np.float32)
        time_train = train_df['time'].values
        event_train = train_df['event'].values

        X_test = test_df[feature_cols].values.astype(np.float32)
        time_test = test_df['time'].values
        event_test = test_df['event'].values

        # Discretize
        time_train_binned, time_test_binned, n_bins = discretize_times(time_train, time_test, n_bins=20)

        # Ground truth
        true_train_time = time_train_binned.copy()
        true_train_event = event_train.copy()

        # Artificially censor
        artificial_time, artificial_event = artificially_censor(
            time_train_binned, event_train, proportion=0.5, random_state=seed
        )

        # Train initial model
        model_init = BayesianSurvivalModel(
            n_features=X_train.shape[1],
            n_time_bins=n_bins,
            n_ensemble=5,
            hidden_size=64
        )
        model_init.fit(X_train, artificial_time, artificial_event, epochs=20, batch_size=32, verbose=False)

        # Initial test
        test_preds_init = model_init.predict_proba(X_test)
        initial_c_index = concordance_index(test_preds_init, time_test_binned, event_test)

        # Predictions
        train_preds = model_init.predict_proba(X_train)

        # Oracle
        oracle = Oracle(true_train_time, true_train_event, probe_depth)
        oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
            train_preds, artificial_time, artificial_event
        )

        # Select batch
        if method_name == 'Entropy (Baseline)':
            mean_preds = train_preds.mean(axis=0)
            scores = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=1)
            scores[artificial_event == 1] = -np.inf
            selected = np.argsort(scores)[-batch_size:]
            valid = scores[selected] > -np.inf
            selected = selected[valid]
        else:
            selected = acq_func.select_batch(
                oracle_probs,
                train_preds,
                artificial_time,
                batch_size,
                artificial_event,
                greedy=False
            )

        if len(selected) == 0:
            results.append({'improvement': 0.0, 'n_revealed': 0})
            continue

        # Query
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

        results.append({
            'improvement': improvement,
            'n_revealed': n_revealed,
            'seed': seed
        })

    return results


def main():
    print("\n" + "="*80)
    print("CONSERVATIVE BATCHBALD: Testing for Statistical Significance")
    print("="*80 + "\n")

    # Load data
    data = load_nacd_data()
    print(f"Loaded NACD: {data.shape}, Event rate: {data['event'].mean():.2%}\n")

    # Test settings
    probe_depth = 3
    batch_size = 30
    n_runs = 20  # More runs for better statistical power

    # Generate different random seeds for diversity
    random_seeds = [42 + i * 7 for i in range(n_runs)]

    print(f"Settings: probe_depth={probe_depth}, batch_size={batch_size}, n_runs={n_runs}")
    print(f"Random seeds: {random_seeds[:5]}... (showing first 5)\n")
    print("="*80 + "\n")

    # Methods to test
    methods = {
        'Entropy (Baseline)': None,
        'Conservative (min_p_death=0.15)': ConservativeBatchBALD(probe_depth, min_p_death=0.15),
        'Conservative (min_p_death=0.20)': ConservativeBatchBALD(probe_depth, min_p_death=0.20),
        'Conservative (min_p_death=0.10)': ConservativeBatchBALD(probe_depth, min_p_death=0.10),
        'Data-Driven': DataDrivenBatchBALD(probe_depth),
        'Optimal (reference)': OptimalBatchBALD(probe_depth, strategy='adaptive_fusion'),
    }

    all_results = {}

    for method_name, acq_func in methods.items():
        print(f"{method_name}:")
        print(f"  Running {n_runs} trials...", end=" ", flush=True)

        results = test_method(
            method_name, acq_func, data, probe_depth, batch_size, n_runs, random_seeds
        )

        all_results[method_name] = results

        improvements = [r['improvement'] for r in results]
        reveals = [r['n_revealed'] for r in results]

        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        median_imp = np.median(improvements)
        wins = sum(1 for imp in improvements if imp > 0)

        print(f"Done!")
        print(f"    Mean Δ: {mean_imp:+.4f} ± {std_imp:.4f}")
        print(f"    Median Δ: {median_imp:+.4f}")
        print(f"    Wins: {wins}/{n_runs} ({100*wins/n_runs:.0f}%)")
        print(f"    Avg reveals: {np.mean(reveals):.1f}\n")

    # Statistical comparison
    print("="*80)
    print("STATISTICAL SIGNIFICANCE vs Entropy")
    print("="*80 + "\n")

    entropy_imps = [r['improvement'] for r in all_results['Entropy (Baseline)']]

    print(f"{'Method':<40} {'Mean Δ':<12} {'vs Entropy':<12} {'p-value':<10} {'Sig?'}")
    print("-" * 90)

    significant_methods = []

    for method_name in methods.keys():
        if method_name == 'Entropy (Baseline)':
            continue

        imps = [r['improvement'] for r in all_results[method_name]]
        mean_imp = np.mean(imps)
        diff = mean_imp - np.mean(entropy_imps)

        t_stat, p_val = ttest_rel(imps, entropy_imps)

        sig_marker = "✓ YES" if p_val < 0.05 else ("~ CLOSE" if p_val < 0.10 else "  No")

        if p_val < 0.05:
            significant_methods.append(method_name)

        print(f"{method_name:<40} {mean_imp:+.4f}      {diff:+.4f}      {p_val:.4f}     {sig_marker}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80 + "\n")

    if len(significant_methods) > 0:
        print(f"✓ Found {len(significant_methods)} statistically significant method(s):")
        for method in significant_methods:
            imps = [r['improvement'] for r in all_results[method]]
            mean_imp = np.mean(imps)
            std_imp = np.std(improvements)
            p_val = ttest_rel(imps, entropy_imps)[1]
            print(f"  - {method}: Δ={mean_imp:+.4f}±{std_imp:.4f}, p={p_val:.4f}")
    else:
        print("✗ No methods achieved statistical significance (p < 0.05)")

        # Find closest
        best_p = min(
            ttest_rel([r['improvement'] for r in all_results[m]], entropy_imps)[1]
            for m in methods if m != 'Entropy (Baseline)'
        )
        best_method = [
            m for m in methods if m != 'Entropy (Baseline)'
            and ttest_rel([r['improvement'] for r in all_results[m]], entropy_imps)[1] == best_p
        ][0]

        print(f"\nClosest to significance: {best_method} (p={best_p:.4f})")

    print("\n" + "="*80 + "\n")

    return all_results


if __name__ == '__main__':
    results = main()
