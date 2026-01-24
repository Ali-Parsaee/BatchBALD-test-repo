"""
ULTIMATE SHOWDOWN: Test all BatchBALD variants against entropy/variance.

This is the definitive test to see which method wins.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel

# Import acquisition functions
from src.acquisition.batchbald import SurvivalBatchBALD
from src.acquisition.improved_batchbald import ImprovedBatchBALD
from src.acquisition.weighted_batchbald import WeightedBatchBALD
from src.acquisition.optimal_batchbald import OptimalBatchBALD, UltraAggressiveBatchBALD

# Import oracle and model
from src.oracle.oracle import Oracle
from src.models.survival_model import BayesianSurvivalModel
from src.evaluation.metrics import concordance_index
from src.data.synthetic import artificially_censor


def load_nacd_data():
    """Load NACD dataset."""
    df = pd.read_csv('data/MIMIC/NACD/NACD_Full.csv')
    df = df.rename(columns={'SURVIVAL': 'time', 'CENSORED': 'event'})
    print(f"Loaded NACD: {df.shape}")
    print(f"Event rate: {df['event'].mean():.2%}")
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


def test_strategy(
    strategy_name,
    acq_func,
    data,
    probe_depth=3,
    batch_size=30,
    n_runs=5,
    verbose=True
):
    """Test a single acquisition strategy."""
    results = []

    for run in range(n_runs):
        if verbose:
            print(f"    Run {run+1}/{n_runs}...", end=" ", flush=True)

        # Split data
        train_df, test_df = split_data(data, random_state=42 + run)

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
            time_train_binned, event_train, proportion=0.5, random_state=42 + run
        )

        # Train initial model
        model_init = BayesianSurvivalModel(
            n_features=X_train.shape[1],
            n_time_bins=n_bins,
            n_ensemble=5,
            hidden_size=64
        )
        model_init.fit(X_train, artificial_time, artificial_event, epochs=30, batch_size=32, verbose=False)

        # Initial test performance
        test_preds_init = model_init.predict_proba(X_test)
        initial_c_index = concordance_index(test_preds_init, time_test_binned, event_test)

        # Get ensemble predictions
        train_preds = model_init.predict_proba(X_train)  # (K, N, T)

        # Oracle setup
        oracle = Oracle(true_train_time, true_train_event, probe_depth)
        oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
            train_preds, artificial_time, artificial_event
        )

        # Select batch based on strategy type
        if strategy_name == 'Random':
            # Random selection
            censored_mask = (artificial_event == 0)
            censored_indices = np.where(censored_mask)[0]
            if len(censored_indices) >= batch_size:
                selected = np.random.choice(censored_indices, size=batch_size, replace=False)
            else:
                selected = censored_indices

        elif strategy_name in ['Entropy', 'Variance']:
            # Simple baselines
            if strategy_name == 'Entropy':
                mean_preds = train_preds.mean(axis=0)
                scores = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=1)
            else:  # Variance
                scores = train_preds.var(axis=0).sum(axis=1)

            scores[artificial_event == 1] = -np.inf
            selected = np.argsort(scores)[-batch_size:]
            valid = scores[selected] > -np.inf
            selected = selected[valid]

        else:
            # BatchBALD variants
            if isinstance(acq_func, (ImprovedBatchBALD, OptimalBatchBALD, UltraAggressiveBatchBALD)):
                selected = acq_func.select_batch(
                    oracle_probs,
                    train_preds,
                    artificial_time,
                    batch_size,
                    artificial_event,
                    greedy=False  # Use top-k for speed
                )
            else:
                # Plain BatchBALD or Weighted BatchBALD
                selected = acq_func.select_batch(
                    oracle_probs,
                    batch_size,
                    artificial_event
                )

        if len(selected) == 0:
            if verbose:
                print("No valid samples!")
            results.append({'improvement': 0.0, 'n_revealed': 0})
            continue

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
        model_updated.fit(X_train, updated_time, updated_event, epochs=30, batch_size=32, verbose=False)

        # Evaluate
        test_preds_updated = model_updated.predict_proba(X_test)
        updated_c_index = concordance_index(test_preds_updated, time_test_binned, event_test)
        improvement = updated_c_index - initial_c_index

        if verbose:
            print(f"C: {initial_c_index:.4f} → {updated_c_index:.4f} (Δ={improvement:+.4f}, reveals={n_revealed})")

        results.append({
            'improvement': improvement,
            'n_revealed': n_revealed,
            'initial_c': initial_c_index,
            'final_c': updated_c_index
        })

    return results


def main():
    print("\n" + "="*100)
    print("ULTIMATE SHOWDOWN: Which BatchBALD Variant Wins?")
    print("="*100 + "\n")

    # Load data
    print("Loading NACD dataset...")
    data = load_nacd_data()

    # Define all strategies
    probe_depth = 3

    strategies = {
        # Baselines
        'Random': None,
        'Entropy': None,
        'Variance': None,

        # BatchBALD variants
        'BatchBALD (Plain MI)': SurvivalBatchBALD(probe_depth=probe_depth),
        'BatchBALD (Observable Mass)': ImprovedBatchBALD(probe_depth=probe_depth, variant='observable_mass'),
        'BatchBALD (MI × P(death))': ImprovedBatchBALD(probe_depth=probe_depth, variant='mi_times_pdeath'),
        'BatchBALD (Combined)': ImprovedBatchBALD(probe_depth=probe_depth, variant='combined'),
        'Weighted BatchBALD': WeightedBatchBALD(probe_depth=probe_depth, death_weight=1.0, censor_weight=0.3),

        # NEW: Optimal variants
        'Optimal (Adaptive Fusion)': OptimalBatchBALD(probe_depth=probe_depth, strategy='adaptive_fusion', alpha=1.0, beta=0.5, gamma=0.3),
        'Optimal (Obs Mass Boosted)': OptimalBatchBALD(probe_depth=probe_depth, strategy='observable_mass_boosted', alpha=1.0),
        'Optimal (Ensemble Disagree)': OptimalBatchBALD(probe_depth=probe_depth, strategy='ensemble_disagreement'),
        'Optimal (Multi-Factor)': OptimalBatchBALD(probe_depth=probe_depth, strategy='multi_factor', beta=0.5, gamma=0.5),
        'Ultra-Aggressive': UltraAggressiveBatchBALD(probe_depth=probe_depth),
    }

    n_runs = 5
    batch_size = 30

    print(f"\nTesting {len(strategies)} strategies")
    print(f"Settings: probe_depth={probe_depth}, batch_size={batch_size}, n_runs={n_runs}\n")
    print("="*100)

    all_results = {}

    for strategy_name, acq_func in strategies.items():
        print(f"\n{strategy_name}:")
        try:
            results = test_strategy(
                strategy_name,
                acq_func,
                data,
                probe_depth=probe_depth,
                batch_size=batch_size,
                n_runs=n_runs,
                verbose=True
            )
            all_results[strategy_name] = results
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ===========================================================================================
    # RESULTS
    # ===========================================================================================

    print("\n" + "="*100)
    print("FINAL RESULTS")
    print("="*100 + "\n")

    # Compute statistics
    stats = []
    for strategy_name, results in all_results.items():
        improvements = [r['improvement'] for r in results]
        reveals = [r['n_revealed'] for r in results]

        stats.append({
            'Strategy': strategy_name,
            'Mean Δ': np.mean(improvements),
            'Std Δ': np.std(improvements),
            'Median Δ': np.median(improvements),
            'Wins': sum(1 for imp in improvements if imp > 0),
            'Total': len(improvements),
            'Avg Reveals': np.mean(reveals),
            'Improvements': improvements
        })

    # Sort by mean improvement
    stats.sort(key=lambda x: x['Mean Δ'], reverse=True)

    # Print table
    print(f"{'Rank':<5} {'Strategy':<35} {'Mean Δ':<15} {'Median Δ':<12} {'Wins':<10} {'Reveals':<10}")
    print("-" * 100)

    for rank, s in enumerate(stats, 1):
        marker = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        print(f"{marker} {rank:<3} {s['Strategy']:<35} {s['Mean Δ']:+.4f}±{s['Std Δ']:.4f}   {s['Median Δ']:+.4f}    {s['Wins']}/{s['Total']}      {s['Avg Reveals']:.1f}")

    # Statistical significance tests
    print("\n" + "="*100)
    print("STATISTICAL SIGNIFICANCE (vs Entropy)")
    print("="*100 + "\n")

    if 'Entropy' in all_results:
        entropy_improvements = [r['improvement'] for r in all_results['Entropy']]

        print(f"{'Strategy':<35} {'Mean Δ Diff':<15} {'p-value':<12} {'Significant?'}")
        print("-" * 70)

        for s in stats:
            if s['Strategy'] == 'Entropy':
                continue

            try:
                t_stat, p_val = ttest_rel(s['Improvements'], entropy_improvements)
                mean_diff = s['Mean Δ'] - np.mean(entropy_improvements)
                sig = "✓ YES" if p_val < 0.05 else "  No"
                print(f"{s['Strategy']:<35} {mean_diff:+.4f}          {p_val:.4f}      {sig}")
            except:
                print(f"{s['Strategy']:<35} {'N/A':<15} {'N/A':<12} {'N/A'}")

    # Winner announcement
    print("\n" + "="*100)
    winner = stats[0]
    print(f"🎉 WINNER: {winner['Strategy']}")
    print(f"   Mean improvement: {winner['Mean Δ']:+.4f} ± {winner['Std Δ']:.4f}")
    print(f"   Win rate: {winner['Wins']}/{winner['Total']} ({100*winner['Wins']/winner['Total']:.0f}%)")
    print(f"   Average reveals: {winner['Avg Reveals']:.1f}")
    print("="*100 + "\n")

    return all_results, stats


if __name__ == '__main__':
    results, stats = main()
