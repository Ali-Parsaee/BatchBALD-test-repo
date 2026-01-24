"""
Deep Analysis: What Actually Predicts AL Success?

Analyze successful vs unsuccessful runs to understand what really matters.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

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


def analyze_single_run(data, probe_depth, batch_size, random_seed):
    """
    Run one trial and extract detailed analytics about what led to success/failure.
    """
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

    # Discretize
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

    # Oracle
    oracle = Oracle(true_train_time, true_train_event, probe_depth)
    oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
        train_preds, artificial_time, artificial_event
    )

    # Compute various features for ALL censored instances
    censored_mask = (artificial_event == 0)
    censored_indices = np.where(censored_mask)[0]
    n_censored = len(censored_indices)

    features = {}

    # 1. Observable mass
    obs_mass = np.zeros(n_censored)
    for i, idx in enumerate(censored_indices):
        c = artificial_time[idx]
        max_obs = min(c + probe_depth, n_bins - 1)
        if c + 1 <= max_obs:
            obs_mass[i] = train_preds.mean(0)[idx, c+1:max_obs+1].sum()
    features['obs_mass'] = obs_mass

    # 2. Mutual Information
    mi = np.zeros(n_censored)
    for i, idx in enumerate(censored_indices):
        # H(E[Y])
        mean_probs = oracle_probs[:, idx, :].mean(0)
        H_exp = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        # E[H(Y|theta)]
        H_cond = -np.sum(oracle_probs[:, idx, :] * np.log(oracle_probs[:, idx, :] + 1e-10), axis=1).mean()
        mi[i] = H_exp - H_cond
    features['mi'] = mi

    # 3. P(death in window)
    p_death = oracle_probs[:, censored_indices, :-1].mean(0).sum(1)
    features['p_death'] = p_death

    # 4. Censoring time
    features['censor_time'] = artificial_time[censored_indices]

    # 5. True outcome (will they be revealed?)
    will_reveal = np.zeros(n_censored)
    for i, idx in enumerate(censored_indices):
        c = artificial_time[idx]
        if true_train_event[idx] == 1 and true_train_time[idx] <= c + probe_depth:
            will_reveal[i] = 1
    features['will_reveal'] = will_reveal

    # 6. Distance to true event
    dist_to_true = np.zeros(n_censored)
    for i, idx in enumerate(censored_indices):
        c = artificial_time[idx]
        dist_to_true[i] = true_train_time[idx] - c
    features['dist_to_true'] = dist_to_true

    # 7. Ensemble variance
    ens_var = train_preds[:, censored_indices, :].var(0).sum(1)
    features['ens_var'] = ens_var

    # 8. Entropy
    mean_preds_censored = train_preds.mean(0)[censored_indices, :]
    entropy = -np.sum(mean_preds_censored * np.log(mean_preds_censored + 1e-10), axis=1)
    features['entropy'] = entropy

    # Now select batch using different strategies
    strategies = {
        'obs_mass': obs_mass,
        'mi': mi,
        'p_death': p_death,
        'mi_times_pdeath': mi * p_death,
        'obs_mass_times_pdeath': obs_mass * p_death,
        'entropy': entropy,
        'will_reveal_oracle': will_reveal  # Oracle strategy (cheating - for analysis only)
    }

    results = {}

    for strategy_name, scores in strategies.items():
        # Select top batch
        if len(scores) < batch_size:
            selected_local = np.arange(len(scores))
        else:
            selected_local = np.argsort(scores)[-batch_size:]

        selected = censored_indices[selected_local]

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

        # Analyze what was selected
        selected_features = {}
        for feat_name, feat_values in features.items():
            selected_features[f'selected_{feat_name}'] = feat_values[selected_local].mean()

        results[strategy_name] = {
            'improvement': improvement,
            'n_revealed': n_revealed,
            'initial_c': initial_c_index,
            'final_c': updated_c_index,
            **selected_features
        }

    return results, features


def main():
    print("\n" + "="*80)
    print("DEEP ANALYSIS: What Actually Predicts Success?")
    print("="*80 + "\n")

    # Load data
    data = load_nacd_data()
    print(f"Loaded NACD: {data.shape}\n")

    probe_depth = 3
    batch_size = 30
    n_runs = 20  # More runs for robust analysis

    print(f"Running {n_runs} trials to analyze what works...")
    print(f"Settings: probe_depth={probe_depth}, batch_size={batch_size}\n")

    all_results = []

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", flush=True)
        run_results, _ = analyze_single_run(data, probe_depth, batch_size, 42 + run)
        all_results.append(run_results)

    # Aggregate results
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80 + "\n")

    strategy_names = list(all_results[0].keys())

    print(f"{'Strategy':<25} {'Mean Δ':<12} {'Std':<8} {'Avg Reveals':<12} {'Win Rate'}")
    print("-" * 80)

    for strategy in strategy_names:
        improvements = [r[strategy]['improvement'] for r in all_results]
        reveals = [r[strategy]['n_revealed'] for r in all_results]

        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        mean_reveals = np.mean(reveals)
        win_rate = sum(1 for imp in improvements if imp > 0) / len(improvements)

        marker = "🎯" if strategy == 'will_reveal_oracle' else "  "
        print(f"{marker}{strategy:<25} {mean_imp:+.4f}      {std_imp:.4f}   {mean_reveals:.1f}         {win_rate:.0%}")

    # Find what correlates with success
    print("\n" + "="*80)
    print("WHAT PREDICTS SUCCESS?")
    print("="*80 + "\n")

    # For each strategy, correlate selected features with improvement
    for strategy in ['obs_mass', 'mi', 'p_death', 'mi_times_pdeath']:
        print(f"\n{strategy.upper()}:")

        improvements = [r[strategy]['improvement'] for r in all_results]

        feature_correlations = {}
        for feat_name in ['obs_mass', 'mi', 'p_death', 'will_reveal', 'censor_time', 'dist_to_true', 'entropy']:
            selected_feat = [r[strategy][f'selected_{feat_name}'] for r in all_results]

            corr, p_val = pearsonr(selected_feat, improvements)
            feature_correlations[feat_name] = (corr, p_val)

        # Sort by absolute correlation
        sorted_feats = sorted(feature_correlations.items(), key=lambda x: abs(x[1][0]), reverse=True)

        print(f"  Correlations with improvement:")
        for feat_name, (corr, p_val) in sorted_feats:
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"    {feat_name:<20} ρ={corr:+.3f}  p={p_val:.4f} {sig}")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80 + "\n")

    # Find best performing strategy
    best_strategy = max(strategy_names, key=lambda s: np.mean([r[s]['improvement'] for r in all_results]))
    best_mean = np.mean([r[best_strategy]['improvement'] for r in all_results])

    print(f"1. Best strategy: {best_strategy} (Δ={best_mean:+.4f})")

    # Oracle performance
    oracle_mean = np.mean([r['will_reveal_oracle']['improvement'] for r in all_results])
    print(f"2. Oracle (cheating): Δ={oracle_mean:+.4f}")
    print(f"3. Gap to oracle: {oracle_mean - best_mean:.4f}")

    print("\n" + "="*80 + "\n")

    return all_results


if __name__ == '__main__':
    results = main()
