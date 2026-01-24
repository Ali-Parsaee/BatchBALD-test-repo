"""
Diagnose: What is the distribution of P(death in window)?

Understand why Conservative methods are filtering out everything.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from src.oracle.oracle import Oracle
from src.models.survival_model import BayesianSurvivalModel
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


def main():
    print("\n" + "="*80)
    print("DIAGNOSTIC: P(death in window) Distribution")
    print("="*80 + "\n")

    # Load data
    data = load_nacd_data()
    train_df, test_df = split_data(data, random_state=42)

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

    print(f"Dataset: {len(train_df)} train, {len(test_df)} test")
    print(f"Time bins: {n_bins}")
    print(f"Event rate (train): {event_train.mean():.2%}\n")

    # Ground truth
    true_train_time = time_train_binned.copy()
    true_train_event = event_train.copy()

    # Artificially censor
    probe_depth = 3
    artificial_time, artificial_event = artificially_censor(
        time_train_binned, event_train, proportion=0.5, random_state=42
    )

    print(f"After artificial censoring:")
    print(f"  Event rate: {artificial_event.mean():.2%}")
    print(f"  Censored: {(artificial_event == 0).sum()} instances\n")

    # Train model
    print("Training model...")
    model = BayesianSurvivalModel(
        n_features=X_train.shape[1],
        n_time_bins=n_bins,
        n_ensemble=5,
        hidden_size=64
    )
    model.fit(X_train, artificial_time, artificial_event, epochs=20, batch_size=32, verbose=False)
    print("Done!\n")

    # Get predictions
    train_preds = model.predict_proba(X_train)  # (K, N, T)

    # Oracle
    oracle = Oracle(true_train_time, true_train_event, probe_depth)
    oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
        train_preds, artificial_time, artificial_event
    )  # (K, N, W) where W = probe_depth + 1

    # Analyze censored instances only
    censored_mask = (artificial_event == 0)
    censored_indices = np.where(censored_mask)[0]
    n_censored = len(censored_indices)

    print(f"Analyzing {n_censored} censored instances...\n")

    # Compute P(death in window) for each censored instance
    # oracle_probs has shape (K, N, W) where W = probe_depth + 1
    # Last outcome is censoring extension, others are death outcomes
    p_death_all = []

    for idx in censored_indices:
        # Mean over ensemble, sum over death outcomes (exclude last = censoring)
        p_death = oracle_probs[:, idx, :-1].mean(0).sum()
        p_death_all.append(p_death)

    p_death_all = np.array(p_death_all)

    print("P(death in window) Statistics:")
    print(f"  Mean: {p_death_all.mean():.4f}")
    print(f"  Median: {np.median(p_death_all):.4f}")
    print(f"  Std: {p_death_all.std():.4f}")
    print(f"  Min: {p_death_all.min():.4f}")
    print(f"  Max: {p_death_all.max():.4f}\n")

    print("Distribution:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(p_death_all, p)
        print(f"  {p}th percentile: {val:.4f}")

    print("\nThreshold Analysis:")
    for threshold in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        n_above = (p_death_all >= threshold).sum()
        pct = 100 * n_above / len(p_death_all)
        print(f"  P(death) >= {threshold:.2f}: {n_above}/{len(p_death_all)} ({pct:.1f}%)")

    print("\nActual Oracle Revelation Rate:")
    # Check ground truth
    will_reveal = []
    for idx in censored_indices:
        c = artificial_time[idx]
        if true_train_event[idx] == 1 and true_train_time[idx] <= c + probe_depth:
            will_reveal.append(1)
        else:
            will_reveal.append(0)
    will_reveal = np.array(will_reveal)
    print(f"  {will_reveal.sum()}/{len(will_reveal)} ({100*will_reveal.mean():.1f}%) will actually be revealed by oracle")

    # Correlation
    if will_reveal.sum() > 0:
        from scipy.stats import pearsonr
        corr, pval = pearsonr(p_death_all, will_reveal)
        print(f"  Correlation P(death) vs actual revelation: ρ={corr:.3f}, p={pval:.4f}")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
