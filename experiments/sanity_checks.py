"""
Sanity Checks for Survival Active Learning on NACD Dataset

This script performs three essential sanity checks:
1. Does the model (BayesMtlr) learn well on NACD data?
2. Does having more labeled data help the model?
3. What types of points are most valuable for learning?

NO artificial censoring, NO specific acquisition functions.
Just basic validation that the components work.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

from model import BayesMtlr


# ============ Data Loading ============

def load_nacd_data():
    """Load NACD dataset."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'MIMIC', 'NACD', 'NACD_Full.csv')
    data = pd.read_csv(data_path)
    data = data.rename(columns={'SURVIVAL': 'time', 'CENSORED': 'event'})

    # Handle missing values
    for col in data.columns:
        if col not in ['time', 'event']:
            data[col] = data[col].replace(-1, 0)
            data[col] = data[col].fillna(0)

    # Standardize features
    feature_cols = [c for c in data.columns if c not in ['time', 'event']]
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])

    return data.astype(float)


# ============ Model Config ============

class ModelConfig:
    """Configuration for BayesMtlr."""
    def __init__(self):
        self.lr = 0.001
        self.weight_decay = 0.001
        self.dropout = 0.2
        self.device = 'cpu'
        self.batch_size = 32
        self.n_samples_train = 5
        self.hidden_size = 64
        self.rho_scale = -5.0
        self.mu_scale = None
        self.sigma_1 = 1.0
        self.sigma_2 = 0.0025


# ============ Helper Functions ============

def make_time_bins(times, num_bins=10):
    """Create time bins from survival times."""
    valid_times = times[times > 0]
    if len(valid_times) == 0:
        return np.linspace(0, 1, num_bins + 1)[1:]
    bins = np.quantile(valid_times, np.linspace(0, 1, num_bins + 1))
    bins = np.unique(bins)
    return bins[1:]  # Return bin edges (excluding 0)


def discretize_times(times, bins):
    """Convert continuous times to bin indices."""
    return np.searchsorted(bins, times)


def encode_survival_mtlr(time_bins, event, num_bins):
    """
    Encode survival data for MTLR.
    For uncensored: one-hot at death time
    For censored: 1s from censoring time onwards
    """
    n = len(time_bins)
    y = np.zeros((n, num_bins))

    for i in range(n):
        bin_idx = min(int(time_bins[i]), num_bins - 1)
        if event[i] == 1:  # Death observed
            y[i, bin_idx] = 1
        else:  # Censored
            y[i, bin_idx:] = 1

    return y


def compute_c_index(risk_scores, time, event):
    """
    Compute concordance index.
    Higher risk score should correspond to earlier death.
    """
    n = len(time)
    concordant = 0
    total = 0

    for i in range(n):
        if event[i] == 0:  # Only count pairs where i has observed event
            continue
        for j in range(n):
            if time[j] > time[i]:  # j survived longer than i
                total += 1
                if risk_scores[i] > risk_scores[j]:  # Correct ordering
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5

    return concordant / total if total > 0 else 0.5


def train_model(X, y, config, epochs=100, verbose=False):
    """Train BayesMtlr model."""
    num_bins = y.shape[1]
    model = BayesMtlr(in_features=X.shape[1], num_time_bins=num_bins, config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    dataset_size = len(X)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, _, _, nll = model.sample_elbo(X_tensor, y_tensor, dataset_size, see1=0.01)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, nll={nll.item():.4f}")

    return model


def evaluate_model(model, X, time_bins, event):
    """Evaluate model using C-index."""
    model.eval()
    with torch.no_grad():
        # Get predictions (mean over samples)
        preds = model(torch.FloatTensor(X), sample=True, n_samples=10)
        preds = torch.softmax(preds.mean(0), dim=1).numpy()

    # Risk score = negative expected survival time
    expected_time = np.sum(preds * np.arange(preds.shape[1]), axis=1)
    risk_scores = -expected_time  # Higher risk = lower expected time

    c_index = compute_c_index(risk_scores, time_bins, event)
    return c_index


# ============ Sanity Check 1: Does the model learn? ============

def sanity_check_1_model_learns(data, n_runs=5):
    """
    Test if BayesMtlr actually learns on NACD data.
    Compare trained model vs random predictions.
    """
    print("\n" + "="*80)
    print("SANITY CHECK 1: Does the model learn on NACD?")
    print("="*80)

    feature_cols = [c for c in data.columns if c not in ['time', 'event']]
    X = data[feature_cols].values.astype(np.float32)
    time = data['time'].values
    event = data['event'].values.astype(int)

    print(f"\nDataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"Events: {event.sum()} deaths ({100*event.mean():.1f}%), {len(event)-event.sum()} censored")

    trained_c_indices = []
    random_c_indices = []

    for run in range(n_runs):
        np.random.seed(42 + run)
        torch.manual_seed(42 + run)

        # Split data
        X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
            X, time, event, test_size=0.3, random_state=42 + run, stratify=event
        )

        # Create time bins from training data
        bins = make_time_bins(time_train, num_bins=10)
        num_bins = len(bins)

        # Discretize
        time_train_bins = discretize_times(time_train, bins)
        time_test_bins = discretize_times(time_test, bins)

        # Encode for MTLR
        y_train = encode_survival_mtlr(time_train_bins, event_train, num_bins)

        # Train model
        config = ModelConfig()
        model = train_model(X_train, y_train, config, epochs=100, verbose=(run == 0))

        # Evaluate
        c_index = evaluate_model(model, X_test, time_test_bins, event_test)
        trained_c_indices.append(c_index)

        # Random baseline
        random_risks = np.random.rand(len(X_test))
        random_c = compute_c_index(random_risks, time_test_bins, event_test)
        random_c_indices.append(random_c)

        print(f"  Run {run+1}: Trained C-index={c_index:.4f}, Random={random_c:.4f}")

    mean_trained = np.mean(trained_c_indices)
    mean_random = np.mean(random_c_indices)

    print(f"\nSummary:")
    print(f"  Trained model: {mean_trained:.4f} ± {np.std(trained_c_indices):.4f}")
    print(f"  Random baseline: {mean_random:.4f} ± {np.std(random_c_indices):.4f}")
    print(f"  Improvement: {mean_trained - mean_random:+.4f}")

    if mean_trained > 0.55:
        print("\n✓ PASS: Model learns (C-index > 0.55)")
        return True
    else:
        print("\n✗ FAIL: Model doesn't learn well (C-index ≤ 0.55)")
        return False


# ============ Sanity Check 2: Does more data help? ============

def sanity_check_2_more_data_helps(data, n_runs=5):
    """
    Test if having more training data improves the model.
    Train on 20%, 40%, 60%, 80% of data and compare C-index.
    """
    print("\n" + "="*80)
    print("SANITY CHECK 2: Does more data help the model?")
    print("="*80)

    feature_cols = [c for c in data.columns if c not in ['time', 'event']]
    X = data[feature_cols].values.astype(np.float32)
    time = data['time'].values
    event = data['event'].values.astype(int)

    train_fractions = [0.2, 0.4, 0.6, 0.8]
    results = {frac: [] for frac in train_fractions}

    for run in range(n_runs):
        np.random.seed(42 + run)
        torch.manual_seed(42 + run)

        # Fixed test set (20%)
        X_rest, X_test, time_rest, time_test, event_rest, event_test = train_test_split(
            X, time, event, test_size=0.2, random_state=42 + run, stratify=event
        )

        # Create bins from all non-test data
        bins = make_time_bins(time_rest, num_bins=10)
        num_bins = len(bins)
        time_test_bins = discretize_times(time_test, bins)

        for frac in train_fractions:
            # Use fraction of remaining data for training
            n_train = int(len(X_rest) * frac)
            indices = np.random.choice(len(X_rest), n_train, replace=False)

            X_train = X_rest[indices]
            time_train = time_rest[indices]
            event_train = event_rest[indices]

            time_train_bins = discretize_times(time_train, bins)
            y_train = encode_survival_mtlr(time_train_bins, event_train, num_bins)

            # Train
            config = ModelConfig()
            model = train_model(X_train, y_train, config, epochs=100)

            # Evaluate
            c_index = evaluate_model(model, X_test, time_test_bins, event_test)
            results[frac].append(c_index)

        print(f"  Run {run+1}: ", end="")
        for frac in train_fractions:
            print(f"{int(frac*100)}%={results[frac][-1]:.3f} ", end="")
        print()

    print(f"\nSummary (mean C-index):")
    for frac in train_fractions:
        mean_c = np.mean(results[frac])
        std_c = np.std(results[frac])
        print(f"  {int(frac*100)}% data: {mean_c:.4f} ± {std_c:.4f}")

    # Check if trend is increasing
    means = [np.mean(results[f]) for f in train_fractions]
    if means[-1] > means[0]:
        print(f"\n✓ PASS: More data helps (80% > 20%: {means[-1]:.4f} > {means[0]:.4f})")
        return True
    else:
        print(f"\n✗ FAIL: More data doesn't help (80% ≤ 20%)")
        return False


# ============ Sanity Check 3: Which points are most valuable? ============

def sanity_check_3_valuable_points(data, n_runs=5):
    """
    Test which types of points are most valuable for learning.

    Start with a small training set, then add different types of points:
    - Uncensored (deaths observed)
    - Early censored (censored early)
    - Late censored (censored late)
    - Random mix

    See which additions improve C-index most.
    """
    print("\n" + "="*80)
    print("SANITY CHECK 3: Which point types are most valuable?")
    print("="*80)

    feature_cols = [c for c in data.columns if c not in ['time', 'event']]
    X = data[feature_cols].values.astype(np.float32)
    time = data['time'].values
    event = data['event'].values.astype(int)

    results = {
        'baseline': [],
        'add_uncensored': [],
        'add_early_censored': [],
        'add_late_censored': [],
        'add_random': []
    }

    for run in range(n_runs):
        np.random.seed(42 + run)
        torch.manual_seed(42 + run)

        # Split: 20% test, 80% pool
        indices = np.random.permutation(len(X))
        n_test = int(0.2 * len(X))
        test_idx = indices[:n_test]
        pool_idx = indices[n_test:]

        X_test = X[test_idx]
        time_test = time[test_idx]
        event_test = event[test_idx]

        X_pool = X[pool_idx]
        time_pool = time[pool_idx]
        event_pool = event[pool_idx]

        # Create bins
        bins = make_time_bins(time_pool, num_bins=10)
        num_bins = len(bins)
        time_test_bins = discretize_times(time_test, bins)
        time_pool_bins = discretize_times(time_pool, bins)

        # Start with small baseline (20% of pool = 16% of total)
        n_baseline = int(0.2 * len(pool_idx))
        baseline_idx = np.random.choice(len(pool_idx), n_baseline, replace=False)

        X_base = X_pool[baseline_idx]
        time_base_bins = time_pool_bins[baseline_idx]
        event_base = event_pool[baseline_idx]
        y_base = encode_survival_mtlr(time_base_bins, event_base, num_bins)

        # Train baseline
        config = ModelConfig()
        model_base = train_model(X_base, y_base, config, epochs=100)
        c_base = evaluate_model(model_base, X_test, time_test_bins, event_test)
        results['baseline'].append(c_base)

        # Remaining pool
        remaining_mask = np.ones(len(pool_idx), dtype=bool)
        remaining_mask[baseline_idx] = False
        remaining_idx = np.where(remaining_mask)[0]

        n_add = int(0.2 * len(pool_idx))  # Add 20% more

        # Categorize remaining points
        uncensored_idx = remaining_idx[event_pool[remaining_idx] == 1]
        censored_idx = remaining_idx[event_pool[remaining_idx] == 0]

        # Split censored by time
        if len(censored_idx) > 0:
            censored_times = time_pool_bins[censored_idx]
            median_censor_time = np.median(censored_times)
            early_censored_idx = censored_idx[censored_times <= median_censor_time]
            late_censored_idx = censored_idx[censored_times > median_censor_time]
        else:
            early_censored_idx = np.array([], dtype=int)
            late_censored_idx = np.array([], dtype=int)

        # Test each type
        for point_type, available_idx in [
            ('add_uncensored', uncensored_idx),
            ('add_early_censored', early_censored_idx),
            ('add_late_censored', late_censored_idx),
            ('add_random', remaining_idx)
        ]:
            if len(available_idx) < n_add:
                # Not enough points, use what's available
                add_idx = available_idx
            else:
                add_idx = np.random.choice(available_idx, n_add, replace=False)

            if len(add_idx) == 0:
                results[point_type].append(c_base)
                continue

            # Combine with baseline
            combined_idx = np.concatenate([baseline_idx, add_idx])
            X_train = X_pool[combined_idx]
            time_train_bins = time_pool_bins[combined_idx]
            event_train = event_pool[combined_idx]
            y_train = encode_survival_mtlr(time_train_bins, event_train, num_bins)

            # Train
            torch.manual_seed(42 + run)  # Same init for fair comparison
            model = train_model(X_train, y_train, config, epochs=100)
            c_index = evaluate_model(model, X_test, time_test_bins, event_test)
            results[point_type].append(c_index)

        print(f"  Run {run+1}: base={c_base:.3f}, +uncens={results['add_uncensored'][-1]:.3f}, "
              f"+early_cens={results['add_early_censored'][-1]:.3f}, +late_cens={results['add_late_censored'][-1]:.3f}, "
              f"+random={results['add_random'][-1]:.3f}")

    print(f"\nSummary (mean C-index, improvement over baseline):")
    baseline_mean = np.mean(results['baseline'])
    print(f"  Baseline: {baseline_mean:.4f}")

    improvements = {}
    for point_type in ['add_uncensored', 'add_early_censored', 'add_late_censored', 'add_random']:
        mean_c = np.mean(results[point_type])
        improvement = mean_c - baseline_mean
        improvements[point_type] = improvement
        print(f"  {point_type}: {mean_c:.4f} ({improvement:+.4f})")

    # Find best
    best_type = max(improvements, key=improvements.get)
    print(f"\n✓ Most valuable: {best_type} ({improvements[best_type]:+.4f})")

    return improvements


# ============ Main ============

def main():
    print("\n" + "="*100)
    print("SANITY CHECKS FOR SURVIVAL ACTIVE LEARNING ON NACD")
    print("="*100)
    print(f"Started: {datetime.now()}")

    # Load data
    print("\nLoading NACD dataset...")
    data = load_nacd_data()
    print(f"Loaded: {data.shape[0]} samples, {data.shape[1]-2} features")

    # Run sanity checks
    check1_passed = sanity_check_1_model_learns(data, n_runs=5)
    check2_passed = sanity_check_2_more_data_helps(data, n_runs=5)
    improvements = sanity_check_3_valuable_points(data, n_runs=5)

    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"\n1. Model learns: {'PASS ✓' if check1_passed else 'FAIL ✗'}")
    print(f"2. More data helps: {'PASS ✓' if check2_passed else 'FAIL ✗'}")
    print(f"3. Most valuable points: {max(improvements, key=improvements.get)}")

    if check1_passed and check2_passed:
        print("\n→ The model and data are working. Problem is likely in the AL setup.")
    elif not check1_passed:
        print("\n→ The model doesn't learn well. Need to fix model/data first.")
    elif not check2_passed:
        print("\n→ More data doesn't help. Model may be overfitting or data is noisy.")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()
