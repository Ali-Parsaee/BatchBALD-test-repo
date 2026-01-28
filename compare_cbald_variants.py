"""
Compare C-BALD vs BatchBALD vs CBALD_True across multiple budgets.

Comparison:
1. C-BALD (Simplified): time_variance * (0.5 + 0.5 * death_prob)
2. BatchBALD (Pure): Joint entropy maximization
3. CBALD_True (Paper-correct): I(l;θ|x) + I(y;θ|l,x)

Budgets: 10, 20, 30
Trials: 5 per method per budget
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'Model_stuff')

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import argparse
from scipy import stats
import time
import copy

from Model_stuff.model import BayesLinMtlr, mtlr_survival
from Model_stuff.acquisition import (
    cbald_censored_regression,  # C-BALD (simplified)
    batchbald_acquire_budget,   # BatchBALD (pure)
    cbald_true_acquire,         # CBALD_True (paper-correct)
)


# ==================== HELPER FUNCTIONS ====================

def encode_survival(time, event, bins):
    """Encodes survival time and event indicator for MTLR training."""
    if isinstance(time, (float, int, np.ndarray)):
        time = np.atleast_1d(time)
        time = torch.tensor(time)
    if isinstance(event, (int, bool, np.ndarray)):
        event = np.atleast_1d(event)
        event = torch.tensor(event)
    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins)

    bins = torch.as_tensor(bins, device=time.device, dtype=time.dtype)
    device = bins.device if hasattr(bins, 'device') else "cpu"
    time = np.clip(time, 0, bins.max())
    y = torch.zeros((time.shape[0], bins.shape[0] + 1), dtype=torch.float, device=device)
    bin_idxs = torch.bucketize(time, bins, right=True)
    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y.squeeze()


def reformat_survival(dataset, time_bins):
    """Reformat survival data for training."""
    x = torch.tensor(dataset.drop(["time", "event"], axis=1).values, dtype=torch.float)
    y = encode_survival(dataset["time"].values, dataset["event"].values, time_bins)
    return x, y


def artificially_censor_true(times, events, num_initial_samples=50, seed=None):
    """Artificially censor data points."""
    if seed is not None:
        np.random.seed(seed)

    censored_times = np.copy(times)
    new_events = np.copy(events)
    uncensored_indices = np.random.choice(len(times), num_initial_samples, replace=False)
    censored_indices = []

    for i in range(len(times)):
        if i in uncensored_indices:
            continue
        censoring_time = np.random.uniform(0, times[i])
        censored_times[i] = censoring_time
        new_events[i] = 0
        censored_indices.append(i)

    return censored_times, new_events, np.array(censored_indices)


def concordance(y_pred, y_test, cens):
    """Calculate the concordance index (C-index)."""
    n = 0
    n_concordant = 0
    for i in range(len(y_pred)):
        if cens[i] == 1:
            for j in range(len(y_pred)):
                if cens[j] == 1 and y_test[j] > y_test[i]:
                    n += 1
                    if y_pred[j] > y_pred[i]:
                        n_concordant += 1
                elif cens[j] == 0 and y_test[j] >= y_test[i]:
                    n += 1
                    if y_pred[j] > y_pred[i]:
                        n_concordant += 1
    return n_concordant / n if n > 0 else 0.0


class SimpleConfig:
    """Simple configuration class."""
    def __init__(self, n_samples_train=10, n_samples_test=50):
        self.n_samples_train = n_samples_train
        self.n_samples_test = n_samples_test


# ==================== MAIN EXPERIMENT ====================

def run_single_trial(budget, method_name, acq_func, trial_num, device='cpu'):
    """Run a single trial for a given method and budget."""

    print(f"\n{'='*80}")
    print(f"Trial {trial_num + 1}/5 - Budget={budget} - Method={method_name}")
    print(f"{'='*80}\n")

    # Set seed for reproducibility
    seed = 42 + trial_num
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    nacd_path = "data/MIMIC/NACD/NACD_Full.csv"
    data = pd.read_csv(nacd_path)

    # Drop columns
    cols_to_drop = ['PERFORMANCE_STATUS', 'STAGE_NUMERICAL', 'AGE65']
    data = data.drop(columns=[c for c in cols_to_drop if c in data.columns], errors='ignore')
    data = data.dropna()

    # Rename to match expected columns
    if 'SURVIVAL' in data.columns and 'CENSORED' in data.columns:
        data = data.rename(columns={'SURVIVAL': 'time', 'CENSORED': 'event'})

    # Train/test split
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=seed)

    # Artificially censor training data
    num_initial_samples = 200
    artificial_time, artificial_event, censored_indices = artificially_censor_true(
        train_data['time'].values, train_data['event'].values,
        num_initial_samples=num_initial_samples, seed=seed
    )

    train_data_copy = train_data.copy()
    train_data_copy['time'] = artificial_time
    train_data_copy['event'] = artificial_event

    # Standardize features
    scaler = StandardScaler()
    features = [c for c in train_data_copy.columns if c not in ['time', 'event']]
    train_data_copy[features] = scaler.fit_transform(train_data_copy[features])
    test_data[features] = scaler.transform(test_data[features])

    # Create time bins
    train_times = train_data['time'].values
    time_bins = np.percentile(train_times, np.linspace(0, 100, 11)[1:])
    time_bins = torch.tensor(np.unique(time_bins), dtype=torch.float32)

    # Initial model training
    x_train, y_train = reformat_survival(train_data_copy, time_bins)
    x_test, y_test = reformat_survival(test_data, time_bins)

    config = SimpleConfig()
    num_time_bins = len(time_bins) - 1
    model = BayesLinMtlr(in_features=x_train.shape[1], num_time_bins=num_time_bins)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    model.train()

    for epoch in range(100):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            log_hazard = model.forward(batch_x, sample=True, n_samples=10)
            loss = model.mtlr_nll(log_hazard, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate initial model
    model.eval()
    with torch.no_grad():
        test_logits = model.forward(x_test, sample=False, n_samples=0)
        test_probs = mtlr_survival(test_logits)
        test_probs_np = test_probs.cpu().numpy()
        test_median_survival = np.array([
            time_bins[np.searchsorted(test_probs_np[i], 0.5)] for i in range(len(test_probs_np))
        ])

    initial_cindex = concordance(test_median_survival, test_data['time'].values, test_data['event'].values)
    print(f"Initial C-index: {initial_cindex:.4f}")

    # Active learning loop
    increment = 10000
    batch_size = budget

    X_pool = train_data_copy[features].values

    # Get predictions for pool
    model.eval()
    with torch.no_grad():
        x_pool_tensor = torch.FloatTensor(X_pool).to(device)
        pool_logits = model.forward(x_pool_tensor, sample=True, n_samples=50)

    # Select batch based on method
    in_data_train = {
        'time': artificial_time,
        'event': artificial_event
    }

    if method_name == 'BatchBALD':
        # BatchBALD needs censored_indices
        censored_mask = (artificial_event == 0)
        censored_idx_list = np.where(censored_mask)[0].tolist()
        selected_indices = acq_func(
            model, X_pool, batch_size, time_bins, config,
            device=device, in_data_train=in_data_train, increment=increment,
            censored_indices=censored_idx_list, costlist=None, budget=0
        )
    else:
        # C-BALD and CBALD_True
        selected_indices, scores = acq_func(
            model, X_pool, batch_size, time_bins, config,
            device=device, in_data_train=in_data_train, increment=increment,
            costlist=None, budget=0
        )

    print(f"Selected {len(selected_indices)} samples")

    # Reveal oracle labels for selected samples
    true_times = train_data['time'].values
    true_events = train_data['event'].values

    for idx in selected_indices:
        current_time = artificial_time[idx]
        true_time = true_times[idx]
        true_event = true_events[idx]

        # Reveal up to min(true_time, current_time + increment)
        revealed_time = min(true_time, current_time + increment)
        revealed_event = 1 if (true_event == 1 and true_time <= current_time + increment) else 0

        artificial_time[idx] = revealed_time
        artificial_event[idx] = revealed_event

    # Update training data
    train_data_copy['time'] = artificial_time
    train_data_copy['event'] = artificial_event

    # Retrain model
    x_train, y_train = reformat_survival(train_data_copy, time_bins)
    model = BayesLinMtlr(in_features=x_train.shape[1], num_time_bins=num_time_bins)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    model.train()

    for epoch in range(100):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            log_hazard = model.forward(batch_x, sample=True, n_samples=10)
            loss = model.mtlr_nll(log_hazard, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluate final model
    model.eval()
    with torch.no_grad():
        test_logits = model.forward(x_test, sample=False, n_samples=0)
        test_probs = mtlr_survival(test_logits)
        test_probs_np = test_probs.cpu().numpy()
        test_median_survival = np.array([
            time_bins[np.searchsorted(test_probs_np[i], 0.5)] for i in range(len(test_probs_np))
        ])

    final_cindex = concordance(test_median_survival, test_data['time'].values, test_data['event'].values)
    improvement = final_cindex - initial_cindex

    print(f"Final C-index: {final_cindex:.4f}")
    print(f"Improvement: {improvement:+.4f}")

    return initial_cindex, final_cindex, improvement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--budgets', type=str, default='10,20,30', help='Comma-separated budgets')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials per method')
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(',')]

    # Methods to compare
    methods = [
        (cbald_censored_regression, 'C-BALD'),
        (batchbald_acquire_budget, 'BatchBALD'),
        (cbald_true_acquire, 'CBALD_True'),
    ]

    # Store results
    all_results = {}

    for budget in budgets:
        print(f"\n{'#'*80}")
        print(f"# BUDGET = {budget}")
        print(f"{'#'*80}\n")

        budget_results = {}

        for acq_func, method_name in methods:
            improvements = []

            for trial in range(args.trials):
                try:
                    initial, final, improvement = run_single_trial(
                        budget, method_name, acq_func, trial
                    )
                    improvements.append(improvement)
                except Exception as e:
                    print(f"ERROR in {method_name} trial {trial}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            if improvements:
                budget_results[method_name] = improvements
                mean_imp = np.mean(improvements)
                std_imp = np.std(improvements)
                print(f"\n{method_name}: {mean_imp:+.4f} ± {std_imp:.4f}")

        all_results[budget] = budget_results

    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY: C-BALD vs BatchBALD vs CBALD_True")
    print(f"{'='*80}\n")

    for budget in budgets:
        print(f"\n{'='*60}")
        print(f"Budget = {budget}")
        print(f"{'='*60}")

        results = all_results[budget]
        sorted_methods = sorted(results.items(), key=lambda x: np.mean(x[1]), reverse=True)

        print(f"\n{'Rank':<6} {'Method':<15} {'Improvement':<20} {'vs C-BALD'}")
        print("-" * 60)

        cbald_mean = np.mean(results['C-BALD']) if 'C-BALD' in results else 0

        for rank, (method_name, improvements) in enumerate(sorted_methods, 1):
            mean_imp = np.mean(improvements)
            std_imp = np.std(improvements)
            diff_cbald = mean_imp - cbald_mean

            emoji = '🏆' if rank == 1 else '🥈' if rank == 2 else '🥉'
            print(f"{emoji} {rank:<3} {method_name:<15} {mean_imp:+.4f} ± {std_imp:.4f}     {diff_cbald:+.4f}")

        # Statistical tests
        if 'C-BALD' in results and 'CBALD_True' in results and len(results['C-BALD']) >= 3:
            t_stat, p_value = stats.ttest_rel(results['CBALD_True'], results['C-BALD'])
            print(f"\nCBALD_True vs C-BALD: t={t_stat:.3f}, p={p_value:.4f}")
            if p_value < 0.05:
                winner = 'CBALD_True' if np.mean(results['CBALD_True']) > np.mean(results['C-BALD']) else 'C-BALD'
                print(f"  → {winner} is significantly better (p<0.05)")

        if 'C-BALD' in results and 'BatchBALD' in results and len(results['C-BALD']) >= 3:
            t_stat, p_value = stats.ttest_rel(results['C-BALD'], results['BatchBALD'])
            print(f"\nC-BALD vs BatchBALD: t={t_stat:.3f}, p={p_value:.4f}")
            if p_value < 0.05:
                winner = 'C-BALD' if np.mean(results['C-BALD']) > np.mean(results['BatchBALD']) else 'BatchBALD'
                print(f"  → {winner} is significantly better (p<0.05)")


if __name__ == '__main__':
    main()
