"""
Fair Comparison: C-BALD vs CBALD_True

Using EXACT settings from test_cbald_variants.py that produced C-BALD's +0.0173:
- Budget: 20
- Trials: 5
- increment: 60
- Dataset: NACD
- num_initial_samples: 50
- test_size: 0.2
- num_bins: 10
- Model: BayesLinMtlr
- lr: 8e-5
- epochs: 100 (early stopping, patience=10)
- n_samples_test: 100

Methods:
1. C-BALD: time_variance * (0.5 + 0.5 * death_prob)
2. CBALD_True: I(l;theta|x) + I(y;theta|l,x) (paper-correct)
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


def expected_times_from_survival(surv_np, tbins):
    """Convert survival probabilities to expected survival times."""
    if isinstance(tbins, torch.Tensor):
        tb = tbins.clone().detach().cpu().numpy()
    else:
        tb = np.array(tbins)

    # Handle shape mismatch: survival output may have one more column than time_bins
    n_bins = len(tb)
    surv = surv_np[:, :n_bins] if surv_np.shape[1] > n_bins else surv_np

    left_edges = np.concatenate([[0.0], tb[:-1]])
    right_edges = tb
    mids = (left_edges + right_edges) / 2.0
    pdf = np.zeros_like(surv)
    pdf[:, 0] = 1.0 - surv[:, 0]
    pdf[:, 1:] = surv[:, :-1] - surv[:, 1:]
    pdf = np.clip(pdf, 0.0, 1.0)
    pdf = pdf / (pdf.sum(axis=1, keepdims=True) + 1e-12)
    return (pdf * mids.reshape(1, -1)).sum(axis=1)


def concordance(y_pred, y_test, cens):
    """Calculate concordance index."""
    n = len(y_pred)
    if n < 2:
        return np.nan

    concordant = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            if cens[i] == 1 and cens[j] == 1:
                total_pairs += 1
                if (y_test[i] < y_test[j] and y_pred[i] < y_pred[j]) or \
                   (y_test[i] > y_test[j] and y_pred[i] > y_pred[j]):
                    concordant += 1
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5
            elif cens[i] == 1 and cens[j] == 0:
                if y_test[i] < y_test[j]:
                    total_pairs += 1
                    if y_pred[i] < y_pred[j]:
                        concordant += 1
                    elif y_pred[i] == y_pred[j]:
                        concordant += 0.5
            elif cens[i] == 0 and cens[j] == 1:
                if y_test[j] < y_test[i]:
                    total_pairs += 1
                    if y_pred[j] < y_pred[i]:
                        concordant += 1
                    elif y_pred[i] == y_pred[j]:
                        concordant += 0.5

    if total_pairs == 0:
        return np.nan
    return concordant / total_pairs


# ==================== MAIN EXPERIMENT ====================

def run_single_trial(trial_num, budget=20, methods_to_run=None):
    """Run single trial comparing C-BALD and CBALD_True.

    Args:
        trial_num: Trial number (1-indexed)
        budget: Number of samples to acquire
        methods_to_run: List of method names to run. If None, runs all.
                       Options: ['C-BALD', 'CBALD_True']
    """

    print(f"  Trial {trial_num}/5")
    print("  " + "-" * 68)

    # Set seed for reproducibility (same as test_cbald_variants.py)
    seed = 100 + trial_num - 1
    print(f"      Setting random seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    df = pd.read_csv('data/MIMIC/NACD/NACD_Full.csv')

    # Handle column names (NACD uses CENSORED/SURVIVAL)
    if "CENSORED" in df.columns:
        df["event"] = 1 - df["CENSORED"]
        df = df.drop(columns=["CENSORED"])
    if "SURVIVAL" in df.columns:
        df = df.rename(columns={"SURVIVAL": "time"})

    # Preprocess
    X = df.drop(columns=['time', 'event'])
    y_time = df['time'].values
    y_event = df['event'].values

    X_train_val, X_test, y_time_train_val, y_time_test, y_event_train_val, y_event_test = train_test_split(
        X, y_time, y_event, test_size=0.2, random_state=seed, stratify=y_event
    )

    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)

    # Artificially censor (num_initial_samples=50 as in test_cbald_variants.py)
    y_time_censored, y_event_censored, censored_indices = artificially_censor_true(
        y_time_train_val, y_event_train_val, num_initial_samples=50, seed=seed
    )

    # Setup (num_bins=10 as in test_cbald_variants.py)
    num_bins = 10
    event_times = y_time_train_val[y_event_train_val == 1]
    quantiles = np.linspace(0, 1, num_bins + 1)[1:]
    time_bins = np.quantile(event_times, quantiles)
    time_bins[-1] *= 1.05
    time_bins = np.array([0] + list(time_bins))

    # Config (exact same as test_cbald_variants.py)
    config = argparse.Namespace()
    config.pi = 0.5
    config.sigma1 = 1.0
    config.sigma2 = 0.0025
    config.rho_scale = -3.0
    config.mu_scale = 0.1
    config.batch_size = 32
    config.c1 = 0.01
    config.n_samples_train = 10
    config.n_samples_test = 100
    config.device = "cpu"
    config.patience = 10

    # Train SHARED base model
    print(f"      Training SHARED base model for all methods...")
    start_time = time.time()

    train_df = pd.DataFrame(X_train_val_scaled)
    train_df['time'] = y_time_censored
    train_df['event'] = y_event_censored
    test_df = pd.DataFrame(X_test_scaled)
    test_df['time'] = y_time_test
    test_df['event'] = y_event_test

    x_train, y_train = reformat_survival(train_df, time_bins)
    x_test, y_test = reformat_survival(test_df, time_bins)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    shared_model = BayesLinMtlr(
        in_features=X_train_val_scaled.shape[1],
        num_time_bins=len(time_bins),
        config=config
    )

    # lr=8e-5 as in test_cbald_variants.py
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=8e-5)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(100):
        shared_model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)

            optimizer.zero_grad()
            loss, _, _, _ = shared_model.sample_elbo(batch_x, batch_y, len(x_train), see1=config.c1)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    train_time = time.time() - start_time
    print(f"      Shared base model trained ({train_time:.1f}s)")

    # Evaluate shared model on test set
    shared_model.eval()
    with torch.no_grad():
        logits = shared_model.forward(x_test, sample=True, n_samples=config.n_samples_test)
        survival_probs = mtlr_survival(logits, with_sample=True)
        mean_survival = survival_probs.mean(dim=0).cpu().numpy()

    pred_times = expected_times_from_survival(mean_survival, time_bins)
    initial_cindex = concordance(-pred_times, y_time_test, y_event_test)
    print(f"      Shared initial C-index: {initial_cindex:.4f}")
    print("      All methods will start from this SAME model!")

    # Setup for acquisition
    X_censored = X_train_val_scaled[censored_indices]
    y_time_original = y_time_train_val[censored_indices]
    y_event_original = y_event_train_val[censored_indices]

    train_df_censored = pd.DataFrame(X_train_val_scaled)
    train_df_censored['time'] = y_time_censored
    train_df_censored['event'] = y_event_censored
    x_train_censored, y_train_censored = reformat_survival(train_df_censored, time_bins)

    # model_data_censored needs 'time' key for cbald_score function
    # 'time' should be the censored times corresponding to samples in X_pool (X_censored)
    model_data_censored = {
        'x_train': x_train_censored,
        'y_train': y_train_censored,
        'x_test': x_test,
        'y_test': y_test,
        'time_bins': time_bins,
        'time': y_time_censored[censored_indices],  # censored times for the pool
        'event': y_event_censored[censored_indices]  # censored events for the pool
    }

    # Define acquisition functions to test
    all_acquisition_functions = [
        (cbald_censored_regression, 'C-BALD', {}),
        (cbald_true_acquire, 'CBALD_True', {}),
    ]

    # Filter if specific methods requested
    if methods_to_run is not None:
        acquisition_functions = [
            (func, name, params) for func, name, params in all_acquisition_functions
            if name in methods_to_run
        ]
    else:
        acquisition_functions = all_acquisition_functions

    results = {}
    # increment=60 as in test_cbald_variants.py
    increment = 60
    costlist = np.ones(len(X_censored))

    for acq_func, acq_name, acq_params in acquisition_functions:
        print(f"\n      [{acq_name}] Starting acquisition...")

        # Clone the shared model by saving/loading state dict
        model = BayesLinMtlr(
            in_features=X_train_val_scaled.shape[1],
            num_time_bins=len(time_bins),
            config=config
        )
        model.load_state_dict(shared_model.state_dict())

        # Verify same initial performance
        model.eval()
        with torch.no_grad():
            logits = model.forward(x_test, sample=True, n_samples=config.n_samples_test)
            survival_probs = mtlr_survival(logits, with_sample=True)
            mean_survival = survival_probs.mean(dim=0).cpu().numpy()
        pred_times = expected_times_from_survival(mean_survival, time_bins)
        init_c = concordance(-pred_times, y_time_test, y_event_test)
        print(f"      [{acq_name}] Initial C-index: {init_c:.4f} (verified same as base)")

        # Run acquisition
        acq_start = time.time()

        pool_indices, _ = acq_func(
            model=model,
            X_pool=X_censored,
            batch_size=budget,
            time_bins=time_bins,
            config=config,
            device=config.device,
            in_data_train=model_data_censored,
            increment=increment,
            costlist=costlist,
            budget=budget,
            **acq_params
        )

        acq_time = time.time() - acq_start
        print(f"      [{acq_name}] Running acquisition (budget={budget})... done ({acq_time:.1f}s, acquired {len(pool_indices)} samples)")

        # Reveal oracle labels
        revealed_indices = censored_indices[pool_indices]
        y_time_updated = np.copy(y_time_censored)
        y_event_updated = np.copy(y_event_censored)
        y_time_updated[revealed_indices] = y_time_original[pool_indices]
        y_event_updated[revealed_indices] = y_event_original[pool_indices]

        # Retrain model
        print(f"      [{acq_name}] Retraining model...", end=" ")
        retrain_start = time.time()

        train_df_updated = pd.DataFrame(X_train_val_scaled)
        train_df_updated['time'] = y_time_updated
        train_df_updated['event'] = y_event_updated
        x_train_updated, y_train_updated = reformat_survival(train_df_updated, time_bins)

        train_dataset_updated = TensorDataset(x_train_updated, y_train_updated)
        train_loader_updated = DataLoader(train_dataset_updated, batch_size=config.batch_size, shuffle=True)

        model = BayesLinMtlr(
            in_features=X_train_val_scaled.shape[1],
            num_time_bins=len(time_bins),
            config=config
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(100):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader_updated:
                batch_x = batch_x.to(config.device)
                batch_y = batch_y.to(config.device)

                optimizer.zero_grad()
                loss, _, _, _ = model.sample_elbo(batch_x, batch_y, len(x_train_updated), see1=config.c1)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader_updated)

            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    break

        retrain_time = time.time() - retrain_start
        print(f"done ({retrain_time:.1f}s)")

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model.forward(x_test, sample=True, n_samples=config.n_samples_test)
            survival_probs = mtlr_survival(logits, with_sample=True)
            mean_survival = survival_probs.mean(dim=0).cpu().numpy()

        pred_times = expected_times_from_survival(mean_survival, time_bins)
        final_cindex = concordance(-pred_times, y_time_test, y_event_test)
        improvement = final_cindex - initial_cindex
        total_time = acq_time + retrain_time

        print(f"      [{acq_name}] Final C-index: {final_cindex:.4f} (Delta={improvement:+.4f}) [Total: {total_time:.1f}s]")

        results[acq_name] = {
            'initial_cindex': initial_cindex,
            'final_cindex': final_cindex,
            'improvement': improvement,
            'time': total_time
        }

    print(f"\n  Trial {trial_num} completed\n")

    return results


def main():
    """Run all trials and report results."""

    parser = argparse.ArgumentParser(description='Compare C-BALD vs CBALD_True')
    parser.add_argument('--verify-cbald-only', action='store_true',
                        help='Only run C-BALD to verify +0.0173 result')
    parser.add_argument('--trials', type=int, default=5,
                        help='Number of trials (default: 5)')
    parser.add_argument('--budget', type=int, default=20,
                        help='Budget for acquisition (default: 20)')
    args = parser.parse_args()

    print("=" * 70)
    if args.verify_cbald_only:
        print("VERIFICATION: C-BALD ONLY")
        print("=" * 70)
        print()
        print("Verifying C-BALD achieves ~+0.0173 with exact settings")
        methods_to_run = ['C-BALD']
    else:
        print("FAIR COMPARISON: C-BALD vs CBALD_True")
        print("=" * 70)
        print()
        print("Methods:")
        print("  1. C-BALD: time_variance * (0.5 + 0.5 * death_prob)")
        print("  2. CBALD_True: I(l;theta|x) + I(y;theta|l,x) (paper-correct)")
        methods_to_run = None

    print()
    print("Settings (EXACT match to test_cbald_variants.py):")
    print(f"  Trials: {args.trials}")
    print(f"  Budget: {args.budget}")
    print("  increment: 60")
    print("  num_initial_samples: 50")
    print("  test_size: 0.2")
    print("  num_bins: 10")
    print("  lr: 8e-5")
    print("  patience: 10")
    print("  n_samples_test: 100")
    print()

    # Load data once to display info
    df = pd.read_csv('data/MIMIC/NACD/NACD_Full.csv')
    if "CENSORED" in df.columns:
        df["event"] = 1 - df["CENSORED"]
        df = df.drop(columns=["CENSORED"])
    if "SURVIVAL" in df.columns:
        df = df.rename(columns={"SURVIVAL": "time"})
    print("Loading NACD dataset...")
    print(f"   Dataset shape: {df.shape}")
    print(f"   Device: cpu")
    print()

    # Run trials
    print(f"Running {args.trials} trials...")
    print()

    all_results = {}

    for trial_num in range(1, args.trials + 1):
        trial_results = run_single_trial(trial_num, budget=args.budget, methods_to_run=methods_to_run)

        for method, metrics in trial_results.items():
            if method not in all_results:
                all_results[method] = {
                    'initial_cindices': [],
                    'final_cindices': [],
                    'improvements': [],
                    'times': []
                }
            all_results[method]['initial_cindices'].append(metrics['initial_cindex'])
            all_results[method]['final_cindices'].append(metrics['final_cindex'])
            all_results[method]['improvements'].append(metrics['improvement'])
            all_results[method]['times'].append(metrics['time'])

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # Compute statistics
    method_stats = {}
    for method, data in all_results.items():
        method_stats[method] = {
            'mean_improvement': np.mean(data['improvements']),
            'std_improvement': np.std(data['improvements'], ddof=1),
            'mean_time': np.mean(data['times']),
            'mean_initial': np.mean(data['initial_cindices']),
            'std_initial': np.std(data['initial_cindices'], ddof=1),
            'improvements': data['improvements']
        }

    # Sort by mean improvement
    sorted_methods = sorted(method_stats.items(), key=lambda x: x[1]['mean_improvement'], reverse=True)

    print("Method Rankings (by mean IMPROVEMENT):")
    print("-" * 70)
    for i, (method, stats) in enumerate(sorted_methods):
        print(f"{i+1}. {method:30s} {stats['mean_improvement']:+.4f} +/- {stats['std_improvement']:.4f}")
        print(f"   Individual trials: {[f'{x:+.4f}' for x in stats['improvements']]}")

    print()
    print("Initial C-index (should be SAME for all methods):")
    print("-" * 70)
    for method, stats in sorted_methods:
        print(f"{method:30s} {stats['mean_initial']:.4f} +/- {stats['std_initial']:.4f}")

    # Verification check
    if args.verify_cbald_only:
        print()
        print("=" * 70)
        print("VERIFICATION RESULT")
        print("=" * 70)
        if 'C-BALD' in method_stats:
            mean_imp = method_stats['C-BALD']['mean_improvement']
            expected = 0.0173
            diff = abs(mean_imp - expected)
            print(f"C-BALD mean improvement: {mean_imp:+.4f}")
            print(f"Expected (from CBALD_VARIANTS_IMPROVEMENTS.md): +0.0173")
            print(f"Difference: {diff:.4f}")
            if diff < 0.005:
                print("VERIFICATION PASSED: C-BALD result is consistent!")
            else:
                print("WARNING: C-BALD result differs from expected +0.0173")
                print("This may be due to randomness or environment differences.")
    else:
        # Statistical comparison
        print()
        print("Statistical Comparison (Paired t-test on improvement):")
        print("-" * 70)

        if 'C-BALD' in all_results and 'CBALD_True' in all_results:
            cbald_improvements = all_results['C-BALD']['improvements']
            cbald_true_improvements = all_results['CBALD_True']['improvements']

            t_stat, p_value = stats.ttest_rel(cbald_improvements, cbald_true_improvements)
            mean_diff = np.mean(cbald_improvements) - np.mean(cbald_true_improvements)

            print(f"\nC-BALD vs CBALD_True:")
            print(f"  C-BALD mean:      {np.mean(cbald_improvements):+.4f}")
            print(f"  CBALD_True mean:  {np.mean(cbald_true_improvements):+.4f}")
            print(f"  Mean difference:  {mean_diff:+.4f}")
            print(f"  t-statistic:      {t_stat:.3f}")
            print(f"  p-value:          {p_value:.4f}")

            if p_value < 0.05:
                winner = 'C-BALD' if np.mean(cbald_improvements) > np.mean(cbald_true_improvements) else 'CBALD_True'
                print(f"\n  SIGNIFICANT (p<0.05): {winner} is better!")
            else:
                print(f"\n  NOT SIGNIFICANT (p>=0.05): No clear winner")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
