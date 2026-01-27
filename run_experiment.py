#!/usr/bin/env python
"""
Flexible Active Learning Experiment Runner for Survival Analysis

Run custom experiments with any combination of:
- Acquisition functions (C-BALD, CBALD-Diverse, BatchBALD, Variance, etc.)
- Budget sizes (number of samples to acquire)
- Increments (survival time window)
- Number of trials

Usage:
    # Run with defaults (C-BALD, budget=20, 5 trials)
    python run_experiment.py

    # Test CBALD-Diverse vs C-BALD with budget=30
    python run_experiment.py --methods cbald_diverse c_bald --budget 30

    # Run all methods with budget=50, 10 trials
    python run_experiment.py --methods all --budget 50 --trials 10

    # Custom configuration
    python run_experiment.py --methods cbald_adaptive cbald_twostage --budget 20 --increment 60 --trials 3

Available acquisition functions:
    - cbald_diverse: CBALD-Diverse (original winner, +0.0180)
    - cbald_adaptive: CBALD-Adaptive (adaptive ratio 90%→50%)
    - cbald_nofilter: CBALD-NoFilter (no pre-filtering bias)
    - cbald_twostage: CBALD-TwoStage (exploit then explore)
    - c_bald: C-BALD (reference, +0.0173)
    - batchbald: BatchBALD (original)
    - variance: Variance-based acquisition
    - entropy: Entropy-based acquisition
    - random: Random selection baseline
    - all: Run all available methods
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'Model_stuff')

import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import time
import copy

from Model_stuff.model import BayesLinMtlr, mtlr_survival
from Model_stuff.acquisition import (
    cbald_censored_regression,
    cbald_diverse_acquire,
    cbald_diverse_adaptive_acquire,
    cbald_diverse_nofilter_acquire,
    cbald_twostage_acquire,
    batchbald_acquire_budget,
    variance_of_probs,
    entropy_of_probs,
    random_knapsack
)


# ==================== ACQUISITION FUNCTION REGISTRY ====================

ACQUISITION_FUNCTIONS = {
    'cbald_diverse': (cbald_diverse_acquire, 'CBALD-Diverse', {}),
    'cbald_adaptive': (cbald_diverse_adaptive_acquire, 'CBALD-Adaptive', {'start_ratio': 0.1, 'end_ratio': 0.5}),
    'cbald_nofilter': (cbald_diverse_nofilter_acquire, 'CBALD-NoFilter', {'diversity_ratio': 0.3}),
    'cbald_twostage': (cbald_twostage_acquire, 'CBALD-TwoStage', {'exploit_ratio': 0.5, 'diversity_ratio': 0.3}),
    'c_bald': (cbald_censored_regression, 'C-BALD', {}),
    'batchbald': (batchbald_acquire_budget, 'BatchBALD', {}),
    'variance': (variance_of_probs, 'Variance', {}),
    'entropy': (entropy_of_probs, 'Entropy', {}),
    'random': (random_knapsack, 'Random', {}),
}


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
    left_edges = np.concatenate([[0.0], tb[:-1]])
    right_edges = tb
    mids = (left_edges + right_edges) / 2.0
    pdf = np.zeros_like(surv_np)
    pdf[:, 0] = 1.0 - surv_np[:, 0]
    pdf[:, 1:] = surv_np[:, :-1] - surv_np[:, 1:]
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


def train_model(model, train_loader, config, epochs=100, patience=10):
    """Train Bayesian MTLR model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)

            optimizer.zero_grad()
            loss, _, _, _ = model.sample_elbo(batch_x, batch_y, len(train_loader.dataset), see1=config.c1)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model


def evaluate_model(model, x_test, y_test, e_test, time_bins, config):
    """Evaluate model and return c-index."""
    model.eval()
    with torch.no_grad():
        logits = model.forward(x_test, sample=True, n_samples=config.n_samples_test)
        survival_probs = mtlr_survival(logits, with_sample=True)
        mean_survival = survival_probs.mean(dim=0).cpu().numpy()

    pred_times = expected_times_from_survival(mean_survival, time_bins)
    c_index = concordance(-pred_times, y_test, e_test)
    return c_index


# ==================== MAIN EXPERIMENT ====================

def run_experiment(methods, budget=20, increment=60, initial_samples=50,
                   trials=5, prefilter_size=500, random_seed=42):
    """Run active learning experiment with specified configuration."""

    print("="*70)
    print("ACTIVE LEARNING EXPERIMENT FOR SURVIVAL ANALYSIS")
    print("="*70)
    print()
    print(f"Configuration:")
    print(f"  Methods: {', '.join([ACQUISITION_FUNCTIONS[m][1] for m in methods])}")
    print(f"  Budget: {budget} samples")
    print(f"  Increment: {increment}")
    print(f"  Initial samples: {initial_samples}")
    print(f"  Trials: {trials}")
    print(f"  Pre-filter size: {prefilter_size}")
    print(f"  Random seed: {random_seed}")
    print()

    # Load data
    print("Loading NACD dataset...")
    df = pd.read_csv('data/MIMIC/NACD/NACD_Full.csv')

    # Handle column names
    if "CENSORED" in df.columns:
        df["event"] = 1 - df["CENSORED"]
        df = df.drop(columns=["CENSORED"])
    if "SURVIVAL" in df.columns:
        df = df.rename(columns={"SURVIVAL": "time"})

    # Drop optional columns if present
    cols_to_drop = ['PERFORMANCE_STATUS', 'STAGE_NUMERICAL', 'AGE65']
    df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)

    # Standardize numerical columns
    cols_standardize = ['BOX1_SCORE', 'BOX2_SCORE', 'BOX3_SCORE', 'BMI', 'WEIGHT_CHANGEPOINT',
                        'AGE', 'GRANULOCYTES', 'LDH_SERUM', 'LYMPHOCYTES',
                        'PLATELET', 'WBC_COUNT', 'CALCIUM_SERUM', 'HGB', 'CREATININE_SERUM', 'ALBUMIN']
    cols_standardize = [c for c in cols_standardize if c in df.columns]
    df[cols_standardize] = df[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())

    X = df.drop(["time", "event"], axis=1).values
    y = df["time"].values
    e = df["event"].values

    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    print()

    # Train/test split
    np.random.seed(random_seed)
    X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(
        X, y, e, test_size=0.1, random_state=random_seed, stratify=e
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Setup time bins
    num_bins = 10
    event_times = y_train[e_train == 1]
    quantiles = np.linspace(0, 1, num_bins + 1)[1:]
    time_bins = np.quantile(event_times, quantiles)
    time_bins[-1] *= 1.05
    time_bins = np.array([0] + list(time_bins))

    # Config
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
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.patience = 10

    print(f"Device: {config.device}")
    print()

    # Store results
    all_results = {ACQUISITION_FUNCTIONS[m][1]: [] for m in methods}

    print(f"Running {trials} trials...")
    print()

    for trial in range(trials):
        print(f"  Trial {trial + 1}/{trials}")
        print("  " + "-" * 68)

        seed = 100 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Artificially censor
        y_censored, e_censored, censored_indices = artificially_censor_true(
            y_train, e_train, num_initial_samples=initial_samples, seed=seed
        )

        # Prepare data
        train_df = pd.DataFrame(X_train)
        train_df['time'] = y_censored
        train_df['event'] = e_censored
        test_df = pd.DataFrame(X_test)
        test_df['time'] = y_test
        test_df['event'] = e_test

        x_train, y_train_encoded = reformat_survival(train_df, time_bins[1:])
        x_test, y_test_encoded = reformat_survival(test_df, time_bins[1:])

        train_dataset = TensorDataset(x_train, y_train_encoded)
        test_dataset = TensorDataset(x_test, y_test_encoded)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        # Train shared base model
        print(f"      Training shared base model...")
        shared_model = BayesLinMtlr(
            in_features=X_train.shape[1],
            num_time_bins=len(time_bins),
            config=config
        )
        shared_model = train_model(shared_model, train_loader, config)

        initial_cindex = evaluate_model(shared_model, x_test, y_test, e_test, time_bins, config)
        print(f"      Initial C-index: {initial_cindex:.4f}")

        # Setup for acquisition
        X_censored = X_train[censored_indices]
        y_original = y_train[censored_indices]
        e_original = e_train[censored_indices]

        model_data = {
            'x_train': x_train,
            'y_train': y_train_encoded,
            'x_test': x_test,
            'y_test': y_test_encoded,
            'time_bins': time_bins
        }

        costlist = np.ones(len(X_censored))

        # Test each acquisition function
        for method_key in methods:
            acq_func, acq_name, acq_params = ACQUISITION_FUNCTIONS[method_key]

            print(f"      [{acq_name}] Running acquisition...")

            # Clone model
            model = copy.deepcopy(shared_model)

            # Run acquisition
            start_time = time.time()
            pool_indices, _ = acq_func(
                model=model,
                X_pool=X_censored,
                batch_size=budget,
                time_bins=time_bins,
                config=config,
                device=config.device,
                in_data_train=model_data,
                increment=increment,
                costlist=costlist,
                budget=budget,
                **acq_params
            )

            # Reveal oracle labels
            revealed_indices = censored_indices[pool_indices]
            y_updated = np.copy(y_censored)
            e_updated = np.copy(e_censored)
            y_updated[revealed_indices] = y_original[pool_indices]
            e_updated[revealed_indices] = e_original[pool_indices]

            # Retrain
            train_df_updated = pd.DataFrame(X_train)
            train_df_updated['time'] = y_updated
            train_df_updated['event'] = e_updated
            x_train_updated, y_train_updated = reformat_survival(train_df_updated, time_bins[1:])

            train_dataset_updated = TensorDataset(x_train_updated, y_train_updated)
            train_loader_updated = DataLoader(train_dataset_updated, batch_size=config.batch_size, shuffle=True)

            model = BayesLinMtlr(
                in_features=X_train.shape[1],
                num_time_bins=len(time_bins),
                config=config
            )
            model = train_model(model, train_loader_updated, config)

            # Evaluate
            final_cindex = evaluate_model(model, x_test, y_test, e_test, time_bins, config)
            improvement = final_cindex - initial_cindex
            elapsed = time.time() - start_time

            print(f"      [{acq_name}] Final C-index: {final_cindex:.4f} (Δ={improvement:+.4f}) [{elapsed:.1f}s]")

            all_results[acq_name].append({
                'initial': initial_cindex,
                'final': final_cindex,
                'improvement': improvement
            })

        print(f"  Trial {trial + 1} completed")
        print()

    # Print results
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()

    # Compute statistics
    method_stats = {}
    for method_name, results in all_results.items():
        improvements = [r['improvement'] for r in results]
        method_stats[method_name] = {
            'mean': np.mean(improvements),
            'std': np.std(improvements, ddof=1) if len(improvements) > 1 else 0
        }

    # Sort by mean improvement
    sorted_methods = sorted(method_stats.items(), key=lambda x: x[1]['mean'], reverse=True)

    print("Rankings (by mean improvement):")
    print("-" * 70)
    medals = ['🏆', '🥈', '🥉']
    for i, (method, stats) in enumerate(sorted_methods):
        medal = medals[i] if i < 3 else ''
        print(f"{i+1}. {method:30s} {stats['mean']:+.4f} ± {stats['std']:.4f} {medal}")

    print()
    print("Experiment complete!")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Run active learning experiments for survival analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--methods', nargs='+', default=['cbald_diverse', 'c_bald'],
                        choices=list(ACQUISITION_FUNCTIONS.keys()) + ['all'],
                        help='Acquisition methods to test (default: cbald_diverse c_bald)')
    parser.add_argument('--budget', type=int, default=20,
                        help='Number of samples to acquire (default: 20)')
    parser.add_argument('--increment', type=int, default=60,
                        help='Survival time window increment (default: 60)')
    parser.add_argument('--initial-samples', type=int, default=50,
                        help='Number of initial uncensored samples (default: 50)')
    parser.add_argument('--trials', type=int, default=5,
                        help='Number of trials to run (default: 5)')
    parser.add_argument('--prefilter-size', type=int, default=500,
                        help='Pre-filter pool size (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--list-methods', action='store_true',
                        help='List all available acquisition methods and exit')

    args = parser.parse_args()

    if args.list_methods:
        print("Available acquisition methods:")
        print()
        for key, (_, name, _) in ACQUISITION_FUNCTIONS.items():
            print(f"  {key:20s} - {name}")
        print()
        print("  all                  - Run all available methods")
        return

    # Handle 'all' option
    methods = args.methods
    if 'all' in methods:
        methods = list(ACQUISITION_FUNCTIONS.keys())

    run_experiment(
        methods=methods,
        budget=args.budget,
        increment=args.increment,
        initial_samples=args.initial_samples,
        trials=args.trials,
        prefilter_size=args.prefilter_size,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()
