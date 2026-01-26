"""
FAST version: Optimized acquisition function comparison.

Optimizations:
1. SampledJointEntropy instead of ExactJointEntropy (10-50× faster)
2. Pre-filter to top 500 uncertain samples (4× faster)
3. Reduced trials (5) and budget (10) (6× faster)
4. Better progress printing

Expected runtime: ~30-45 minutes instead of 5+ hours
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import argparse
from scipy import stats
from typing import Dict, List, Tuple
import time

# Import from Model_stuff
from Model_stuff.model import BayesLinMtlr, mtlr_survival
from Model_stuff.acquisition import (
    batchbald_acquire_budget,
    entropy_of_probs,
    variance_of_probs,
    random_knapsack
)


# Define helper functions inline
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


def artificially_censor_true(times, events, num_initial_samples=50):
    """Artificially censor data points."""
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

    for i in range(len(y_test)):
        for j in range(i + 1, len(y_test)):
            if (y_test[i] < y_test[j] and cens[i] == 1) or (y_test[j] < y_test[i] and cens[j] == 1):
                n += 1
                if y_test[i] < y_test[j]:
                    if y_pred[i] > y_pred[j]:
                        n_concordant += 1
                else:
                    if y_pred[j] > y_pred[i]:
                        n_concordant += 1

    if n == 0:
        return 0.0
    return n_concordant / n


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


def get_uncertainty_scores(model, X_pool, time_bins, config, device):
    """Get uncertainty scores for pre-filtering (using entropy)."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_pool).to(device)
        logits = model.forward(X_tensor, sample=True, n_samples=config.n_samples_test)
        survival_probs = mtlr_survival(logits, with_sample=True)

        # Calculate entropy as uncertainty measure
        mean_probs = survival_probs.mean(dim=0).cpu().numpy()
        entropy = -(mean_probs * np.log(mean_probs + 1e-12)).sum(axis=1)

    return entropy


def prefilter_pool(model, X_pool, y_pool, e_pool, time_bins, config, device, top_k=500):
    """
    Pre-filter pool to top K most uncertain samples.
    This dramatically speeds up BatchBALD.
    """
    if len(X_pool) <= top_k:
        return np.arange(len(X_pool))

    # Get uncertainty scores
    uncertainty = get_uncertainty_scores(model, X_pool, time_bins, config, device)

    # Select top K most uncertain
    top_indices = np.argsort(uncertainty)[-top_k:]

    return top_indices


def train_model(model, X_train, y_train, e_train, time_bins, config, epochs=100, patience=20, verbose=False):
    """Train Bayesian Linear MTLR model."""
    train_df = pd.DataFrame(X_train)
    train_df['time'] = y_train
    train_df['event'] = e_train
    x_formatted, y_formatted = reformat_survival(train_df, time_bins[1:])

    train_dataset = TensorDataset(x_formatted, y_formatted)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for xi, yi in train_loader:
            xi, yi = xi.to(config.device), yi.to(config.device)
            optimizer.zero_grad()
            loss, _, _, _ = model.sample_elbo(xi, yi, len(X_train), see1=config.c1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model, best_loss


def evaluate_model(model, X_test, y_test, e_test, time_bins, config):
    """Evaluate model and return c-index."""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(config.device)
        logits = model.forward(X_test_tensor, sample=True, n_samples=config.n_samples_test)
        survival_probs = mtlr_survival(logits, with_sample=True)
        mean_survival = survival_probs.mean(dim=0).cpu().numpy()

    pred_times = expected_times_from_survival(mean_survival, time_bins)
    c_index = concordance(-pred_times, y_test, e_test)
    return c_index


def run_single_trial(
    seed: int,
    acquisition_function,
    acq_name: str,
    X_train, y_train, e_train,
    X_test, y_test, e_test,
    time_bins,
    config,
    initial_samples: int = 200,
    budget: int = 10,
    increment: int = 60,
    verbose: bool = False,
    prefilter_size: int = 500
):
    """Run a single trial for one acquisition function."""

    start_time = time.time()

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Apply artificial censoring
    y_labeled, e_labeled, censored_indices = artificially_censor_true(
        y_train.copy(), e_train.copy(), num_initial_samples=initial_samples
    )

    # Train base model
    print(f"      [{acq_name}] Training base model...", end=' ', flush=True)
    train_start = time.time()
    model = BayesLinMtlr(
        in_features=X_train.shape[1],
        num_time_bins=len(time_bins) - 1,
        config=config
    ).to(config.device)

    model, _ = train_model(
        model, X_train, y_labeled, e_labeled, time_bins, config,
        epochs=100, patience=20, verbose=False
    )
    train_time = time.time() - train_start
    print(f"done ({train_time:.1f}s)", flush=True)

    # Evaluate initial model
    initial_cindex = evaluate_model(model, X_test, y_test, e_test, time_bins, config)
    print(f"      [{acq_name}] Initial C-index: {initial_cindex:.4f}", flush=True)

    # Get censored pool
    censored_list = list(censored_indices)
    X_censored = X_train[censored_list]

    # PRE-FILTER: Select top uncertain samples
    if acq_name == 'BatchBALD' and len(censored_list) > prefilter_size:
        print(f"      [{acq_name}] Pre-filtering {len(censored_list)} → {prefilter_size} samples...", end=' ', flush=True)
        filter_start = time.time()

        # Get top uncertain samples from censored pool
        filter_indices = prefilter_pool(
            model, X_censored, y_labeled[censored_list], e_labeled[censored_list],
            time_bins, config, config.device, top_k=prefilter_size
        )

        # Update censored_list to only include filtered samples
        censored_list = [censored_list[i] for i in filter_indices]
        X_censored = X_train[censored_list]

        filter_time = time.time() - filter_start
        print(f"done ({filter_time:.1f}s)", flush=True)

    # Prepare censored data
    model_data_censored = pd.DataFrame(X_censored)
    model_data_censored["time"] = y_labeled[censored_list]
    model_data_censored["event"] = e_labeled[censored_list]

    # Create cost list
    costlist = pd.DataFrame([1] * len(X_censored), columns=['Cost'])

    # Run acquisition function
    print(f"      [{acq_name}] Running acquisition (budget={budget})...", end=' ', flush=True)
    acq_start = time.time()

    try:
        if acq_name == 'Random':
            pool_indices = acquisition_function(costlist, budget)
        else:
            # Use SAMPLED joint entropy (much faster than exact)
            pool_indices, _ = acquisition_function(
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
                num_samples=10000  # Use sampled (not exact!) - 10-50× faster
            )

        acquired_indices = [censored_list[i] for i in pool_indices]
        acq_time = time.time() - acq_start
        print(f"done ({acq_time:.1f}s, acquired {len(acquired_indices)} samples)", flush=True)

        # Update labels with acquired information
        for idx in acquired_indices:
            y_labeled[idx] = min(y_train[idx], y_labeled[idx] + increment)
            if y_labeled[idx] >= y_train[idx]:
                e_labeled[idx] = e_train[idx]

        # Retrain model
        print(f"      [{acq_name}] Retraining model...", end=' ', flush=True)
        retrain_start = time.time()
        model = BayesLinMtlr(
            in_features=X_train.shape[1],
            num_time_bins=len(time_bins) - 1,
            config=config
        ).to(config.device)

        model, _ = train_model(
            model, X_train, y_labeled, e_labeled, time_bins, config,
            epochs=100, patience=20, verbose=False
        )
        retrain_time = time.time() - retrain_start
        print(f"done ({retrain_time:.1f}s)", flush=True)

        # Evaluate final model
        final_cindex = evaluate_model(model, X_test, y_test, e_test, time_bins, config)

        total_time = time.time() - start_time
        improvement = final_cindex - initial_cindex

        print(f"      [{acq_name}] Final C-index: {final_cindex:.4f} (Δ={improvement:+.4f}) [Total: {total_time:.1f}s]", flush=True)

        return {
            'initial_cindex': initial_cindex,
            'final_cindex': final_cindex,
            'improvement': improvement,
            'success': True,
            'time': total_time
        }

    except Exception as e:
        print(f"\n      [{acq_name}] ERROR: {str(e)}", flush=True)
        return {
            'initial_cindex': initial_cindex,
            'final_cindex': initial_cindex,
            'improvement': 0.0,
            'success': False,
            'time': time.time() - start_time
        }


def run_comparison(
    n_trials: int = 5,
    initial_samples: int = 200,
    budget: int = 10,
    increment: int = 60,
    prefilter_size: int = 500,
    verbose: bool = True
):
    """Run optimized comparison across multiple trials."""

    print("="*70)
    print("FAST ACQUISITION FUNCTION COMPARISON")
    print("="*70)
    print(f"\nOptimizations enabled:")
    print(f"  • SampledJointEntropy (10-50× faster than exact)")
    print(f"  • Pre-filter to top {prefilter_size} uncertain samples (4× faster)")
    print(f"  • Reduced trials and budget")
    print(f"\nSettings:")
    print(f"  Dataset: NACD")
    print(f"  Initial labeled samples: {initial_samples}")
    print(f"  Acquisition budget: {budget}")
    print(f"  Probe increment: {increment}")
    print(f"  Number of trials: {n_trials}")
    print(f"  Pre-filter pool size: {prefilter_size}")
    print()

    comparison_start = time.time()

    # Load data
    print("1. Loading NACD dataset...")
    nacd_path = "data/MIMIC/NACD/NACD_Full.csv"
    if not os.path.exists(nacd_path):
        raise FileNotFoundError(f"NACD file not found at {nacd_path}")

    data = pd.read_csv(nacd_path)

    # Preprocess NACD data
    cols_to_drop = ['PERFORMANCE_STATUS', 'STAGE_NUMERICAL', 'AGE65']
    data = data.drop([c for c in cols_to_drop if c in data.columns], axis=1)

    if "CENSORED" in data.columns:
        data["event"] = 1 - data["CENSORED"]
        data = data.drop(columns=["CENSORED"])
    if "SURVIVAL" in data.columns:
        data = data.rename(columns={"SURVIVAL": "time"})

    cols_standardize = ['BOX1_SCORE', 'BOX2_SCORE', 'BOX3_SCORE', 'BMI', 'WEIGHT_CHANGEPOINT',
                        'AGE', 'GRANULOCYTES', 'LDH_SERUM', 'LYMPHOCYTES',
                        'PLATELET', 'WBC_COUNT', 'CALCIUM_SERUM', 'HGB', 'CREATININE_SERUM', 'ALBUMIN']
    cols_standardize = [c for c in cols_standardize if c in data.columns]
    data[cols_standardize] = data[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())

    print(f"   Dataset shape: {data.shape}")

    # Prepare data
    X = data.drop(["time", "event"], axis=1).values
    y = data["time"].values
    e = data["event"].values

    # Train/test split (fixed across all trials)
    np.random.seed(42)
    X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(
        X, y, e, test_size=0.1, random_state=42, stratify=e
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create time bins
    num_bins = 10
    event_times = y_train[e_train == 1]
    quantiles = np.linspace(0, 1, num_bins + 1)[1:]
    time_bins = np.quantile(event_times, quantiles)
    time_bins[-1] *= 1.05
    time_bins = np.array([0] + list(time_bins))

    # Create config
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

    print(f"\n2. Running {n_trials} trials for each acquisition function...")
    print()

    # Define acquisition functions to test
    acquisition_functions = [
        (batchbald_acquire_budget, 'BatchBALD'),
        (entropy_of_probs, 'Entropy'),
        (variance_of_probs, 'Variance'),
        (random_knapsack, 'Random')
    ]

    # Store results
    results = {name: [] for _, name in acquisition_functions}

    # Run trials
    for trial in range(n_trials):
        trial_start = time.time()
        print(f"  Trial {trial + 1}/{n_trials}")
        print(f"  " + "-"*66)
        seed = 100 + trial

        for acq_func, acq_name in acquisition_functions:
            result = run_single_trial(
                seed=seed,
                acquisition_function=acq_func,
                acq_name=acq_name,
                X_train=X_train,
                y_train=y_train,
                e_train=e_train,
                X_test=X_test,
                y_test=y_test,
                e_test=e_test,
                time_bins=time_bins,
                config=config,
                initial_samples=initial_samples,
                budget=budget,
                increment=increment,
                verbose=verbose,
                prefilter_size=prefilter_size
            )
            results[acq_name].append(result)

        trial_time = time.time() - trial_start
        print(f"  Trial {trial + 1} completed in {trial_time/60:.1f} minutes")
        print()

    # Print results
    total_time = time.time() - comparison_start

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print()

    # Summary statistics
    print("Summary Statistics:")
    print("-" * 70)
    print(f"{'Method':<15} {'Initial C-Index':<20} {'Final C-Index':<20} {'Improvement':<15}")
    print("-" * 70)

    summary = {}
    for name in results:
        successful_trials = [r for r in results[name] if r['success']]
        if len(successful_trials) > 0:
            initial_mean = np.mean([r['initial_cindex'] for r in successful_trials])
            initial_std = np.std([r['initial_cindex'] for r in successful_trials])
            final_mean = np.mean([r['final_cindex'] for r in successful_trials])
            final_std = np.std([r['final_cindex'] for r in successful_trials])
            improvement_mean = np.mean([r['improvement'] for r in successful_trials])
            improvement_std = np.std([r['improvement'] for r in successful_trials])

            summary[name] = {
                'initial': [r['initial_cindex'] for r in successful_trials],
                'final': [r['final_cindex'] for r in successful_trials],
                'improvement': [r['improvement'] for r in successful_trials],
                'n_success': len(successful_trials)
            }

            print(f"{name:<15} {initial_mean:.4f} ± {initial_std:.4f}    "
                  f"{final_mean:.4f} ± {final_std:.4f}    "
                  f"{improvement_mean:+.4f} ± {improvement_std:.4f}")

    # Statistical significance testing
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTING")
    print("="*70)
    print()

    # Pairwise comparisons using t-test
    methods = list(summary.keys())
    print("Pairwise t-tests (final C-index):")
    print("-" * 70)

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1, method2 = methods[i], methods[j]
            data1 = summary[method1]['final']
            data2 = summary[method2]['final']

            # Two-tailed t-test
            t_stat, p_value = stats.ttest_ind(data1, data2)

            # Effect size (Cohen's d)
            mean1, mean2 = np.mean(data1), np.mean(data2)
            std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
            pooled_std = np.sqrt(((len(data1) - 1) * std1**2 + (len(data2) - 1) * std2**2) /
                                (len(data1) + len(data2) - 2))
            cohens_d = (mean1 - mean2) / pooled_std

            significance = ""
            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"

            print(f"{method1} vs {method2}:")
            print(f"  t-statistic: {t_stat:+.4f}, p-value: {p_value:.4f} {significance}")
            print(f"  Effect size (Cohen's d): {cohens_d:+.4f}")
            print(f"  Mean difference: {mean1 - mean2:+.4f}")
            print()

    # Ranking
    print("="*70)
    print("FINAL RANKING (by mean final C-index)")
    print("="*70)

    rankings = [(name, np.mean(summary[name]['final'])) for name in summary]
    rankings.sort(key=lambda x: x[1], reverse=True)

    for rank, (name, mean_cindex) in enumerate(rankings, 1):
        improvement = np.mean(summary[name]['improvement'])
        marker = " 🏆" if rank == 1 else ""
        print(f"{rank}. {name:<15} Final C-index: {mean_cindex:.4f}, "
              f"Improvement: {improvement:+.4f}{marker}")

    print()
    print("Significance levels: * p<0.05, ** p<0.01, *** p<0.001")
    print(f"\nTotal runtime: {total_time/60:.1f} minutes")
    print()

    return results, summary


if __name__ == "__main__":
    import argparse as arg_parser

    parser = arg_parser.ArgumentParser(description='Fast acquisition function comparison')
    parser.add_argument('--n_trials', type=int, default=5, help='Number of trials to run')
    parser.add_argument('--initial_samples', type=int, default=200, help='Initial labeled samples')
    parser.add_argument('--budget', type=int, default=10, help='Acquisition budget')
    parser.add_argument('--increment', type=int, default=60, help='Probe increment')
    parser.add_argument('--prefilter_size', type=int, default=500, help='Pre-filter pool size')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    results, summary = run_comparison(
        n_trials=args.n_trials,
        initial_samples=args.initial_samples,
        budget=args.budget,
        increment=args.increment,
        prefilter_size=args.prefilter_size,
        verbose=args.verbose
    )
