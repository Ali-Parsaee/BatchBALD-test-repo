"""
Compare original vs optimized BatchBALD implementations.

Tests:
1. BatchBALD (original)
2. BatchBALD_optimal (λ₁=0.0) - window weighting only
3. BatchBALD_optimal (λ₁=0.3) - window + variance
4. Variance
5. Entropy
6. Random

Settings: 5 trials, budget=10, prefilter_size=500
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

from Model_stuff.model import BayesLinMtlr, mtlr_survival
from Model_stuff.acquisition import (
    batchbald_acquire_budget,
    entropy_of_probs,
    variance_of_probs,
    random_knapsack
)
from optimized_batchbald import batchbald_optimal


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


def train_model(model, X_train, y_train, e_train, time_bins, config, epochs=100, patience=20):
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

    return model


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


def get_uncertainty_scores(model, X_pool, time_bins, config, device):
    """Get uncertainty scores for pre-filtering."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_pool).to(device)
        logits = model.forward(X_tensor, sample=True, n_samples=config.n_samples_test)
        survival_probs = mtlr_survival(logits, with_sample=True)

    # Compute variance across ensemble
    variance = survival_probs.var(dim=0).sum(dim=1).cpu().numpy()
    return variance


def prefilter_pool(model, X_pool, y_pool, e_pool, time_bins, config, device, top_k=500):
    """Pre-filter pool to top K most uncertain samples."""
    uncertainty = get_uncertainty_scores(model, X_pool, time_bins, config, device)
    top_indices = np.argsort(uncertainty)[-top_k:]
    return top_indices


# ==================== MAIN COMPARISON ====================

def run_single_trial(
    X_train, y_train, e_train, X_test, y_test, e_test,
    time_bins, config, acquisition_functions,
    initial_samples, budget, increment, prefilter_size, seed
):
    """Run a single trial comparing all acquisition functions."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Apply artificial censoring
    y_labeled, e_labeled, censored_indices = artificially_censor_true(
        y_train.copy(), e_train.copy(), num_initial_samples=initial_samples
    )

    results = {}

    for acq_func, acq_name in acquisition_functions:
        print(f"      [{acq_name}] Training base model...", end=' ', flush=True)
        start_time = time.time()

        # Train base model
        model = BayesLinMtlr(
            in_features=X_train.shape[1],
            num_time_bins=len(time_bins) - 1,
            config=config
        ).to(config.device)

        model = train_model(
            model, X_train, y_labeled, e_labeled, time_bins, config,
            epochs=100, patience=20
        )

        # Initial evaluation
        initial_cindex = evaluate_model(model, X_test, y_test, e_test, time_bins, config)
        train_time = time.time() - start_time
        print(f"done ({train_time:.1f}s)", flush=True)
        print(f"      [{acq_name}] Initial C-index: {initial_cindex:.4f}")

        # Get censored pool
        censored_list = list(censored_indices)
        X_censored = X_train[censored_list]

        # Pre-filter for BatchBALD variants
        if 'BatchBALD' in acq_name and len(censored_list) > prefilter_size:
            print(f"      [{acq_name}] Pre-filtering {len(censored_list)} → {prefilter_size} samples...", end=' ', flush=True)
            filter_start = time.time()
            filter_indices = prefilter_pool(
                model, X_censored, y_labeled[censored_list],
                e_labeled[censored_list], time_bins, config, config.device, top_k=prefilter_size
            )
            censored_list = [censored_list[i] for i in filter_indices]
            X_censored = X_train[censored_list]
            filter_time = time.time() - filter_start
            print(f"done ({filter_time:.1f}s)", flush=True)

        # Prepare data for acquisition
        model_data_censored = pd.DataFrame(X_censored)
        model_data_censored["time"] = y_labeled[censored_list]
        model_data_censored["event"] = e_labeled[censored_list]
        costlist = pd.DataFrame([1] * len(X_censored), columns=['Cost'])

        # Run acquisition function
        print(f"      [{acq_name}] Running acquisition (budget={budget})...", end=' ', flush=True)
        acq_start = time.time()

        try:
            if acq_name == 'Random':
                pool_indices = acq_func(costlist, budget)
            elif 'BatchBALD_optimal' in acq_name:
                # Extract lambda values from name
                if 'λ₁=0.0' in acq_name:
                    lambda_var = 0.0
                elif 'λ₁=0.3' in acq_name:
                    lambda_var = 0.3
                else:
                    lambda_var = 0.0

                pool_indices = acq_func(
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
                    num_samples=10000,
                    lambda_variance=lambda_var,
                    lambda_window=2.0,
                    verbose=False
                )
            elif acq_name == 'BatchBALD':
                pool_indices = acq_func(
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
                    num_samples=10000
                )
            else:
                # Entropy and Variance
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
                    budget=budget
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

            model = train_model(
                model, X_train, y_labeled, e_labeled, time_bins, config,
                epochs=100, patience=20
            )

            # Final evaluation
            final_cindex = evaluate_model(model, X_test, y_test, e_test, time_bins, config)
            retrain_time = time.time() - retrain_start
            print(f"done ({retrain_time:.1f}s)", flush=True)

            improvement = final_cindex - initial_cindex
            total_time = time.time() - start_time

            print(f"      [{acq_name}] Final C-index: {final_cindex:.4f} (Δ={improvement:+.4f}) [Total: {total_time:.1f}s]")

            results[acq_name] = {
                'initial_cindex': initial_cindex,
                'final_cindex': final_cindex,
                'improvement': improvement,
                'time': total_time
            }

        except Exception as e:
            print(f"ERROR: {e}")
            results[acq_name] = None

    return results


def run_comparison(n_trials=5, initial_samples=200, budget=10, increment=60, prefilter_size=500):
    """Run full comparison across multiple trials."""

    print("="*70)
    print("OPTIMIZED BATCHBALD COMPARISON")
    print("="*70)
    print()
    print("Methods:")
    print("  1. BatchBALD (original)")
    print("  2. BatchBALD_optimal (λ₁=0.0) - window weighting only")
    print("  3. BatchBALD_optimal (λ₁=0.3) - window + variance")
    print("  4. Variance")
    print("  5. Entropy")
    print("  6. Random")
    print()
    print("Settings:")
    print(f"  Trials: {n_trials}")
    print(f"  Budget: {budget}")
    print(f"  Pre-filter size: {prefilter_size}")
    print(f"  Increment: {increment}")
    print()

    # Load data
    print("Loading NACD dataset...")
    nacd_path = "data/MIMIC/NACD/NACD_Full.csv"
    data = pd.read_csv(nacd_path)

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

    X = data.drop(["time", "event"], axis=1).values
    y = data["time"].values
    e = data["event"].values

    np.random.seed(42)
    X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(
        X, y, e, test_size=0.1, random_state=42, stratify=e
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    num_bins = 10
    event_times = y_train[e_train == 1]
    quantiles = np.linspace(0, 1, num_bins + 1)[1:]
    time_bins = np.quantile(event_times, quantiles)
    time_bins[-1] *= 1.05
    time_bins = np.array([0] + list(time_bins))

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

    print(f"   Dataset shape: {data.shape}")
    print(f"   Device: {config.device}")
    print()

    # Define acquisition functions
    acquisition_functions = [
        (batchbald_acquire_budget, 'BatchBALD'),
        (batchbald_optimal, 'BatchBALD_optimal (λ₁=0.0)'),
        (batchbald_optimal, 'BatchBALD_optimal (λ₁=0.3)'),
        (variance_of_probs, 'Variance'),
        (entropy_of_probs, 'Entropy'),
        (random_knapsack, 'Random')
    ]

    # Store results
    all_results = {name: [] for _, name in acquisition_functions}

    print(f"Running {n_trials} trials...")
    print()

    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials}")
        print("  " + "-" * 68)

        seed = 100 + trial
        trial_results = run_single_trial(
            X_train, y_train, e_train, X_test, y_test, e_test,
            time_bins, config, acquisition_functions,
            initial_samples, budget, increment, prefilter_size, seed
        )

        for acq_name, result in trial_results.items():
            if result is not None:
                all_results[acq_name].append(result)

        print(f"  Trial {trial + 1} completed in {sum(r['time'] for r in trial_results.values() if r) / 60:.1f} minutes")
        print()

    # Compute statistics
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()

    summary = {}
    for acq_name in all_results:
        if len(all_results[acq_name]) > 0:
            improvements = [r['improvement'] for r in all_results[acq_name]]
            final_cindices = [r['final_cindex'] for r in all_results[acq_name]]

            summary[acq_name] = {
                'mean_improvement': np.mean(improvements),
                'std_improvement': np.std(improvements),
                'mean_final': np.mean(final_cindices),
                'std_final': np.std(final_cindices)
            }

    # Sort by mean final c-index
    sorted_methods = sorted(summary.items(), key=lambda x: x[1]['mean_final'], reverse=True)

    print("Method Rankings (by mean final C-index):")
    print("-" * 70)
    for rank, (name, stats) in enumerate(sorted_methods, 1):
        emoji = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
        print(f"{rank}. {name:<35} {stats['mean_final']:.4f} (Δ={stats['mean_improvement']:+.4f}) {emoji}")
    print()

    # Statistical tests
    print("Statistical Comparisons:")
    print("-" * 70)

    # Compare BatchBALD variants
    if len(all_results['BatchBALD']) > 0 and len(all_results['BatchBALD_optimal (λ₁=0.0)']) > 0:
        bb_orig = [r['final_cindex'] for r in all_results['BatchBALD']]
        bb_opt_window = [r['final_cindex'] for r in all_results['BatchBALD_optimal (λ₁=0.0)']]
        t_stat, p_val = stats.ttest_rel(bb_opt_window, bb_orig)
        diff = np.mean(bb_opt_window) - np.mean(bb_orig)
        print(f"BatchBALD_optimal (λ₁=0.0) vs BatchBALD:")
        print(f"  Mean difference: {diff:+.4f}, t={t_stat:.3f}, p={p_val:.4f}")

    if len(all_results['BatchBALD_optimal (λ₁=0.0)']) > 0 and len(all_results['BatchBALD_optimal (λ₁=0.3)']) > 0:
        bb_opt_window = [r['final_cindex'] for r in all_results['BatchBALD_optimal (λ₁=0.0)']]
        bb_opt_both = [r['final_cindex'] for r in all_results['BatchBALD_optimal (λ₁=0.3)']]
        t_stat, p_val = stats.ttest_rel(bb_opt_both, bb_opt_window)
        diff = np.mean(bb_opt_both) - np.mean(bb_opt_window)
        print(f"BatchBALD_optimal (λ₁=0.3) vs BatchBALD_optimal (λ₁=0.0):")
        print(f"  Mean difference: {diff:+.4f}, t={t_stat:.3f}, p={p_val:.4f}")

    if len(all_results['BatchBALD_optimal (λ₁=0.3)']) > 0 and len(all_results['Variance']) > 0:
        bb_opt_both = [r['final_cindex'] for r in all_results['BatchBALD_optimal (λ₁=0.3)']]
        variance_scores = [r['final_cindex'] for r in all_results['Variance']]
        t_stat, p_val = stats.ttest_ind(bb_opt_both, variance_scores)
        diff = np.mean(bb_opt_both) - np.mean(variance_scores)
        print(f"BatchBALD_optimal (λ₁=0.3) vs Variance:")
        print(f"  Mean difference: {diff:+.4f}, t={t_stat:.3f}, p={p_val:.4f}")

    print()

    return all_results, summary


if __name__ == "__main__":
    run_comparison(
        n_trials=5,
        initial_samples=200,
        budget=10,
        increment=60,
        prefilter_size=500
    )
    print("\nComparison complete!")
