"""
Testing C-BatchBALD: Making BatchBALD work with C-BALD!

GOAL: Combine C-BALD information value weighting with BatchBALD diversity!

Variants:
1. CBALD-Diverse (Original baseline: +0.0180 ± 0.0064)
2. CBALD-Adaptive (Exploit 90% early → Explore 50% later)
3. CBALD-NoFilter (No pre-filtering bias, scores all samples)
4. CBALD-TwoStage (Pure C-BALD top 50% + diversity 50%)
5. C-BatchBALD (NEW: C-BALD pre-filter + BatchBALD joint entropy!)
6. C-BALD (Reference: +0.0173 ± 0.0062)

Settings: 5 trials for statistical significance, budget=50
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
    cbald_censored_regression,  # C-BALD (reference)
    cbald_diverse_acquire,  # Original winner
    cbald_diverse_adaptive_acquire,  # NEW: Adaptive ratio
    cbald_diverse_nofilter_acquire,  # NEW: No pre-filtering
    cbald_twostage_acquire,  # NEW: Two-stage exploit/explore
    c_batchbald_acquire,  # NEW: C-BatchBALD (C-BALD + BatchBALD diversity)
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


def deep_copy_model(model, config):
    """Create a deep copy of the model.

    Note: BayesLinMtlr increments num_time_bins by 1 in __init__,
    so we need to use num_time_bins - 1 to get the original value.
    """
    # Create new model with same architecture
    # Important: num_time_bins is incremented in __init__, so subtract 1
    original_num_time_bins = model.num_time_bins - 1

    model_copy = BayesLinMtlr(
        in_features=model.in_features,
        num_time_bins=original_num_time_bins,
        config=config
    ).to(config.device)

    # Verify architecture matches before copying
    assert model_copy.num_time_bins == model.num_time_bins, \
        f"Architecture mismatch: {model_copy.num_time_bins} != {model.num_time_bins}"

    # Copy state dict using strict loading
    state_dict = copy.deepcopy(model.state_dict())
    model_copy.load_state_dict(state_dict, strict=True)

    # Set to eval mode like the base model
    if not model.training:
        model_copy.eval()

    return model_copy


# ==================== MAIN COMPARISON ====================

def run_single_trial(
    X_train, y_train, e_train, X_test, y_test, e_test,
    time_bins, config, acquisition_functions,
    initial_samples, budget, increment, prefilter_size, seed
):
    """Run a single trial comparing all acquisition functions.

    CRITICAL: All methods start from the SAME initial model.
    """

    print(f"\n      Setting random seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Apply artificial censoring
    y_labeled, e_labeled, censored_indices = artificially_censor_true(
        y_train.copy(), e_train.copy(), num_initial_samples=initial_samples, seed=seed
    )

    # ===================================================================
    # CRITICAL FIX: Train ONE base model that ALL methods will use
    # ===================================================================
    print(f"      Training SHARED base model for all methods...")
    start_time = time.time()

    base_model = BayesLinMtlr(
        in_features=X_train.shape[1],
        num_time_bins=len(time_bins) - 1,
        config=config
    ).to(config.device)

    base_model = train_model(
        base_model, X_train, y_labeled, e_labeled, time_bins, config,
        epochs=100, patience=20
    )

    # Evaluate the shared base model
    initial_cindex = evaluate_model(base_model, X_test, y_test, e_test, time_bins, config)
    train_time = time.time() - start_time

    print(f"      Shared base model trained ({train_time:.1f}s)")
    print(f"      Shared initial C-index: {initial_cindex:.4f}")
    print(f"      All methods will start from this SAME model!")
    print()

    results = {}

    for acq_func, acq_name in acquisition_functions:
        print(f"      [{acq_name}] Starting acquisition...", flush=True)
        method_start_time = time.time()

        # ===================================================================
        # CRITICAL FIX: Make a deep copy of the base model for this method
        # ===================================================================
        model = deep_copy_model(base_model, config)

        # Verify it starts with approximately the same C-index
        # Note: Bayesian models have stochastic forward passes, so we use a relaxed tolerance
        verify_cindex = evaluate_model(model, X_test, y_test, e_test, time_bins, config)
        tolerance = 5e-3  # 0.005 tolerance for Bayesian stochastic sampling
        assert abs(verify_cindex - initial_cindex) < tolerance, \
            f"Model copy verification failed! {verify_cindex} != {initial_cindex} (diff: {abs(verify_cindex - initial_cindex):.6f})"

        print(f"      [{acq_name}] Initial C-index: {initial_cindex:.4f} (verified same as base)")

        # Get censored pool
        censored_list = list(censored_indices)
        X_censored = X_train[censored_list]

        # Pre-filter for BatchBALD variants (use base model for fair comparison)
        if 'BatchBALD' in acq_name and len(censored_list) > prefilter_size:
            print(f"      [{acq_name}] Pre-filtering {len(censored_list)} → {prefilter_size} samples...", end=' ', flush=True)
            filter_start = time.time()
            # IMPORTANT: Use base_model for pre-filtering, not the method's model
            filter_indices = prefilter_pool(
                base_model, X_censored, y_labeled[censored_list],
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
            elif acq_name == 'CBALD-Diverse':
                # NEW: CBALD-Diverse hybrid
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
                    diversity_ratio=0.3  # Balance C-BALD (70%) + diversity (30%)
                )
            elif 'BatchBALD' in acq_name:
                # BatchBALD variants
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
                # Entropy, Variance, and C-BALD
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

            # Update labels with acquired information (same for all methods)
            y_labeled_copy = y_labeled.copy()
            e_labeled_copy = e_labeled.copy()

            for idx in acquired_indices:
                y_labeled_copy[idx] = min(y_train[idx], y_labeled_copy[idx] + increment)
                if y_labeled_copy[idx] >= y_train[idx]:
                    e_labeled_copy[idx] = e_train[idx]

            # Retrain the method's model with new labels
            print(f"      [{acq_name}] Retraining model...", end=' ', flush=True)
            retrain_start = time.time()

            model = train_model(
                model, X_train, y_labeled_copy, e_labeled_copy, time_bins, config,
                epochs=100, patience=20
            )

            # Final evaluation
            final_cindex = evaluate_model(model, X_test, y_test, e_test, time_bins, config)
            retrain_time = time.time() - retrain_start
            print(f"done ({retrain_time:.1f}s)", flush=True)

            improvement = final_cindex - initial_cindex
            total_time = time.time() - method_start_time

            print(f"      [{acq_name}] Final C-index: {final_cindex:.4f} (Δ={improvement:+.4f}) [Total: {total_time:.1f}s]")
            print()

            results[acq_name] = {
                'initial_cindex': initial_cindex,
                'final_cindex': final_cindex,
                'improvement': improvement,
                'time': total_time
            }

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[acq_name] = None

    return results


def run_comparison(n_trials=5, initial_samples=200, budget=50, increment=60, prefilter_size=500):
    """Run full comparison across multiple trials."""

    print("="*70)
    print("TESTING C-BatchBALD: MAKING BATCHBALD WORK!")
    print("="*70)
    print()
    print("GOAL: Combine C-BALD information value with BatchBALD diversity!")
    print()
    print("Variants:")
    print("  1. CBALD-Diverse (Original - baseline: +0.0180)")
    print("  2. CBALD-Adaptive (Exploit 90% early → Explore 50% later)")
    print("  3. CBALD-NoFilter (No pre-filtering bias)")
    print("  4. CBALD-TwoStage (Pure C-BALD top 50% + diversity 50%)")
    print("  5. C-BatchBALD (NEW: C-BALD + BatchBALD joint entropy!)")
    print("  6. C-BALD (Reference: +0.0173)")
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

    # Define acquisition functions (NOW WITH C-BatchBALD!)
    acquisition_functions = [
        (cbald_diverse_acquire, 'CBALD-Diverse'),           # Original baseline: +0.0180
        (cbald_diverse_adaptive_acquire, 'CBALD-Adaptive'), # NEW: Adaptive ratio 90%→50%
        (cbald_diverse_nofilter_acquire, 'CBALD-NoFilter'), # NEW: No pre-filtering
        (cbald_twostage_acquire, 'CBALD-TwoStage'),         # NEW: Two-stage exploit/explore
        (c_batchbald_acquire, 'C-BatchBALD'),               # NEW: C-BALD + BatchBALD diversity!
        (cbald_censored_regression, 'C-BALD'),              # Reference: +0.0173
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

        print(f"  Trial {trial + 1} completed")
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
            initial_cindices = [r['initial_cindex'] for r in all_results[acq_name]]

            summary[acq_name] = {
                'mean_improvement': np.mean(improvements),
                'std_improvement': np.std(improvements, ddof=1),
                'mean_final': np.mean(final_cindices),
                'std_final': np.std(final_cindices, ddof=1),
                'mean_initial': np.mean(initial_cindices),
                'std_initial': np.std(initial_cindices, ddof=1)
            }

    # Sort by mean improvement (the fair metric!)
    sorted_methods = sorted(summary.items(), key=lambda x: x[1]['mean_improvement'], reverse=True)

    print("Method Rankings (by mean IMPROVEMENT - the fair metric!):")
    print("-" * 70)
    for rank, (name, stats_dict) in enumerate(sorted_methods, 1):
        emoji = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
        print(f"{rank}. {name:<35} {stats_dict['mean_improvement']:+.4f} ± {stats_dict['std_improvement']:.4f} {emoji}")
    print()

    print("Initial C-index (should be SAME for all methods):")
    print("-" * 70)
    for name, stats_dict in summary.items():
        print(f"{name:<35} {stats_dict['mean_initial']:.4f} ± {stats_dict['std_initial']:.4f}")
    print()

    # Statistical tests
    print("Statistical Comparisons (Paired t-tests on improvement):")
    print("-" * 70)

    if 'BatchBALD' in summary and summary['BatchBALD']['mean_improvement']:
        bb_improvements = [r['improvement'] for r in all_results['BatchBALD']]

        for method in ['Variance', 'Entropy', 'C-BALD', 'Random']:
            if method in summary:
                other_improvements = [r['improvement'] for r in all_results[method]]
                t_stat, p_val = stats.ttest_rel(bb_improvements, other_improvements)
                diff = np.mean(bb_improvements) - np.mean(other_improvements)
                sig = "✓ Significant" if p_val < 0.05 else ""
                print(f"\nBatchBALD vs {method}:")
                print(f"  Mean difference: {diff:+.4f}")
                print(f"  t-statistic: {t_stat:.3f}, p-value: {p_val:.4f} {sig}")

    print()

    return all_results, summary


if __name__ == "__main__":
    run_comparison(
        n_trials=5,
        initial_samples=200,
        budget=50,
        increment=60,
        prefilter_size=500
    )
    print("\nBudget=50 test complete!")
