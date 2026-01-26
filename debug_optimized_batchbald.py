"""
Debug optimized BatchBALD and analyze why hypotheses failed.

This script will:
1. Check if window weighting is being applied correctly
2. Check if variance boost is being applied correctly
3. Compare characteristics of selected samples across methods
4. Identify what's different from the investigation
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'Model_stuff')

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from torch.utils.data import DataLoader, TensorDataset
import argparse

from Model_stuff.model import BayesLinMtlr, mtlr_survival
from Model_stuff.acquisition import batchbald_acquire_budget
from optimized_batchbald import batchbald_optimal


# ==================== HELPER FUNCTIONS ====================

def encode_survival(time, event, bins):
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
    x = torch.tensor(dataset.drop(["time", "event"], axis=1).values, dtype=torch.float)
    y = encode_survival(dataset["time"].values, dataset["event"].values, time_bins)
    return x, y


def artificially_censor_true(times, events, num_initial_samples=50):
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


def train_model(model, X_train, y_train, e_train, time_bins, config, epochs=100, patience=20):
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


def get_uncertainty_scores(model, X_pool, time_bins, config, device):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_pool).to(device)
        logits = model.forward(X_tensor, sample=True, n_samples=config.n_samples_test)
        survival_probs = mtlr_survival(logits, with_sample=True)

    variance = survival_probs.var(dim=0).sum(dim=1).cpu().numpy()
    return variance


def prefilter_pool(model, X_pool, y_pool, e_pool, time_bins, config, device, top_k=500):
    uncertainty = get_uncertainty_scores(model, X_pool, time_bins, config, device)
    top_indices = np.argsort(uncertainty)[-top_k:]
    return top_indices


def compute_selection_characteristics(model, X_pool, y_pool_censored, selected_indices,
                                     time_bins, config, increment, y_pool_true):
    """Compute characteristics of selected samples."""

    model.eval()
    device = config.device

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_pool[selected_indices]).to(device)
        logits = model.forward(X_tensor, sample=True, n_samples=config.n_samples_test)
        survival_probs = mtlr_survival(logits, with_sample=True)  # [K, N, T]

    K, N, T = survival_probs.shape

    # Convert to PDF
    survival_np = survival_probs.cpu().numpy()
    pdf = np.zeros_like(survival_np)
    pdf[:, :, 0] = 1.0 - survival_np[:, :, 0]
    pdf[:, :, 1:] = survival_np[:, :, :-1] - survival_np[:, :, 1:]
    pdf = np.clip(pdf, 0, 1.0)

    for k in range(K):
        for n in range(N):
            if pdf[k, n, :].sum() > 0:
                pdf[k, n, :] /= pdf[k, n, :].sum()

    mean_pdf = pdf.mean(axis=0)  # [N, T]

    # Compute characteristics
    time_bins_np = np.array(time_bins)
    temp_bins = np.array([0] + list(time_bins_np)[:-1])

    censor_times = y_pool_censored[selected_indices]
    censor_bins = np.searchsorted(temp_bins, censor_times, side='right') - 1
    reveal_end_bins = np.searchsorted(temp_bins, censor_times + increment, side='right') - 1

    results = {
        'censoring_time': censor_times,
        'n_samples': len(selected_indices)
    }

    # Total variance
    total_variances = []
    for i in range(N):
        variance = pdf[:, i, :].var(axis=0).sum()
        total_variances.append(variance)
    results['total_variance'] = np.array(total_variances)

    # Window uncertainty
    window_variances = []
    for i, (c_bin, e_bin) in enumerate(zip(censor_bins, reveal_end_bins)):
        c_bin = max(0, min(c_bin, T-1))
        e_bin = max(0, min(e_bin, T-1))
        window_probs = pdf[:, i, c_bin:e_bin+1].sum(axis=1)  # [K]
        variance = window_probs.var()
        window_variances.append(variance)
    results['window_uncertainty'] = np.array(window_variances)

    # Spatial diversity
    if len(selected_indices) > 1:
        X_selected = X_pool[selected_indices]
        distances = euclidean_distances(X_selected, X_selected)
        mask = np.triu(np.ones_like(distances), k=1)
        avg_distance = (distances * mask).sum() / mask.sum()
        results['spatial_diversity'] = avg_distance
    else:
        results['spatial_diversity'] = 0.0

    return results


# ==================== MAIN DEBUG ====================

def debug_optimizations():
    """Debug optimized BatchBALD and compare to original."""

    print("="*70)
    print("DEBUGGING OPTIMIZED BATCHBALD")
    print("="*70)
    print()

    # Load data
    print("Loading data...")
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

    # Set up one trial
    seed = 100
    np.random.seed(seed)
    torch.manual_seed(seed)

    initial_samples = 200
    budget = 10
    increment = 60
    prefilter_size = 500

    y_labeled, e_labeled, censored_indices = artificially_censor_true(
        y_train.copy(), e_train.copy(), num_initial_samples=initial_samples
    )

    # Train model
    print("Training base model...")
    model = BayesLinMtlr(
        in_features=X_train.shape[1],
        num_time_bins=len(time_bins) - 1,
        config=config
    ).to(config.device)

    model = train_model(
        model, X_train, y_labeled, e_labeled, time_bins, config,
        epochs=100, patience=20
    )

    # Get censored pool
    censored_list = list(censored_indices)
    X_censored = X_train[censored_list]

    # Pre-filter
    print(f"Pre-filtering {len(censored_list)} → {prefilter_size} samples...")
    filter_indices = prefilter_pool(
        model, X_censored, y_labeled[censored_list],
        e_labeled[censored_list], time_bins, config, config.device, top_k=prefilter_size
    )
    censored_list = [censored_list[i] for i in filter_indices]
    X_censored = X_train[censored_list]

    # Prepare data
    model_data_censored = pd.DataFrame(X_censored)
    model_data_censored["time"] = y_labeled[censored_list]
    model_data_censored["event"] = e_labeled[censored_list]
    costlist = pd.DataFrame([1] * len(X_censored), columns=['Cost'])

    print()
    print("="*70)
    print("TEST 1: Check if optimized code runs correctly")
    print("="*70)

    # Run original BatchBALD
    print("\n1. Running ORIGINAL BatchBALD...")
    try:
        selected_orig = batchbald_acquire_budget(
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
        print(f"   ✓ Selected {len(selected_orig)} samples")
        print(f"   Indices: {selected_orig[:5]}... (showing first 5)")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        selected_orig = None

    # Run optimized with window only
    print("\n2. Running OPTIMIZED (λ₁=0.0, window only)...")
    try:
        selected_window = batchbald_optimal(
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
            lambda_variance=0.0,
            lambda_window=2.0,
            verbose=True
        )
        print(f"   ✓ Selected {len(selected_window)} samples")
        print(f"   Indices: {selected_window[:5]}... (showing first 5)")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        selected_window = None

    # Run optimized with both
    print("\n3. Running OPTIMIZED (λ₁=0.3, window + variance)...")
    try:
        selected_both = batchbald_optimal(
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
            lambda_variance=0.3,
            lambda_window=2.0,
            verbose=True
        )
        print(f"   ✓ Selected {len(selected_both)} samples")
        print(f"   Indices: {selected_both[:5]}... (showing first 5)")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        selected_both = None

    print()
    print("="*70)
    print("TEST 2: Compare characteristics of selected samples")
    print("="*70)

    # Convert to global indices
    if selected_orig is not None:
        global_orig = [censored_list[i] for i in selected_orig]
        chars_orig = compute_selection_characteristics(
            model, X_train, y_labeled, global_orig,
            time_bins, config, increment, y_train
        )
        print("\nOriginal BatchBALD:")
        print(f"  Total variance: mean={chars_orig['total_variance'].mean():.6f}, "
              f"std={chars_orig['total_variance'].std():.6f}")
        print(f"  Window uncertainty: mean={chars_orig['window_uncertainty'].mean():.6f}, "
              f"std={chars_orig['window_uncertainty'].std():.6f}")
        print(f"  Spatial diversity: {chars_orig['spatial_diversity']:.4f}")
        print(f"  Censoring time: mean={chars_orig['censoring_time'].mean():.2f}, "
              f"std={chars_orig['censoring_time'].std():.2f}")

    if selected_window is not None:
        global_window = [censored_list[i] for i in selected_window]
        chars_window = compute_selection_characteristics(
            model, X_train, y_labeled, global_window,
            time_bins, config, increment, y_train
        )
        print("\nOptimized (λ₁=0.0, window only):")
        print(f"  Total variance: mean={chars_window['total_variance'].mean():.6f}, "
              f"std={chars_window['total_variance'].std():.6f}")
        print(f"  Window uncertainty: mean={chars_window['window_uncertainty'].mean():.6f}, "
              f"std={chars_window['window_uncertainty'].std():.6f}")
        print(f"  Spatial diversity: {chars_window['spatial_diversity']:.4f}")
        print(f"  Censoring time: mean={chars_window['censoring_time'].mean():.2f}, "
              f"std={chars_window['censoring_time'].std():.2f}")

        if selected_orig is not None:
            print("\n  Changes vs Original:")
            print(f"    Total variance: {chars_window['total_variance'].mean() - chars_orig['total_variance'].mean():+.6f} "
                  f"({100*(chars_window['total_variance'].mean() / chars_orig['total_variance'].mean() - 1):+.1f}%)")
            print(f"    Window uncertainty: {chars_window['window_uncertainty'].mean() - chars_orig['window_uncertainty'].mean():+.6f} "
                  f"({100*(chars_window['window_uncertainty'].mean() / chars_orig['window_uncertainty'].mean() - 1):+.1f}%)")
            print(f"    Spatial diversity: {chars_window['spatial_diversity'] - chars_orig['spatial_diversity']:+.4f} "
                  f"({100*(chars_window['spatial_diversity'] / chars_orig['spatial_diversity'] - 1):+.1f}%)")

    if selected_both is not None:
        global_both = [censored_list[i] for i in selected_both]
        chars_both = compute_selection_characteristics(
            model, X_train, y_labeled, global_both,
            time_bins, config, increment, y_train
        )
        print("\nOptimized (λ₁=0.3, window + variance):")
        print(f"  Total variance: mean={chars_both['total_variance'].mean():.6f}, "
              f"std={chars_both['total_variance'].std():.6f}")
        print(f"  Window uncertainty: mean={chars_both['window_uncertainty'].mean():.6f}, "
              f"std={chars_both['window_uncertainty'].std():.6f}")
        print(f"  Spatial diversity: {chars_both['spatial_diversity']:.4f}")
        print(f"  Censoring time: mean={chars_both['censoring_time'].mean():.2f}, "
              f"std={chars_both['censoring_time'].std():.2f}")

        if selected_orig is not None:
            print("\n  Changes vs Original:")
            print(f"    Total variance: {chars_both['total_variance'].mean() - chars_orig['total_variance'].mean():+.6f} "
                  f"({100*(chars_both['total_variance'].mean() / chars_orig['total_variance'].mean() - 1):+.1f}%)")
            print(f"    Window uncertainty: {chars_both['window_uncertainty'].mean() - chars_orig['window_uncertainty'].mean():+.6f} "
                  f"({100*(chars_both['window_uncertainty'].mean() / chars_orig['window_uncertainty'].mean() - 1):+.1f}%)")
            print(f"    Spatial diversity: {chars_both['spatial_diversity'] - chars_orig['spatial_diversity']:+.4f} "
                  f"({100*(chars_both['spatial_diversity'] / chars_orig['spatial_diversity'] - 1):+.1f}%)")

    print()
    print("="*70)
    print("TEST 3: Check overlap between selections")
    print("="*70)

    if selected_orig is not None and selected_window is not None:
        overlap_window = len(set(selected_orig) & set(selected_window))
        print(f"\nOriginal vs Window-only: {overlap_window}/{budget} samples overlap ({100*overlap_window/budget:.0f}%)")

    if selected_orig is not None and selected_both is not None:
        overlap_both = len(set(selected_orig) & set(selected_both))
        print(f"Original vs Window+Variance: {overlap_both}/{budget} samples overlap ({100*overlap_both/budget:.0f}%)")

    if selected_window is not None and selected_both is not None:
        overlap_opts = len(set(selected_window) & set(selected_both))
        print(f"Window-only vs Window+Variance: {overlap_opts}/{budget} samples overlap ({100*overlap_opts/budget:.0f}%)")

    print()
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    debug_optimizations()
