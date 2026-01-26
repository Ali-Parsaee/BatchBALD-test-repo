"""
Investigation: What makes acquisition functions succeed in censored survival analysis?

This script tests multiple hypotheses about what characteristics predict
acquisition function performance.

Hypotheses:
1. H1: Methods that select samples with higher death probability in reveal window perform better
2. H2: Methods that select diverse samples (spatially distributed) perform better
3. H3: Methods that select samples where model is most uncertain in reveal window perform better
4. H4: Methods that avoid early-censored samples (small reveal window) perform better
5. H5: Methods that select samples where reveal would change predictions most perform better
6. H6: Reveal likelihood (prob oracle gives useful info) correlates with performance
7. H7: Balance between uncertainty and diversity predicts performance
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
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from Model_stuff.model import BayesLinMtlr, mtlr_survival
from Model_stuff.acquisition import (
    batchbald_acquire_budget,
    entropy_of_probs,
    variance_of_probs,
    random_knapsack
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


# ==================== CHARACTERISTIC COMPUTATION ====================

def compute_selection_characteristics(
    model, X_pool, y_pool_censored, selected_indices,
    time_bins, config, increment, y_pool_true
):
    """
    Compute detailed characteristics of selected samples.

    Returns dict with:
    - death_prob_in_window: P(death in reveal window)
    - uncertainty_in_window: Variance of predictions in reveal window
    - reveal_likelihood: P(oracle reveals new information)
    - spatial_diversity: Average pairwise distance
    - censoring_time: When samples are censored
    - window_size: Size of reveal window
    - prediction_change_potential: How much predictions might change
    """

    model.eval()
    device = config.device

    # Get model predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_pool[selected_indices]).to(device)
        logits = model.forward(X_tensor, sample=True, n_samples=config.n_samples_test)
        survival_probs = mtlr_survival(logits, with_sample=True)  # [K, N, T]

    K, N, T = survival_probs.shape

    # Convert to death probabilities (PDF)
    survival_np = survival_probs.cpu().numpy()
    pdf = np.zeros_like(survival_np)
    pdf[:, :, 0] = 1.0 - survival_np[:, :, 0]
    pdf[:, :, 1:] = survival_np[:, :, :-1] - survival_np[:, :, 1:]
    pdf = np.clip(pdf, 0, 1.0)

    # Normalize
    for k in range(K):
        for n in range(N):
            if pdf[k, n, :].sum() > 0:
                pdf[k, n, :] /= pdf[k, n, :].sum()

    # Average over ensemble
    mean_pdf = pdf.mean(axis=0)  # [N, T]

    # Compute time bin indices
    time_bins_np = np.array(time_bins)
    temp_bins = np.array([0] + list(time_bins_np)[:-1])

    results = {}

    # For each selected sample
    censor_times = y_pool_censored[selected_indices]
    true_times = y_pool_true[selected_indices]

    # Map times to bins
    censor_bins = np.searchsorted(temp_bins, censor_times, side='right') - 1
    reveal_end_bins = np.searchsorted(temp_bins, censor_times + increment, side='right') - 1
    true_bins = np.searchsorted(temp_bins, true_times, side='right') - 1

    # H1: Death probability in reveal window
    death_probs = []
    for i, (c_bin, e_bin) in enumerate(zip(censor_bins, reveal_end_bins)):
        c_bin = max(0, min(c_bin, T-1))
        e_bin = max(0, min(e_bin, T-1))
        prob = mean_pdf[i, c_bin:e_bin+1].sum()
        death_probs.append(prob)
    results['death_prob_in_window'] = np.array(death_probs)

    # H2: Spatial diversity
    if len(selected_indices) > 1:
        X_selected = X_pool[selected_indices]
        distances = euclidean_distances(X_selected, X_selected)
        # Average pairwise distance
        mask = np.triu(np.ones_like(distances), k=1)
        avg_distance = (distances * mask).sum() / mask.sum()
        results['spatial_diversity'] = avg_distance
    else:
        results['spatial_diversity'] = 0.0

    # H3: Uncertainty in reveal window (variance across ensemble)
    window_variances = []
    for i, (c_bin, e_bin) in enumerate(zip(censor_bins, reveal_end_bins)):
        c_bin = max(0, min(c_bin, T-1))
        e_bin = max(0, min(e_bin, T-1))
        # Variance of ensemble predictions in window
        window_probs = pdf[:, i, c_bin:e_bin+1].sum(axis=1)  # [K]
        variance = window_probs.var()
        window_variances.append(variance)
    results['uncertainty_in_window'] = np.array(window_variances)

    # H4: Censoring time and window size
    results['censoring_time'] = censor_times
    results['window_size'] = reveal_end_bins - censor_bins

    # H5: Prediction change potential
    # How much would predictions change if we revealed?
    # Measure: current uncertainty about outcome
    pred_uncertainties = []
    for i in range(N):
        # Entropy of prediction distribution
        probs = mean_pdf[i]
        entropy = -(probs * np.log(probs + 1e-12)).sum()
        pred_uncertainties.append(entropy)
    results['prediction_uncertainty'] = np.array(pred_uncertainties)

    # H6: Reveal likelihood
    # Probability that oracle reveals useful info (death happens in window)
    reveal_likelihoods = []
    for i, (c_bin, e_bin, t_bin) in enumerate(zip(censor_bins, reveal_end_bins, true_bins)):
        # True death time is in reveal window?
        if c_bin <= t_bin <= e_bin:
            # Will reveal exact death time
            likelihood = 1.0
        else:
            # Will reveal censoring at end of window (less useful)
            likelihood = 0.5
        reveal_likelihoods.append(likelihood)
    results['reveal_likelihood'] = np.array(reveal_likelihoods)

    # Total uncertainty (overall variance)
    total_variances = []
    for i in range(N):
        variance = pdf[:, i, :].var(axis=0).sum()
        total_variances.append(variance)
    results['total_variance'] = np.array(total_variances)

    return results


# ==================== MAIN INVESTIGATION ====================

def run_investigation(n_trials=5, initial_samples=200, budget=10, increment=60):
    """Run comprehensive investigation of acquisition function characteristics."""

    print("="*70)
    print("ACQUISITION FUNCTION INVESTIGATION")
    print("="*70)
    print()

    # Load and prepare data
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

    # Acquisition functions
    acquisition_functions = [
        (batchbald_acquire_budget, 'BatchBALD'),
        (entropy_of_probs, 'Entropy'),
        (variance_of_probs, 'Variance'),
        (random_knapsack, 'Random')
    ]

    # Store results
    all_characteristics = {name: [] for _, name in acquisition_functions}
    all_improvements = {name: [] for _, name in acquisition_functions}

    print(f"\nRunning {n_trials} trials to collect characteristics...\n")

    for trial in range(n_trials):
        print(f"Trial {trial + 1}/{n_trials}")
        print("-" * 70)

        seed = 100 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Apply artificial censoring
        y_labeled, e_labeled, censored_indices = artificially_censor_true(
            y_train.copy(), e_train.copy(), num_initial_samples=initial_samples
        )

        for acq_func, acq_name in acquisition_functions:
            print(f"  [{acq_name}] Analyzing...", end=' ', flush=True)

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

            initial_cindex = evaluate_model(model, X_test, y_test, e_test, time_bins, config)

            # Get censored pool
            censored_list = list(censored_indices)
            X_censored = X_train[censored_list]

            # Prepare data for acquisition
            model_data_censored = pd.DataFrame(X_censored)
            model_data_censored["time"] = y_labeled[censored_list]
            model_data_censored["event"] = e_labeled[censored_list]
            costlist = pd.DataFrame([1] * len(X_censored), columns=['Cost'])

            # Run acquisition
            try:
                if acq_name == 'Random':
                    pool_indices = acq_func(costlist, budget)
                elif acq_name == 'BatchBALD':
                    pool_indices = acq_func(
                        model=model, X_pool=X_censored, batch_size=budget,
                        time_bins=time_bins, config=config, device=config.device,
                        in_data_train=model_data_censored, increment=increment,
                        costlist=costlist, budget=budget, num_samples=10000
                    )
                else:
                    pool_indices, _ = acq_func(
                        model=model, X_pool=X_censored, batch_size=budget,
                        time_bins=time_bins, config=config, device=config.device,
                        in_data_train=model_data_censored, increment=increment,
                        costlist=costlist, budget=budget
                    )

                acquired_indices = [censored_list[i] for i in pool_indices]

                # Compute characteristics of selected samples
                characteristics = compute_selection_characteristics(
                    model, X_train, y_labeled, acquired_indices,
                    time_bins, config, increment, y_train
                )

                # Update labels
                for idx in acquired_indices:
                    y_labeled[idx] = min(y_train[idx], y_labeled[idx] + increment)
                    if y_labeled[idx] >= y_train[idx]:
                        e_labeled[idx] = e_train[idx]

                # Retrain and evaluate
                model = BayesLinMtlr(
                    in_features=X_train.shape[1],
                    num_time_bins=len(time_bins) - 1,
                    config=config
                ).to(config.device)

                model = train_model(
                    model, X_train, y_labeled, e_labeled, time_bins, config,
                    epochs=100, patience=20
                )

                final_cindex = evaluate_model(model, X_test, y_test, e_test, time_bins, config)
                improvement = final_cindex - initial_cindex

                all_characteristics[acq_name].append(characteristics)
                all_improvements[acq_name].append(improvement)

                print(f"improvement: {improvement:+.4f}")

            except Exception as e:
                print(f"ERROR: {e}")
                continue

        print()

    # ==================== HYPOTHESIS TESTING ====================

    print("\n" + "="*70)
    print("HYPOTHESIS TESTING")
    print("="*70)
    print()

    # Aggregate characteristics across trials
    agg_characteristics = {}
    for name in all_characteristics:
        if len(all_characteristics[name]) == 0:
            continue

        agg = {}
        for key in all_characteristics[name][0]:
            if key == 'spatial_diversity':
                # Single value per trial
                agg[key] = np.mean([c[key] for c in all_characteristics[name]])
            else:
                # Array of values - concatenate across trials
                agg[key] = np.concatenate([c[key] for c in all_characteristics[name]])
        agg_characteristics[name] = agg

    avg_improvements = {name: np.mean(all_improvements[name]) for name in all_improvements}

    # Sort methods by performance
    sorted_methods = sorted(avg_improvements.items(), key=lambda x: x[1], reverse=True)

    print("Method Performance:")
    print("-" * 70)
    for name, imp in sorted_methods:
        print(f"  {name:<15} Improvement: {imp:+.4f}")
    print()

    # Test each hypothesis
    hypotheses = []

    # H1: Death probability in reveal window
    print("H1: Higher death probability in reveal window → Better performance")
    print("-" * 70)
    for name in agg_characteristics:
        avg_death_prob = agg_characteristics[name]['death_prob_in_window'].mean()
        improvement = avg_improvements[name]
        print(f"  {name:<15} Avg death prob: {avg_death_prob:.4f}  |  Improvement: {improvement:+.4f}")

    # Correlation
    x = [agg_characteristics[name]['death_prob_in_window'].mean() for name in agg_characteristics]
    y = [avg_improvements[name] for name in agg_characteristics]
    if len(x) > 2:
        corr, p_value = stats.pearsonr(x, y)
        print(f"\n  Correlation: r={corr:.3f}, p={p_value:.3f}")
        hypotheses.append(('H1: Death prob in window', corr, p_value))
    print()

    # H2: Spatial diversity
    print("H2: Higher spatial diversity → Better performance")
    print("-" * 70)
    for name in agg_characteristics:
        diversity = agg_characteristics[name]['spatial_diversity']
        improvement = avg_improvements[name]
        print(f"  {name:<15} Spatial diversity: {diversity:.4f}  |  Improvement: {improvement:+.4f}")

    x = [agg_characteristics[name]['spatial_diversity'] for name in agg_characteristics]
    y = [avg_improvements[name] for name in agg_characteristics]
    if len(x) > 2:
        corr, p_value = stats.pearsonr(x, y)
        print(f"\n  Correlation: r={corr:.3f}, p={p_value:.3f}")
        hypotheses.append(('H2: Spatial diversity', corr, p_value))
    print()

    # H3: Uncertainty in reveal window
    print("H3: Higher uncertainty in reveal window → Better performance")
    print("-" * 70)
    for name in agg_characteristics:
        avg_uncertainty = agg_characteristics[name]['uncertainty_in_window'].mean()
        improvement = avg_improvements[name]
        print(f"  {name:<15} Avg uncertainty: {avg_uncertainty:.6f}  |  Improvement: {improvement:+.4f}")

    x = [agg_characteristics[name]['uncertainty_in_window'].mean() for name in agg_characteristics]
    y = [avg_improvements[name] for name in agg_characteristics]
    if len(x) > 2:
        corr, p_value = stats.pearsonr(x, y)
        print(f"\n  Correlation: r={corr:.3f}, p={p_value:.3f}")
        hypotheses.append(('H3: Uncertainty in window', corr, p_value))
    print()

    # H4: Avoid early censoring
    print("H4: Later censoring time → Better performance")
    print("-" * 70)
    for name in agg_characteristics:
        avg_censor_time = agg_characteristics[name]['censoring_time'].mean()
        improvement = avg_improvements[name]
        print(f"  {name:<15} Avg censor time: {avg_censor_time:.2f}  |  Improvement: {improvement:+.4f}")

    x = [agg_characteristics[name]['censoring_time'].mean() for name in agg_characteristics]
    y = [avg_improvements[name] for name in agg_characteristics]
    if len(x) > 2:
        corr, p_value = stats.pearsonr(x, y)
        print(f"\n  Correlation: r={corr:.3f}, p={p_value:.3f}")
        hypotheses.append(('H4: Later censoring', corr, p_value))
    print()

    # H5: Prediction uncertainty
    print("H5: Higher prediction uncertainty → Better performance")
    print("-" * 70)
    for name in agg_characteristics:
        avg_pred_unc = agg_characteristics[name]['prediction_uncertainty'].mean()
        improvement = avg_improvements[name]
        print(f"  {name:<15} Avg pred uncertainty: {avg_pred_unc:.4f}  |  Improvement: {improvement:+.4f}")

    x = [agg_characteristics[name]['prediction_uncertainty'].mean() for name in agg_characteristics]
    y = [avg_improvements[name] for name in agg_characteristics]
    if len(x) > 2:
        corr, p_value = stats.pearsonr(x, y)
        print(f"\n  Correlation: r={corr:.3f}, p={p_value:.3f}")
        hypotheses.append(('H5: Prediction uncertainty', corr, p_value))
    print()

    # H6: Reveal likelihood
    print("H6: Higher reveal likelihood → Better performance")
    print("-" * 70)
    for name in agg_characteristics:
        avg_reveal = agg_characteristics[name]['reveal_likelihood'].mean()
        improvement = avg_improvements[name]
        print(f"  {name:<15} Avg reveal likelihood: {avg_reveal:.4f}  |  Improvement: {improvement:+.4f}")

    x = [agg_characteristics[name]['reveal_likelihood'].mean() for name in agg_characteristics]
    y = [avg_improvements[name] for name in agg_characteristics]
    if len(x) > 2:
        corr, p_value = stats.pearsonr(x, y)
        print(f"\n  Correlation: r={corr:.3f}, p={p_value:.3f}")
        hypotheses.append(('H6: Reveal likelihood', corr, p_value))
    print()

    # H7: Total variance
    print("H7: Higher total variance → Better performance")
    print("-" * 70)
    for name in agg_characteristics:
        avg_total_var = agg_characteristics[name]['total_variance'].mean()
        improvement = avg_improvements[name]
        print(f"  {name:<15} Avg total variance: {avg_total_var:.6f}  |  Improvement: {improvement:+.4f}")

    x = [agg_characteristics[name]['total_variance'].mean() for name in agg_characteristics]
    y = [avg_improvements[name] for name in agg_characteristics]
    if len(x) > 2:
        corr, p_value = stats.pearsonr(x, y)
        print(f"\n  Correlation: r={corr:.3f}, p={p_value:.3f}")
        hypotheses.append(('H7: Total variance', corr, p_value))
    print()

    # Summary
    print("\n" + "="*70)
    print("HYPOTHESIS SUMMARY (sorted by correlation strength)")
    print("="*70)
    print()

    hypotheses_sorted = sorted(hypotheses, key=lambda x: abs(x[1]), reverse=True)

    for h_name, corr, p_value in hypotheses_sorted:
        significance = ""
        if p_value < 0.05:
            significance = " *"
        if p_value < 0.01:
            significance = " **"
        if p_value < 0.001:
            significance = " ***"

        direction = "POSITIVE" if corr > 0 else "NEGATIVE"
        print(f"{h_name:<40} r={corr:+.3f} (p={p_value:.3f}){significance} [{direction}]")

    print("\n* p<0.05, ** p<0.01, *** p<0.001")
    print()

    # Save detailed results
    results_dict = {
        'characteristics': agg_characteristics,
        'improvements': avg_improvements,
        'hypotheses': hypotheses
    }

    return results_dict


if __name__ == "__main__":
    results = run_investigation(n_trials=5, initial_samples=200, budget=10, increment=60)
    print("\nInvestigation complete!")
