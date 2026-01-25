"""
Simple experiment to compare BatchBALD, Entropy, and Variance acquisition.
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from scipy import stats
from tqdm import tqdm

# Local imports
from model import BayesLinMtlr, mtlr_survival
from utils import (artificially_censor_true, remove_nan_rows, make_prediction,
                   encode_survival, reformat_survival)
# Simple concordance function (doesn't need lifelines)
def concordance(y_pred, y_test, cens):
    """Calculate concordance index."""
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
    return n_concordant / n if n > 0 else 0.0
from acquisition import (batchbald_acquire_budget, entropy_of_probs,
                         variance_of_probs, random_knapsack)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def train_model(model, X_train, y_train, e_train, time_bins, config, device,
                num_epochs=100, patience=20, verbose=False):
    """Train the model with early stopping."""
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import StratifiedKFold

    model.reset_parameters()

    X = torch.FloatTensor(X_train).to(device)
    y = torch.FloatTensor(y_train).to(device)
    e = torch.FloatTensor(e_train).to(device)

    # Create train/val split
    try:
        train_idx, val_idx = next(StratifiedKFold(n_splits=10, shuffle=True,
                                                   random_state=42).split(X.cpu(), e.cpu()))
    except:
        indices = torch.randperm(len(X))
        train_size = int(0.9 * len(X))
        train_idx = indices[:train_size].numpy()
        val_idx = indices[train_size:].numpy()

    x_train_split, y_train_split, e_train_split = X[train_idx], y[train_idx], e[train_idx]
    x_val, y_val, e_val = X[val_idx], y[val_idx], e[val_idx]

    # Format targets
    train_df = pd.DataFrame(x_train_split.cpu().numpy())
    train_df['time'] = y_train_split.cpu().numpy()
    train_df['event'] = e_train_split.cpu().numpy()
    x_formatted, y_formatted = reformat_survival(train_df, time_bins[1:])

    val_df = pd.DataFrame(x_val.cpu().numpy())
    val_df['time'] = y_val.cpu().numpy()
    val_df['event'] = e_val.cpu().numpy()
    x_val_formatted, y_val_formatted = reformat_survival(val_df, time_bins[1:])

    train_dataset = TensorDataset(x_formatted, y_formatted)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float('inf')
    best_ep = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        for xi, yi in train_loader:
            xi, yi = xi.to(device), yi.to(device)
            optimizer.zero_grad()
            loss, _, _, _ = model.sample_elbo(xi, yi, len(train_idx), see1=config.c1)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss, _, _, _ = model.sample_elbo(x_val_formatted, y_val_formatted,
                                                   len(val_idx), see1=config.c1)
            val_loss = val_loss / len(val_idx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ep = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_ep > patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    return model


def run_single_experiment(seed, acquisition_functions, increment=6, budget=10,
                          initial_samples=200, train_size=1400, num_bins=10,
                          num_epochs=100, verbose=False):
    """Run a single experiment comparing acquisition functions."""
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load NACD data
    data = pd.read_csv('../data/MIMIC/NACD/NACD_Full.csv')
    cols_to_drop = ['PERFORMANCE_STATUS', 'STAGE_NUMERICAL', 'AGE65']
    data = data.drop([c for c in cols_to_drop if c in data.columns], axis=1)

    if "CENSORED" in data.columns:
        data["event"] = 1 - data["CENSORED"]
        data = data.drop(columns=["CENSORED"])
    if "SURVIVAL" in data.columns:
        data = data.rename(columns={"SURVIVAL": "time"})

    # Standardize
    cols_standardize = ['BOX1_SCORE', 'BOX2_SCORE', 'BOX3_SCORE', 'BMI', 'WEIGHT_CHANGEPOINT',
                        'AGE', 'GRANULOCYTES', 'LDH_SERUM', 'LYMPHOCYTES',
                        'PLATELET', 'WBC_COUNT', 'CALCIUM_SERUM', 'HGB', 'CREATININE_SERUM', 'ALBUMIN']
    cols_standardize = [c for c in cols_standardize if c in data.columns]
    data[cols_standardize] = data[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())

    # Prepare data
    X = data.drop(["time", "event"], axis=1).values
    y = data["time"].values
    e = data["event"].values
    X, y, e = remove_nan_rows(X, y, e)

    # Train/test split
    test_size = max(0.1, min(0.5, 1.0 - (train_size / len(X))))
    X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(
        X, y, e, test_size=test_size, random_state=seed, stratify=e
    )

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create time bins
    event_times = y_train[e_train == 1]
    quantiles = np.linspace(0, 1, num_bins + 1)[1:]
    time_bins = np.quantile(event_times, quantiles)
    time_bins[-1] *= 1.05
    time_bins = np.array([0] + list(time_bins))

    # Artificially censor
    y_labeled, e_labeled, censored_indices = artificially_censor_true(
        y_train.copy(), e_train.copy(), num_initial_samples=initial_samples
    )

    # Setup config
    class Config:
        batch_size = 32
        lr = 8e-5
        c1 = 0.01
        n_samples_test = 50
        dropout = 0.2
        hidden_size = 50
        pi = 0.5
        sigma1 = 1.0
        sigma2 = 0.0025
        rho_scale = -3.0
        mu_scale = 0.1
    config = Config()

    # Create base model
    num_time_bins_for_model = len(time_bins) - 1
    base_model = BayesLinMtlr(X_train.shape[1], num_time_bins_for_model, config).to(device)

    # Train base model
    base_model = train_model(base_model, X_train, y_labeled, e_labeled,
                             time_bins, config, device, num_epochs=num_epochs)

    # Initial evaluation
    survival_outputs, time_bins_ret, _ = make_prediction(base_model, X_test, time_bins, config)
    if isinstance(survival_outputs, torch.Tensor):
        survival_outputs = survival_outputs.cpu().numpy()
    initial_pred_times = expected_times_from_survival(survival_outputs, time_bins)
    initial_c = concordance(-initial_pred_times, y_test, e_test)

    results = {'initial': initial_c}

    # Filter learnable points
    learnable_censored = [idx for idx in censored_indices if y_labeled[idx] < y_train[idx]]

    if verbose:
        print(f"  Seed {seed}: Initial C-index: {initial_c:.4f}, Learnable points: {len(learnable_censored)}")

    columns = data.drop(["time", "event"], axis=1).columns

    for acq_func in acquisition_functions:
        # Fresh copy of model and data
        model = BayesLinMtlr(X_train.shape[1], num_time_bins_for_model, config).to(device)
        model.load_state_dict(base_model.state_dict())

        y_copy = y_labeled.copy()
        e_copy = e_labeled.copy()

        # Get censored pool
        censored_list = list(learnable_censored)
        if len(censored_list) == 0:
            results[acq_func.__name__] = initial_c
            continue

        X_censored = X_train[censored_list]

        model_data_censored = pd.DataFrame(X_censored, columns=columns)
        model_data_censored["time"] = y_copy[censored_list]
        model_data_censored["event"] = e_copy[censored_list]

        costlist = pd.DataFrame([1] * len(X_censored), columns=['Cost'])

        try:
            if acq_func.__name__ == 'random_knapsack':
                pool_indices = acq_func(costlist, budget)
            else:
                pool_indices, _ = acq_func(
                    model=model, X_pool=X_censored, batch_size=budget,
                    time_bins=time_bins, config=config, device=device,
                    in_data_train=model_data_censored, increment=increment,
                    costlist=costlist, budget=budget
                )

            acquired_indices = [censored_list[i] for i in pool_indices]

        except Exception as ex:
            if verbose:
                print(f"    Error with {acq_func.__name__}: {ex}")
            results[acq_func.__name__] = initial_c
            continue

        # Update labels (oracle query)
        for idx in acquired_indices:
            y_copy[idx] = min(y_train[idx], y_copy[idx] + increment)
            if y_copy[idx] >= y_train[idx]:
                e_copy[idx] = e_train[idx]

        # Retrain
        model = train_model(model, X_train, y_copy, e_copy, time_bins, config,
                           device, num_epochs=num_epochs)

        # Evaluate
        survival_outputs, _, _ = make_prediction(model, X_test, time_bins, config)
        if isinstance(survival_outputs, torch.Tensor):
            survival_outputs = survival_outputs.cpu().numpy()
        pred_times = expected_times_from_survival(survival_outputs, time_bins)
        new_c = concordance(-pred_times, y_test, e_test)

        results[acq_func.__name__] = new_c

        if verbose:
            print(f"    {acq_func.__name__}: {new_c:.4f} (Δ={new_c - initial_c:+.4f})")

    return results


def run_experiments(n_trials=10, **kwargs):
    """Run multiple experiments and compute statistics."""
    acquisition_functions = [
        batchbald_acquire_budget,
        entropy_of_probs,
        variance_of_probs,
        random_knapsack,
    ]

    all_results = {f.__name__: [] for f in acquisition_functions}
    all_results['initial'] = []

    print(f"Running {n_trials} trials...")

    for i in range(n_trials):
        print(f"\nTrial {i+1}/{n_trials}")
        results = run_single_experiment(
            seed=42 + i,
            acquisition_functions=acquisition_functions,
            verbose=True,
            **kwargs
        )

        for name, value in results.items():
            all_results[name].append(value)

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    # Compute statistics
    for name in ['initial'] + [f.__name__ for f in acquisition_functions]:
        values = all_results[name]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{name:30s}: {mean_val:.4f} ± {std_val:.4f}")

    # Statistical tests
    print("\n" + "="*60)
    print("STATISTICAL TESTS (paired t-test vs entropy)")
    print("="*60)

    entropy_vals = all_results['entropy_of_probs']

    for name in ['batchbald_acquire_budget', 'variance_of_probs', 'random_knapsack']:
        vals = all_results[name]
        t_stat, p_val = stats.ttest_rel(vals, entropy_vals)
        diff = np.mean(vals) - np.mean(entropy_vals)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{name:30s} vs entropy: diff={diff:+.4f}, p={p_val:.4f} {sig}")

    return all_results


if __name__ == "__main__":
    results = run_experiments(
        n_trials=5,
        increment=6,
        budget=10,
        initial_samples=200,
        train_size=1400,
        num_bins=10,
        num_epochs=100
    )
