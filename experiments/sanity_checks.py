"""
Sanity Checks for Survival Active Learning on NACD Dataset

Replicates the working training approach from Model_stuff folder.
Key differences from previous version:
1. Uses BayesLinMtlr (linear model)
2. Uses RobustScaler
3. Creates time bins from event times only
4. Uses validation split with early stopping
5. Lower learning rate (8e-5)
6. More epochs with early stopping
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from tqdm import tqdm

# Import from the root model.py (not Model_stuff)
from model import BayesLinMtlr, mtlr_nll


# ============ Data Loading (matching Model_stuff/data.py) ============

def load_nacd_data():
    """Load NACD dataset matching Model_stuff approach."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'MIMIC', 'NACD', 'NACD_Full.csv')
    data = pd.read_csv(data_path)

    # Drop certain columns
    cols_to_drop = ['PERFORMANCE_STATUS', 'STAGE_NUMERICAL', 'AGE65']
    data = data.drop([c for c in cols_to_drop if c in data.columns], axis=1)

    # CENSORED=1 means censored (no death) -> event=0
    # CENSORED=0 means died -> event=1
    if "CENSORED" in data.columns:
        data["event"] = 1 - data["CENSORED"]
        data = data.drop(columns=["CENSORED"])
    if "SURVIVAL" in data.columns:
        data = data.rename(columns={"SURVIVAL": "time"})

    # Standardize specific columns
    cols_standardize = ['BOX1_SCORE', 'BOX2_SCORE', 'BOX3_SCORE', 'BMI', 'WEIGHT_CHANGEPOINT',
                        'AGE', 'GRANULOCYTES', 'LDH_SERUM', 'LYMPHOCYTES',
                        'PLATELET', 'WBC_COUNT', 'CALCIUM_SERUM', 'HGB', 'CREATININE_SERUM', 'ALBUMIN']
    cols_standardize = [c for c in cols_standardize if c in data.columns]
    data[cols_standardize] = data[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())

    return data.astype(float)


# ============ Model Config (matching Model_stuff/main.py) ============

class ModelConfig:
    """Configuration matching Model_stuff."""
    def __init__(self):
        self.lr = 8e-5  # Lower learning rate
        self.learning_rate = 8e-5
        self.batch_size = 32
        self.dropout = 0.2
        self.dropout_rate = 0.2
        self.l2_penalty = 1e-4
        self.weight_decay = 1e-4
        self.hidden_size = 50
        self.num_time_bins = 10
        self.pi = 0.5
        self.sigma1 = 1.0
        self.sigma2 = 0.0025
        self.rho_scale = -3.0  # Different from before
        self.mu_scale = 0.1
        self.c1 = 0.01
        self.n_samples_train = 10
        self.device = 'cpu'


# ============ Helper Functions (matching Model_stuff) ============

def encode_survival(time, event, bins):
    """Encodes survival time and event indicator for MTLR training."""
    if isinstance(time, (float, int, np.ndarray)):
        time = np.atleast_1d(time)
        time = torch.tensor(time, dtype=torch.float32)
    if isinstance(event, (int, bool, np.ndarray)):
        event = np.atleast_1d(event)
        event = torch.tensor(event)
    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins, dtype=torch.float32)

    time = torch.clamp(time, 0, bins.max())
    y = torch.zeros((time.shape[0], bins.shape[0] + 1), dtype=torch.float)
    bin_idxs = torch.bucketize(time, bins, right=True)

    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y


def reformat_survival(X, time, event, bins):
    """Reformat survival data for training."""
    x = torch.tensor(X, dtype=torch.float32)
    y = encode_survival(time, event, bins)
    return x, y


def make_time_bins(times, events, num_bins=10):
    """Create time bins from event times only (matching Model_stuff)."""
    event_times = times[events == 1]
    quantiles = np.linspace(0, 1, num_bins + 1)[1:]
    bins = np.quantile(event_times, quantiles)
    bins[-1] *= 1.05  # Extend last bin slightly
    bins = np.array([0] + list(bins))
    return bins


def train_model_with_validation(model, X_train, time_train, event_train, time_bins, config,
                                 num_epochs=200, patience=20, verbose=False):
    """
    Train model with validation split and early stopping (matching Model_stuff).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    model.reset_parameters()

    device = torch.device(config.device)

    # Create train/val split
    try:
        train_indices, val_indices = next(
            StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(X_train, event_train)
        )
    except:
        indices = np.random.permutation(len(X_train))
        train_size = int(0.9 * len(X_train))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

    X_tr, time_tr, event_tr = X_train[train_indices], time_train[train_indices], event_train[train_indices]
    X_val, time_val, event_val = X_train[val_indices], time_train[val_indices], event_train[val_indices]

    # Use time_bins[1:] because encode_survival creates bins.shape[0] + 1 outputs
    x_train_tensor, y_train_tensor = reformat_survival(X_tr, time_tr, event_tr, time_bins[1:])
    x_val_tensor, y_val_tensor = reformat_survival(X_val, time_val, event_val, time_bins[1:])

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None

    pbar = tqdm(range(num_epochs), desc="Training", disable=not verbose)
    for epoch in pbar:
        model.train()
        total_loss = 0

        for xi, yi in train_loader:
            xi, yi = xi.to(device), yi.to(device)
            optimizer.zero_grad()
            loss, _, _, _ = model.sample_elbo(xi, yi, len(train_indices), see1=config.c1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss, _, _, _ = model.sample_elbo(x_val_tensor, y_val_tensor, len(val_indices), see1=config.c1)
            val_loss = val_loss.item() / len(val_indices)

        pbar.set_postfix({"Train": f"{total_loss/len(train_loader):.4f}", "Val": f"{val_loss:.4f}"})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch > patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    return model


def expected_times_from_survival(surv_np, time_bins):
    """Convert survival probabilities to expected survival times."""
    if isinstance(time_bins, torch.Tensor):
        tb = time_bins.cpu().numpy()
    else:
        tb = np.array(time_bins)

    left_edges = np.concatenate([[0.0], tb[:-1]])
    right_edges = tb
    mids = (left_edges + right_edges) / 2.0

    # PDF from survival function
    pdf = np.zeros_like(surv_np)
    pdf[:, 0] = 1.0 - surv_np[:, 0]
    pdf[:, 1:] = surv_np[:, :-1] - surv_np[:, 1:]
    pdf = np.clip(pdf, 0.0, 1.0)
    pdf = pdf / (pdf.sum(axis=1, keepdims=True) + 1e-12)

    return (pdf * mids.reshape(1, -1)).sum(axis=1)


def compute_c_index(pred_times, true_times, events):
    """
    Compute concordance index.
    Higher predicted time should correspond to later actual death.
    """
    n = len(true_times)
    concordant = 0
    total = 0

    for i in range(n):
        if events[i] == 0:  # Only count pairs where i has observed event
            continue
        for j in range(n):
            if true_times[j] > true_times[i]:  # j survived longer than i
                total += 1
                # Person i died earlier, should have lower predicted time
                if pred_times[i] < pred_times[j]:
                    concordant += 1
                elif pred_times[i] == pred_times[j]:
                    concordant += 0.5

    return concordant / total if total > 0 else 0.5


def evaluate_model(model, X_test, time_test, event_test, time_bins, config):
    """Evaluate model and return C-index."""
    model.eval()
    device = torch.device(config.device)

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_tensor, sample=True, n_samples=20)
        # outputs shape: (n_samples, batch, num_bins)
        # Apply softmax and average over samples
        survival_probs = torch.softmax(outputs, dim=-1).mean(dim=0).cpu().numpy()

    # Get expected survival times
    pred_times = expected_times_from_survival(survival_probs, time_bins)

    # C-index
    c_index = compute_c_index(pred_times, time_test, event_test)

    return c_index, pred_times


# ============ Sanity Check 1: Does the model learn? ============

def sanity_check_1_model_learns(data, n_runs=5):
    """Test if model actually learns on NACD data."""
    print("\n" + "="*80)
    print("SANITY CHECK 1: Does the model learn on NACD?")
    print("="*80)

    feature_cols = [c for c in data.columns if c not in ['time', 'event']]
    X = data[feature_cols].values.astype(np.float32)
    time = data['time'].values.astype(np.float32)
    event = data['event'].values.astype(int)

    print(f"\nDataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"Events: {event.sum()} deaths ({100*event.mean():.1f}%), {len(event)-event.sum()} censored")

    trained_c_indices = []
    random_c_indices = []

    for run in range(n_runs):
        np.random.seed(42 + run)
        torch.manual_seed(42 + run)

        # Stratified split
        X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
            X, time, event, test_size=0.3, random_state=42 + run, stratify=event
        )

        # Apply RobustScaler (matching Model_stuff)
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Create time bins from training event times
        time_bins = make_time_bins(time_train, event_train, num_bins=10)
        num_bins = len(time_bins) - 1  # Exclude leading 0

        # Create model
        config = ModelConfig()
        model = BayesLinMtlr(X_train.shape[1], num_bins, config)

        # Train with validation and early stopping
        model = train_model_with_validation(
            model, X_train, time_train, event_train, time_bins, config,
            num_epochs=200, patience=20, verbose=(run == 0)
        )

        # Evaluate
        c_index, _ = evaluate_model(model, X_test, time_test, event_test, time_bins, config)
        trained_c_indices.append(c_index)

        # Random baseline
        random_pred = np.random.rand(len(X_test))
        random_c = compute_c_index(random_pred, time_test, event_test)
        random_c_indices.append(random_c)

        print(f"  Run {run+1}: Trained C-index={c_index:.4f}, Random={random_c:.4f}")

    mean_trained = np.mean(trained_c_indices)
    mean_random = np.mean(random_c_indices)

    print(f"\nSummary:")
    print(f"  Trained model: {mean_trained:.4f} ± {np.std(trained_c_indices):.4f}")
    print(f"  Random baseline: {mean_random:.4f} ± {np.std(random_c_indices):.4f}")
    print(f"  Improvement: {mean_trained - mean_random:+.4f}")

    if mean_trained > 0.60:
        print("\n✓ PASS: Model learns well (C-index > 0.60)")
        return True
    elif mean_trained > 0.55:
        print("\n~ PARTIAL: Model learns somewhat (C-index > 0.55)")
        return True
    else:
        print("\n✗ FAIL: Model doesn't learn well (C-index ≤ 0.55)")
        return False


# ============ Sanity Check 2: Does more data help? ============

def sanity_check_2_more_data_helps(data, n_runs=5):
    """Test if having more training data improves the model."""
    print("\n" + "="*80)
    print("SANITY CHECK 2: Does more data help the model?")
    print("="*80)

    feature_cols = [c for c in data.columns if c not in ['time', 'event']]
    X = data[feature_cols].values.astype(np.float32)
    time = data['time'].values.astype(np.float32)
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

        for frac in train_fractions:
            # Use fraction of remaining data
            n_train = int(len(X_rest) * frac)
            indices = np.random.choice(len(X_rest), n_train, replace=False)

            X_train = X_rest[indices]
            time_train = time_rest[indices]
            event_train = event_rest[indices]

            # Scale
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Time bins
            time_bins = make_time_bins(time_train, event_train, num_bins=10)
            num_bins = len(time_bins) - 1

            # Train
            config = ModelConfig()
            model = BayesLinMtlr(X_train.shape[1], num_bins, config)
            model = train_model_with_validation(
                model, X_train_scaled, time_train, event_train, time_bins, config,
                num_epochs=200, patience=20, verbose=False
            )

            # Evaluate
            c_index, _ = evaluate_model(model, X_test_scaled, time_test, event_test, time_bins, config)
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

    means = [np.mean(results[f]) for f in train_fractions]
    if means[-1] > means[0] + 0.01:
        print(f"\n✓ PASS: More data helps (80% > 20%: {means[-1]:.4f} > {means[0]:.4f})")
        return True
    else:
        print(f"\n✗ FAIL: More data doesn't help clearly")
        return False


# ============ Sanity Check 3: Which points are most valuable? ============

def sanity_check_3_valuable_points(data, n_runs=5):
    """Test which types of points are most valuable for learning."""
    print("\n" + "="*80)
    print("SANITY CHECK 3: Which point types are most valuable?")
    print("="*80)

    feature_cols = [c for c in data.columns if c not in ['time', 'event']]
    X = data[feature_cols].values.astype(np.float32)
    time = data['time'].values.astype(np.float32)
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

        # Baseline: 20% of pool
        n_baseline = int(0.2 * len(pool_idx))
        baseline_idx = np.random.choice(len(pool_idx), n_baseline, replace=False)

        X_base = X_pool[baseline_idx]
        time_base = time_pool[baseline_idx]
        event_base = event_pool[baseline_idx]

        # Scale
        scaler = RobustScaler()
        X_base_scaled = scaler.fit_transform(X_base)
        X_test_scaled = scaler.transform(X_test)

        # Time bins
        time_bins = make_time_bins(time_base, event_base, num_bins=10)
        num_bins = len(time_bins) - 1

        # Train baseline
        config = ModelConfig()
        model_base = BayesLinMtlr(X_base.shape[1], num_bins, config)
        model_base = train_model_with_validation(
            model_base, X_base_scaled, time_base, event_base, time_bins, config,
            num_epochs=200, patience=20, verbose=False
        )
        c_base, _ = evaluate_model(model_base, X_test_scaled, time_test, event_test, time_bins, config)
        results['baseline'].append(c_base)

        # Remaining pool
        remaining_mask = np.ones(len(pool_idx), dtype=bool)
        remaining_mask[baseline_idx] = False
        remaining_idx = np.where(remaining_mask)[0]

        n_add = int(0.2 * len(pool_idx))

        # Categorize remaining points
        uncensored_idx = remaining_idx[event_pool[remaining_idx] == 1]
        censored_idx = remaining_idx[event_pool[remaining_idx] == 0]

        if len(censored_idx) > 0:
            censored_times = time_pool[censored_idx]
            median_time = np.median(censored_times)
            early_censored_idx = censored_idx[censored_times <= median_time]
            late_censored_idx = censored_idx[censored_times > median_time]
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
                add_idx = available_idx
            else:
                add_idx = np.random.choice(available_idx, n_add, replace=False)

            if len(add_idx) == 0:
                results[point_type].append(c_base)
                continue

            # Combine with baseline
            combined_pool_idx = np.concatenate([baseline_idx, add_idx])
            X_train = X_pool[combined_pool_idx]
            time_train = time_pool[combined_pool_idx]
            event_train = event_pool[combined_pool_idx]

            # Scale (refit on combined)
            scaler_new = RobustScaler()
            X_train_scaled = scaler_new.fit_transform(X_train)
            X_test_scaled_new = scaler_new.transform(X_test)

            # Time bins (recompute)
            time_bins_new = make_time_bins(time_train, event_train, num_bins=10)
            num_bins_new = len(time_bins_new) - 1

            # Train
            torch.manual_seed(42 + run)
            model = BayesLinMtlr(X_train.shape[1], num_bins_new, config)
            model = train_model_with_validation(
                model, X_train_scaled, time_train, event_train, time_bins_new, config,
                num_epochs=200, patience=20, verbose=False
            )
            c_index, _ = evaluate_model(model, X_test_scaled_new, time_test, event_test, time_bins_new, config)
            results[point_type].append(c_index)

        print(f"  Run {run+1}: base={c_base:.3f}, +uncens={results['add_uncensored'][-1]:.3f}, "
              f"+early={results['add_early_censored'][-1]:.3f}, +late={results['add_late_censored'][-1]:.3f}, "
              f"+rand={results['add_random'][-1]:.3f}")

    print(f"\nSummary (mean C-index, improvement over baseline):")
    baseline_mean = np.mean(results['baseline'])
    print(f"  Baseline: {baseline_mean:.4f}")

    improvements = {}
    for point_type in ['add_uncensored', 'add_early_censored', 'add_late_censored', 'add_random']:
        mean_c = np.mean(results[point_type])
        improvement = mean_c - baseline_mean
        improvements[point_type] = improvement
        print(f"  {point_type}: {mean_c:.4f} ({improvement:+.4f})")

    best_type = max(improvements, key=improvements.get)
    print(f"\n✓ Most valuable: {best_type} ({improvements[best_type]:+.4f})")

    return improvements


# ============ Main ============

def main():
    print("\n" + "="*100)
    print("SANITY CHECKS FOR SURVIVAL ACTIVE LEARNING ON NACD")
    print("(Using Model_stuff training approach)")
    print("="*100)
    print(f"Started: {datetime.now()}")

    # Load data
    print("\nLoading NACD dataset...")
    data = load_nacd_data()
    print(f"Loaded: {data.shape[0]} samples, {data.shape[1]-2} features")
    print(f"Events: {int(data['event'].sum())} deaths ({100*data['event'].mean():.1f}%)")

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
        print("\n→ Model and data are working! Ready for AL experiments.")
    elif not check1_passed:
        print("\n→ Model still doesn't learn well. Check model/data.")
    elif not check2_passed:
        print("\n→ More data doesn't help. May need regularization.")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()
