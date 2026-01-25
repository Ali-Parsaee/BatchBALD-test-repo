"""
Part 1: Find the best achievable C-index on NACD
Part 2: Analyze which factors cause the most noise in AL settings
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from scipy.stats import ttest_rel
from tqdm import tqdm

from model import BayesLinMtlr


# ============ Data & Config (from sanity_checks.py) ============

def load_nacd_data():
    """Load NACD dataset."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'MIMIC', 'NACD', 'NACD_Full.csv')
    data = pd.read_csv(data_path)
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
    return data.astype(float)


class ModelConfig:
    def __init__(self):
        self.lr = 8e-5
        self.batch_size = 32
        self.dropout = 0.2
        self.hidden_size = 50
        self.pi = 0.5
        self.sigma1 = 1.0
        self.sigma2 = 0.0025
        self.rho_scale = -3.0
        self.mu_scale = 0.1
        self.c1 = 0.01
        self.n_samples_train = 10
        self.device = 'cpu'


def encode_survival(time, event, bins):
    if isinstance(time, np.ndarray):
        time = torch.tensor(time, dtype=torch.float32)
    if isinstance(event, np.ndarray):
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


def make_time_bins(times, events, num_bins=10):
    event_times = times[events == 1]
    quantiles = np.linspace(0, 1, num_bins + 1)[1:]
    bins = np.quantile(event_times, quantiles)
    bins[-1] *= 1.05
    return np.array([0] + list(bins))


def train_model(model, X_train, time_train, event_train, time_bins, config,
                num_epochs=500, patience=30, verbose=False):
    """Train with early stopping."""
    torch.manual_seed(42)
    model.reset_parameters()

    # Train/val split
    try:
        train_idx, val_idx = next(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(X_train, event_train))
    except:
        n = len(X_train)
        idx = np.random.permutation(n)
        train_idx, val_idx = idx[:int(0.9*n)], idx[int(0.9*n):]

    X_tr, time_tr, event_tr = X_train[train_idx], time_train[train_idx], event_train[train_idx]
    X_val, time_val, event_val = X_train[val_idx], time_train[val_idx], event_train[val_idx]

    x_train = torch.tensor(X_tr, dtype=torch.float32)
    y_train = encode_survival(time_tr, event_tr, time_bins[1:])
    x_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = encode_survival(time_val, event_val, time_bins[1:])

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None

    pbar = tqdm(range(num_epochs), desc="Training", disable=not verbose)
    for epoch in pbar:
        model.train()
        for xi, yi in train_loader:
            optimizer.zero_grad()
            loss, _, _, _ = model.sample_elbo(xi, yi, len(train_idx), see1=config.c1)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss, _, _, _ = model.sample_elbo(x_val, y_val, len(val_idx), see1=config.c1)
            val_loss = val_loss.item() / len(val_idx)

        if verbose:
            pbar.set_postfix({"Val": f"{val_loss:.4f}", "Best": best_epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch > patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_epoch


def expected_times_from_survival(surv_np, time_bins):
    tb = np.array(time_bins)
    left_edges = np.concatenate([[0.0], tb[:-1]])
    right_edges = tb
    mids = (left_edges + right_edges) / 2.0
    pdf = np.zeros_like(surv_np)
    pdf[:, 0] = 1.0 - surv_np[:, 0]
    pdf[:, 1:] = surv_np[:, :-1] - surv_np[:, 1:]
    pdf = np.clip(pdf, 0.0, 1.0)
    pdf = pdf / (pdf.sum(axis=1, keepdims=True) + 1e-12)
    return (pdf * mids.reshape(1, -1)).sum(axis=1)


def compute_c_index(pred_times, true_times, events):
    n = len(true_times)
    concordant = 0
    total = 0
    for i in range(n):
        if events[i] == 0:
            continue
        for j in range(n):
            if true_times[j] > true_times[i]:
                total += 1
                if pred_times[i] < pred_times[j]:
                    concordant += 1
                elif pred_times[i] == pred_times[j]:
                    concordant += 0.5
    return concordant / total if total > 0 else 0.5


def evaluate_model(model, X_test, time_test, event_test, time_bins):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_test), sample=True, n_samples=20)
        survival_probs = torch.softmax(outputs, dim=-1).mean(dim=0).cpu().numpy()
    pred_times = expected_times_from_survival(survival_probs, time_bins)
    return compute_c_index(pred_times, time_test, event_test)


# ============ PART 1: Best Achievable C-index ============

def find_best_cindex(data, n_runs=10):
    """Find the best C-index achievable without any AL complications."""
    print("\n" + "="*80)
    print("PART 1: BEST ACHIEVABLE C-INDEX ON NACD")
    print("="*80)

    feature_cols = [c for c in data.columns if c not in ['time', 'event']]
    X = data[feature_cols].values.astype(np.float32)
    time = data['time'].values.astype(np.float32)
    event = data['event'].values.astype(int)

    print(f"Dataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"Events: {event.sum()} deaths ({100*event.mean():.1f}%)")
    print(f"Training with up to 500 epochs, patience=30")

    c_indices = []
    epochs_used = []

    for run in range(n_runs):
        np.random.seed(42 + run)
        torch.manual_seed(42 + run)

        X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
            X, time, event, test_size=0.2, random_state=42 + run, stratify=event
        )

        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        time_bins = make_time_bins(time_train, event_train, num_bins=10)
        num_bins = len(time_bins) - 1

        config = ModelConfig()
        model = BayesLinMtlr(X_train.shape[1], num_bins, config)
        model, best_epoch = train_model(model, X_train, time_train, event_train, time_bins, config,
                                         num_epochs=500, patience=30, verbose=(run == 0))

        c_index = evaluate_model(model, X_test, time_test, event_test, time_bins)
        c_indices.append(c_index)
        epochs_used.append(best_epoch)

        print(f"  Run {run+1}: C-index={c_index:.4f}, stopped at epoch {best_epoch}")

    print(f"\n{'='*40}")
    print(f"BEST ACHIEVABLE C-INDEX: {np.mean(c_indices):.4f} ± {np.std(c_indices):.4f}")
    print(f"Range: [{np.min(c_indices):.4f}, {np.max(c_indices):.4f}]")
    print(f"Average epochs: {np.mean(epochs_used):.0f}")
    print(f"{'='*40}")

    return np.mean(c_indices), np.std(c_indices)


# ============ PART 2: Noise Source Analysis ============

def artificially_censor_random(time, event, proportion=0.5, seed=None):
    """Random artificial censoring."""
    if seed is not None:
        np.random.seed(seed)
    n = len(time)
    art_time = time.copy()
    art_event = event.copy()
    eligible = np.where((event == 1) & (time > 0))[0]
    n_censor = int(len(eligible) * proportion)
    if n_censor > 0:
        to_censor = np.random.choice(eligible, n_censor, replace=False)
        for idx in to_censor:
            art_time[idx] = np.random.uniform(0, time[idx])
            art_event[idx] = 0
    return art_time, art_event


def artificially_censor_deterministic(time, event, proportion=0.5, seed=None):
    """Deterministic midpoint censoring."""
    if seed is not None:
        np.random.seed(seed)
    n = len(time)
    art_time = time.copy()
    art_event = event.copy()
    eligible = np.where((event == 1) & (time > 0))[0]
    n_censor = int(len(eligible) * proportion)
    if n_censor > 0:
        to_censor = np.random.choice(eligible, n_censor, replace=False)
        for idx in to_censor:
            art_time[idx] = time[idx] / 2.0  # Deterministic midpoint
            art_event[idx] = 0
    return art_time, art_event


def oracle_query(indices, current_time, current_event, true_time, true_event, probe_depth):
    """Simulate oracle query with probe depth."""
    new_time = current_time.copy()
    new_event = current_event.copy()
    for idx in indices:
        max_obs = current_time[idx] + probe_depth
        if true_event[idx] == 1 and true_time[idx] <= max_obs:
            new_time[idx] = true_time[idx]
            new_event[idx] = 1
        else:
            new_time[idx] = min(true_time[idx], max_obs)
            new_event[idx] = 0
    return new_time, new_event


def run_al_experiment(X_train, time_train, event_train, X_test, time_test, event_test,
                      config, censor_mode='random', probe_depth=None, n_rounds=1,
                      batch_size=50, seed=42):
    """
    Run one AL experiment.
    Returns: initial_c, final_c, improvement
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    true_time = time_train.copy()
    true_event = event_train.copy()

    # Artificial censoring
    if censor_mode == 'random':
        current_time, current_event = artificially_censor_random(time_train, event_train, 0.5, seed)
    else:
        current_time, current_event = artificially_censor_deterministic(time_train, event_train, 0.5, seed)

    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Time bins
    time_bins = make_time_bins(current_time, current_event, num_bins=10)
    num_bins = len(time_bins) - 1

    # Train initial model
    model = BayesLinMtlr(X_train.shape[1], num_bins, config)
    model, _ = train_model(model, X_train_scaled, current_time, current_event, time_bins, config,
                           num_epochs=200, patience=20, verbose=False)

    initial_c = evaluate_model(model, X_test_scaled, time_test, event_test, time_bins)

    # AL rounds
    for round_num in range(n_rounds):
        # Find censored points that can be queried
        queryable = np.where(current_event == 0)[0]
        if len(queryable) == 0:
            break

        # Random acquisition
        n_select = min(batch_size, len(queryable))
        selected = np.random.choice(queryable, n_select, replace=False)

        # Oracle query
        if probe_depth is not None:
            current_time, current_event = oracle_query(
                selected, current_time, current_event, true_time, true_event, probe_depth
            )
        else:
            # Full reveal (no probe depth limit)
            for idx in selected:
                current_time[idx] = true_time[idx]
                current_event[idx] = true_event[idx]

        # Retrain
        time_bins = make_time_bins(current_time, current_event, num_bins=10)
        num_bins = len(time_bins) - 1
        model = BayesLinMtlr(X_train.shape[1], num_bins, config)
        model, _ = train_model(model, X_train_scaled, current_time, current_event, time_bins, config,
                               num_epochs=200, patience=20, verbose=False)

    final_c = evaluate_model(model, X_test_scaled, time_test, event_test, time_bins)

    return initial_c, final_c, final_c - initial_c


def analyze_noise_sources(data, n_runs=10):
    """Analyze which factors contribute most to noise."""
    print("\n" + "="*80)
    print("PART 2: NOISE SOURCE ANALYSIS")
    print("="*80)

    feature_cols = [c for c in data.columns if c not in ['time', 'event']]
    X = data[feature_cols].values.astype(np.float32)
    time = data['time'].values.astype(np.float32)
    event = data['event'].values.astype(int)

    config = ModelConfig()

    # Prepare fixed train/test splits for fair comparison
    splits = []
    for run in range(n_runs):
        X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
            X, time, event, test_size=0.2, random_state=42 + run, stratify=event
        )
        splits.append((X_train, X_test, time_train, time_test, event_train, event_test))

    results = {}

    # ============ Factor 1: Random vs Deterministic Censoring ============
    print("\n--- Factor 1: Random vs Deterministic Censoring ---")

    random_improvements = []
    determ_improvements = []

    for run, (X_train, X_test, time_train, time_test, event_train, event_test) in enumerate(splits):
        _, _, imp_rand = run_al_experiment(
            X_train, time_train, event_train, X_test, time_test, event_test,
            config, censor_mode='random', probe_depth=500, n_rounds=1, batch_size=100, seed=42+run
        )
        _, _, imp_det = run_al_experiment(
            X_train, time_train, event_train, X_test, time_test, event_test,
            config, censor_mode='deterministic', probe_depth=500, n_rounds=1, batch_size=100, seed=42+run
        )
        random_improvements.append(imp_rand)
        determ_improvements.append(imp_det)
        print(f"  Run {run+1}: Random={imp_rand:+.4f}, Deterministic={imp_det:+.4f}")

    results['random_censor'] = {'mean': np.mean(random_improvements), 'std': np.std(random_improvements)}
    results['determ_censor'] = {'mean': np.mean(determ_improvements), 'std': np.std(determ_improvements)}

    print(f"\n  Random censoring:       {np.mean(random_improvements):+.4f} ± {np.std(random_improvements):.4f}")
    print(f"  Deterministic censoring: {np.mean(determ_improvements):+.4f} ± {np.std(determ_improvements):.4f}")
    print(f"  Variance reduction: {np.std(random_improvements) - np.std(determ_improvements):.4f}")

    # ============ Factor 2: One-shot vs Multi-round AL ============
    print("\n--- Factor 2: One-shot vs Multi-round AL ---")

    oneshot_improvements = []
    multiround_improvements = []

    for run, (X_train, X_test, time_train, time_test, event_train, event_test) in enumerate(splits):
        _, _, imp_one = run_al_experiment(
            X_train, time_train, event_train, X_test, time_test, event_test,
            config, censor_mode='deterministic', probe_depth=500, n_rounds=1, batch_size=100, seed=42+run
        )
        _, _, imp_multi = run_al_experiment(
            X_train, time_train, event_train, X_test, time_test, event_test,
            config, censor_mode='deterministic', probe_depth=500, n_rounds=5, batch_size=20, seed=42+run
        )
        oneshot_improvements.append(imp_one)
        multiround_improvements.append(imp_multi)
        print(f"  Run {run+1}: One-shot={imp_one:+.4f}, Multi-round(5)={imp_multi:+.4f}")

    results['oneshot'] = {'mean': np.mean(oneshot_improvements), 'std': np.std(oneshot_improvements)}
    results['multiround'] = {'mean': np.mean(multiround_improvements), 'std': np.std(multiround_improvements)}

    print(f"\n  One-shot (100 pts):    {np.mean(oneshot_improvements):+.4f} ± {np.std(oneshot_improvements):.4f}")
    print(f"  Multi-round (5×20 pts): {np.mean(multiround_improvements):+.4f} ± {np.std(multiround_improvements):.4f}")

    # ============ Factor 3: Probe Depth ============
    print("\n--- Factor 3: Probe Depth ---")

    probe_results = {}
    for probe_depth in [100, 500, 1000, None]:  # None = full reveal
        improvements = []
        for run, (X_train, X_test, time_train, time_test, event_train, event_test) in enumerate(splits):
            _, _, imp = run_al_experiment(
                X_train, time_train, event_train, X_test, time_test, event_test,
                config, censor_mode='deterministic', probe_depth=probe_depth, n_rounds=1, batch_size=100, seed=42+run
            )
            improvements.append(imp)

        label = f"k={probe_depth}" if probe_depth else "Full"
        probe_results[label] = {'mean': np.mean(improvements), 'std': np.std(improvements)}
        print(f"  Probe depth {label}: {np.mean(improvements):+.4f} ± {np.std(improvements):.4f}")

    results['probe_depth'] = probe_results

    # ============ Factor 4: Amount of artificial censoring ============
    print("\n--- Factor 4: Amount of Artificial Censoring ---")

    censor_results = {}
    for censor_prop in [0.3, 0.5, 0.7]:
        improvements = []
        for run, (X_train, X_test, time_train, time_test, event_train, event_test) in enumerate(splits):
            np.random.seed(42 + run)
            torch.manual_seed(42 + run)

            true_time = time_train.copy()
            true_event = event_train.copy()

            # Custom censoring with different proportions
            art_time, art_event = artificially_censor_deterministic(time_train, event_train, censor_prop, 42+run)

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            time_bins = make_time_bins(art_time, art_event, num_bins=10)
            num_bins = len(time_bins) - 1

            model = BayesLinMtlr(X_train.shape[1], num_bins, config)
            model, _ = train_model(model, X_train_scaled, art_time, art_event, time_bins, config,
                                   num_epochs=200, patience=20, verbose=False)
            initial_c = evaluate_model(model, X_test_scaled, time_test, event_test, time_bins)

            # Query all censored points with full reveal
            queryable = np.where(art_event == 0)[0]
            n_select = min(100, len(queryable))
            selected = np.random.choice(queryable, n_select, replace=False)

            for idx in selected:
                art_time[idx] = true_time[idx]
                art_event[idx] = true_event[idx]

            time_bins = make_time_bins(art_time, art_event, num_bins=10)
            model = BayesLinMtlr(X_train.shape[1], num_bins, config)
            model, _ = train_model(model, X_train_scaled, art_time, art_event, time_bins, config,
                                   num_epochs=200, patience=20, verbose=False)
            final_c = evaluate_model(model, X_test_scaled, time_test, event_test, time_bins)

            improvements.append(final_c - initial_c)

        censor_results[censor_prop] = {'mean': np.mean(improvements), 'std': np.std(improvements)}
        print(f"  {int(censor_prop*100)}% censored: {np.mean(improvements):+.4f} ± {np.std(improvements):.4f}")

    results['censor_proportion'] = censor_results

    # ============ Factor 5: Training noise (same data, different seeds) ============
    print("\n--- Factor 5: Model Training Stochasticity ---")

    training_variance = []
    X_train, X_test, time_train, time_test, event_train, event_test = splits[0]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    time_bins = make_time_bins(time_train, event_train, num_bins=10)
    num_bins = len(time_bins) - 1

    for seed in range(10):
        torch.manual_seed(seed)
        model = BayesLinMtlr(X_train.shape[1], num_bins, config)
        model, _ = train_model(model, X_train_scaled, time_train, event_train, time_bins, config,
                               num_epochs=200, patience=20, verbose=False)
        c = evaluate_model(model, X_test_scaled, time_test, event_test, time_bins)
        training_variance.append(c)

    results['training_noise'] = {'mean': np.mean(training_variance), 'std': np.std(training_variance)}
    print(f"  Same data, 10 different seeds: {np.mean(training_variance):.4f} ± {np.std(training_variance):.4f}")

    return results


def main():
    print("\n" + "="*100)
    print("BEST C-INDEX AND NOISE SOURCE ANALYSIS")
    print("="*100)
    print(f"Started: {datetime.now()}")

    data = load_nacd_data()
    print(f"Loaded NACD: {data.shape[0]} samples, {data.shape[1]-2} features")

    # Part 1: Best achievable C-index
    best_c, best_std = find_best_cindex(data, n_runs=10)

    # Part 2: Noise source analysis
    results = analyze_noise_sources(data, n_runs=10)

    # Summary
    print("\n" + "="*100)
    print("SUMMARY: NOISE SOURCE RANKING")
    print("="*100)

    print(f"\nBest achievable C-index: {best_c:.4f} ± {best_std:.4f}")

    print(f"\nVariance by factor (lower = better):")
    print(f"  1. Training noise:        {results['training_noise']['std']:.4f}")
    print(f"  2. Random censoring:      {results['random_censor']['std']:.4f}")
    print(f"  3. Deterministic censor:  {results['determ_censor']['std']:.4f}")
    print(f"  4. One-shot AL:           {results['oneshot']['std']:.4f}")
    print(f"  5. Multi-round AL:        {results['multiround']['std']:.4f}")

    print(f"\nMean improvement by factor:")
    print(f"  Random censoring:      {results['random_censor']['mean']:+.4f}")
    print(f"  Deterministic censor:  {results['determ_censor']['mean']:+.4f}")
    print(f"  One-shot AL:           {results['oneshot']['mean']:+.4f}")
    print(f"  Multi-round AL:        {results['multiround']['mean']:+.4f}")

    print(f"\nProbe depth effect:")
    for k, v in results['probe_depth'].items():
        print(f"  {k}: {v['mean']:+.4f} ± {v['std']:.4f}")

    print(f"\nCensoring proportion effect:")
    for k, v in results['censor_proportion'].items():
        print(f"  {int(k*100)}% censored: {v['mean']:+.4f} ± {v['std']:.4f}")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()
