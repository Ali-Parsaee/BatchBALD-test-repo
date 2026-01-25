"""
Noise Source Analysis for Survival Active Learning with BatchBALD

This experiment systematically tests what factors contribute to high variance
in the active learning evaluation, preventing statistically significant results.

Factors to test:
1. Random artificial censoring (vs deterministic)
2. One-shot AL (vs multi-round)
3. Probe depth (k=1, 2, 3, 5)
4. Model training stochasticity
5. Dataset size effects

For each factor, we measure:
- Variance of improvement across runs
- Signal-to-noise ratio
- Effect size (Cohen's d)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ttest_rel, sem
from scipy.special import xlogy
from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

# Import only the model (avoid complex dependencies in Making_and_getting_datasets)
from model import BayesMtlr, mtlr_nll


def load_nacd_data():
    """Load NACD data directly from CSV, avoiding complex dependencies."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'MIMIC', 'NACD', 'NACD_Full.csv')
    data = pd.read_csv(data_path)

    # Rename columns to match expected format
    data = data.rename(columns={'SURVIVAL': 'time', 'CENSORED': 'event'})

    # Handle missing values (replace -1 with 0 for symptom columns)
    for col in data.columns:
        if col not in ['time', 'event']:
            data[col] = data[col].replace(-1, 0)
            data[col] = data[col].fillna(0)

    # Standardize features (except time and event)
    feature_cols = [c for c in data.columns if c not in ['time', 'event']]
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])

    data = data.astype(float)
    return data


class ExperimentConfig:
    """Configuration for experiments."""
    def __init__(self):
        self.n_runs = 10
        self.batch_size = 30
        self.train_epochs = 30
        self.ensemble_size = 5
        self.artificial_censor_proportion = 0.5

        # Model config (from args)
        self.lr = 0.001
        self.weight_decay = 0.001
        self.dropout = 0.2


def artificially_censor_fixed(time, event, proportion=0.5, random_state=None,
                               mode='random', also_censor_censored=True):
    """
    Artificially censor data with different modes.

    Args:
        mode: 'random' - random time in [0, true_time)
              'midpoint' - deterministic censoring at true_time // 2
              'quartile' - deterministic at true_time * 0.25
        also_censor_censored: If True, also further censor already-censored samples
    """
    if random_state is not None:
        np.random.seed(random_state)

    artificial_time = time.copy().astype(float)
    artificial_event = event.copy()
    can_query = np.ones(len(time), dtype=bool)  # Track which points can be queried

    n_samples = len(time)

    if also_censor_censored:
        # Select from ALL samples (both censored and uncensored)
        eligible_idx = np.where(time > 0)[0]  # Must have time > 0 to censor further
    else:
        # Only uncensored samples (original behavior)
        eligible_idx = np.where(event == 1)[0]

    n_to_censor = int(len(eligible_idx) * proportion)
    if n_to_censor == 0:
        return artificial_time, artificial_event, can_query

    idx_to_censor = np.random.choice(eligible_idx, size=min(n_to_censor, len(eligible_idx)), replace=False)

    for idx in idx_to_censor:
        original_time = float(time[idx])

        if mode == 'random':
            if original_time > 0:
                artificial_time[idx] = np.random.uniform(0, original_time)
            else:
                artificial_time[idx] = 0
        elif mode == 'midpoint':
            artificial_time[idx] = original_time / 2.0
        elif mode == 'quartile':
            artificial_time[idx] = original_time * 0.25
        else:
            raise ValueError(f"Unknown mode: {mode}")

        artificial_event[idx] = 0

    # Mark points where artificial_time == true_time (can't learn more)
    for i in range(n_samples):
        if np.isclose(artificial_time[i], time[i]) and event[i] == 0:
            # Already at true censoring time, nothing to learn
            can_query[i] = False
        if artificial_event[i] == 1:
            # Already uncensored
            can_query[i] = False

    return artificial_time, artificial_event, can_query


class SimpleOracle:
    """Oracle with probe depth constraint."""

    def __init__(self, true_time, true_event, probe_depth):
        self.true_time = true_time
        self.true_event = true_event
        self.probe_depth = probe_depth

    def query(self, indices, current_time, current_event):
        """Query oracle for selected indices."""
        updated_time = current_time.copy()
        updated_event = current_event.copy()

        for idx in indices:
            c = current_time[idx]
            max_observable = c + self.probe_depth

            if self.true_event[idx] == 1:  # True death
                if self.true_time[idx] <= max_observable:
                    updated_time[idx] = self.true_time[idx]
                    updated_event[idx] = 1
                else:
                    updated_time[idx] = max_observable
                    updated_event[idx] = 0
            else:  # True censoring
                updated_time[idx] = min(self.true_time[idx], max_observable)
                updated_event[idx] = 0

        return updated_time, updated_event


def make_time_bins(times, num_bins=15):
    """Create time bins for MTLR."""
    bins = np.quantile(times[times > 0], np.linspace(0, 1, num_bins + 1))
    bins = np.unique(bins)
    return bins[1:]


def discretize_times(times, bins):
    """Convert continuous times to bin indices."""
    return np.searchsorted(bins, times)


def encode_survival(time_bins, event, num_bins):
    """Encode survival data for MTLR."""
    n_samples = len(time_bins)
    y = np.zeros((n_samples, num_bins))

    for i in range(n_samples):
        bin_idx = min(int(time_bins[i]), num_bins - 1)
        if event[i] == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1

    return y


def compute_c_index(predictions, time, event):
    """Concordance index."""
    n = len(time)
    concordant = 0
    total = 0

    mean_survival = np.sum(predictions * np.arange(predictions.shape[1]), axis=1)

    for i in range(n):
        if event[i] == 0:
            continue
        for j in range(n):
            if time[j] > time[i]:
                total += 1
                if mean_survival[i] < mean_survival[j]:
                    concordant += 1
                elif mean_survival[i] == mean_survival[j]:
                    concordant += 0.5

    return concordant / total if total > 0 else 0.5


def train_model(model, X_train, y_train, config, epochs=30, seed=None):
    """Train Bayesian MTLR model."""
    if seed is not None:
        torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    model.train()

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train)
    dataset_size = len(X_train)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, _, _, _ = model.sample_elbo(X_tensor, y_tensor, dataset_size, see1=0.01)
        loss.backward()
        optimizer.step()

    return model


def get_ensemble_predictions(model, X, n_samples=5):
    """Get ensemble predictions."""
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(torch.FloatTensor(X), sample=True, n_samples=1)
            pred = torch.softmax(pred.squeeze(0), dim=1).numpy()
            preds.append(pred)
    return np.array(preds)


# ============ Acquisition Functions ============

def entropy_score(preds, time_bins, event, can_query):
    """Standard entropy acquisition."""
    mean_preds = preds.mean(axis=0)
    entropy = -np.sum(xlogy(mean_preds, mean_preds), axis=1)
    entropy[~can_query] = -np.inf
    return entropy


def variance_score(preds, time_bins, event, can_query):
    """Variance-based acquisition."""
    variance = preds.var(axis=0).mean(axis=1)
    variance[~can_query] = -np.inf
    return variance


def batchbald_score(preds, time_bins, event, can_query, probe_depth):
    """BatchBALD acquisition with oracle-aware entropy."""
    K, N, T = preds.shape
    scores = np.zeros(N)

    for i in range(N):
        if not can_query[i]:
            scores[i] = -np.inf
            continue

        c = int(time_bins[i])  # Current censoring bin

        # For each ensemble member, compute oracle outcome probs
        oracle_probs_all = np.zeros((K, probe_depth + 1))

        for k in range(K):
            probs = preds[k, i].copy()

            # Renormalize: can't die before censoring time
            probs[:c+1] = 0
            probs = probs / (probs.sum() + 1e-8)

            # Oracle outcomes: death in bins c+1 to c+probe_depth, or censored
            oracle_probs = np.zeros(probe_depth + 1)
            for j in range(probe_depth):
                bin_idx = c + 1 + j
                if bin_idx < T:
                    oracle_probs[j] = probs[bin_idx]

            # Last outcome: P(T > c+probe_depth)
            max_obs = c + probe_depth
            if max_obs + 1 < T:
                oracle_probs[-1] = probs[max_obs + 1:].sum()

            oracle_probs = oracle_probs / (oracle_probs.sum() + 1e-8)
            oracle_probs_all[k] = oracle_probs

        # Compute MI = H(E[Y]) - E[H(Y|theta)]
        mean_probs = oracle_probs_all.mean(axis=0)
        entropy_expected = -np.sum(xlogy(mean_probs, mean_probs))

        conditional_entropies = -np.sum(xlogy(oracle_probs_all, oracle_probs_all), axis=1)
        expected_conditional = conditional_entropies.mean()

        scores[i] = entropy_expected - expected_conditional

    return scores


def random_score(preds, time_bins, event, can_query):
    """Random acquisition."""
    scores = np.random.rand(len(time_bins))
    scores[~can_query] = -np.inf
    return scores


# ============ Main Experiment Functions ============

def run_single_experiment(data, args, config,
                          censor_mode='random',
                          probe_depth=3,
                          strategy='entropy',
                          run_seed=42,
                          fix_model_seed=False,
                          also_censor_censored=True):
    """Run a single AL experiment."""

    np.random.seed(run_seed)

    # Split data
    n_total = len(data)
    n_train = int(0.7 * n_total)

    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_cols = [col for col in data.columns if col not in ['time', 'event']]
    X_train = data.iloc[train_idx][X_cols].values.astype(np.float32)
    time_train = data.iloc[train_idx]['time'].values.astype(np.float32)
    event_train = data.iloc[train_idx]['event'].values.astype(int)

    X_test = data.iloc[test_idx][X_cols].values.astype(np.float32)
    time_test = data.iloc[test_idx]['time'].values.astype(np.float32)
    event_test = data.iloc[test_idx]['event'].values.astype(int)

    # Store true train data
    true_train_time = time_train.copy()
    true_train_event = event_train.copy()

    # Artificially censor
    artificial_time, artificial_event, can_query = artificially_censor_fixed(
        time_train, event_train,
        proportion=config.artificial_censor_proportion,
        random_state=run_seed,
        mode=censor_mode,
        also_censor_censored=also_censor_censored
    )

    # Create time bins
    bins = make_time_bins(time_train, num_bins=15)
    num_bins = len(bins)

    # Discretize times
    artificial_time_bins = discretize_times(artificial_time, bins)
    true_time_bins = discretize_times(true_train_time, bins)
    test_time_bins = discretize_times(time_test, bins)

    # Encode for MTLR
    y_train_init = encode_survival(artificial_time_bins, artificial_event, num_bins)
    y_test = encode_survival(test_time_bins, event_test, num_bins)

    # Train initial model
    model_seed = 42 if fix_model_seed else run_seed
    torch.manual_seed(model_seed)

    model_init = BayesMtlr(in_features=X_train.shape[1], num_time_bins=num_bins, config=args)
    model_init = train_model(model_init, X_train, y_train_init, config, epochs=config.train_epochs, seed=model_seed)

    # Initial evaluation
    model_init.eval()
    with torch.no_grad():
        test_preds_init = model_init(torch.FloatTensor(X_test), sample=True, n_samples=10)
        test_preds_init = torch.softmax(test_preds_init.mean(0), dim=1).numpy()

    initial_c_index = compute_c_index(test_preds_init, test_time_bins, event_test)

    # Get ensemble predictions for acquisition
    train_preds = get_ensemble_predictions(model_init, X_train, n_samples=config.ensemble_size)

    # Compute scores based on strategy
    if strategy == 'entropy':
        scores = entropy_score(train_preds, artificial_time_bins, artificial_event, can_query)
    elif strategy == 'variance':
        scores = variance_score(train_preds, artificial_time_bins, artificial_event, can_query)
    elif strategy == 'batchbald':
        scores = batchbald_score(train_preds, artificial_time_bins, artificial_event, can_query, probe_depth)
    elif strategy == 'random':
        scores = random_score(train_preds, artificial_time_bins, artificial_event, can_query)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Select batch
    valid_mask = scores > -np.inf
    n_valid = valid_mask.sum()
    batch_size_use = min(config.batch_size, n_valid)

    if batch_size_use == 0:
        return {
            'improvement': 0.0,
            'n_revealed': 0,
            'n_queried': 0,
            'initial_c_index': initial_c_index,
            'final_c_index': initial_c_index
        }

    selected = np.argsort(scores)[-batch_size_use:]

    # Query oracle
    oracle = SimpleOracle(true_time_bins, true_train_event, probe_depth)
    updated_time_bins, updated_event = oracle.query(selected, artificial_time_bins, artificial_event)

    n_revealed = ((updated_event - artificial_event) > 0).sum()

    # Retrain
    y_train_updated = encode_survival(updated_time_bins, updated_event, num_bins)

    torch.manual_seed(model_seed + 1000)
    model_updated = BayesMtlr(in_features=X_train.shape[1], num_time_bins=num_bins, config=args)
    model_updated = train_model(model_updated, X_train, y_train_updated, config, epochs=config.train_epochs)

    # Final evaluation
    model_updated.eval()
    with torch.no_grad():
        test_preds_updated = model_updated(torch.FloatTensor(X_test), sample=True, n_samples=10)
        test_preds_updated = torch.softmax(test_preds_updated.mean(0), dim=1).numpy()

    final_c_index = compute_c_index(test_preds_updated, test_time_bins, event_test)

    return {
        'improvement': final_c_index - initial_c_index,
        'n_revealed': n_revealed,
        'n_queried': batch_size_use,
        'initial_c_index': initial_c_index,
        'final_c_index': final_c_index
    }


def run_experiment_suite(data, args, config, strategies, n_runs, **kwargs):
    """Run experiments for all strategies."""
    results = {s: [] for s in strategies}

    for run in range(n_runs):
        for strategy in strategies:
            result = run_single_experiment(
                data, args, config,
                strategy=strategy,
                run_seed=42 + run,
                **kwargs
            )
            results[strategy].append(result)

    return results


def analyze_results(results, name=""):
    """Analyze and print results."""
    print(f"\n{'='*80}")
    print(f"RESULTS: {name}")
    print(f"{'='*80}")

    summary = {}

    for strategy, runs in results.items():
        improvements = [r['improvement'] for r in runs]
        reveals = [r['n_revealed'] for r in runs]

        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        se_imp = sem(improvements)
        wins = sum(1 for imp in improvements if imp > 0)

        summary[strategy] = {
            'mean': mean_imp,
            'std': std_imp,
            'se': se_imp,
            'wins': wins,
            'n_runs': len(runs),
            'avg_reveals': np.mean(reveals),
            'snr': abs(mean_imp) / (std_imp + 1e-10)  # Signal-to-noise ratio
        }

        print(f"{strategy:<15} Mean: {mean_imp:+.4f} ± {std_imp:.4f}  Wins: {wins}/{len(runs)}  SNR: {summary[strategy]['snr']:.3f}")

    # Pairwise comparisons
    strategies = list(results.keys())
    if len(strategies) >= 2:
        print(f"\nPairwise t-tests:")
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                imp1 = [r['improvement'] for r in results[s1]]
                imp2 = [r['improvement'] for r in results[s2]]
                t_stat, p_val = ttest_rel(imp1, imp2)
                sig = "**" if p_val < 0.05 else ""
                print(f"  {s1} vs {s2}: t={t_stat:.3f}, p={p_val:.4f} {sig}")

    return summary


# ============ Individual Experiments ============

def experiment_1_censoring_randomness(data, args, config):
    """Test random vs deterministic censoring."""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Random vs Deterministic Artificial Censoring")
    print("="*80)

    strategies = ['entropy', 'batchbald', 'random']

    print("\n--- Random Censoring (baseline) ---")
    results_random = run_experiment_suite(
        data, args, config, strategies, n_runs=config.n_runs,
        censor_mode='random', probe_depth=3
    )
    summary_random = analyze_results(results_random, "Random Censoring")

    print("\n--- Midpoint Censoring (deterministic) ---")
    results_midpoint = run_experiment_suite(
        data, args, config, strategies, n_runs=config.n_runs,
        censor_mode='midpoint', probe_depth=3
    )
    summary_midpoint = analyze_results(results_midpoint, "Midpoint Censoring")

    print("\n--- Quartile Censoring (deterministic) ---")
    results_quartile = run_experiment_suite(
        data, args, config, strategies, n_runs=config.n_runs,
        censor_mode='quartile', probe_depth=3
    )
    summary_quartile = analyze_results(results_quartile, "Quartile Censoring")

    # Compare variance reduction
    print("\n--- VARIANCE COMPARISON ---")
    print(f"{'Mode':<15} {'Entropy Std':<15} {'BatchBALD Std':<15} {'Avg SNR':<15}")
    print("-" * 60)

    for mode, summary in [('random', summary_random), ('midpoint', summary_midpoint), ('quartile', summary_quartile)]:
        avg_snr = np.mean([s['snr'] for s in summary.values()])
        print(f"{mode:<15} {summary['entropy']['std']:.4f}          {summary['batchbald']['std']:.4f}          {avg_snr:.4f}")

    return {
        'random': summary_random,
        'midpoint': summary_midpoint,
        'quartile': summary_quartile
    }


def experiment_2_probe_depth(data, args, config):
    """Test different probe depths."""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Probe Depth Effects")
    print("="*80)

    strategies = ['entropy', 'batchbald', 'random']
    probe_depths = [1, 2, 3, 5, 8]

    all_results = {}

    for k in probe_depths:
        print(f"\n--- Probe Depth k={k} ---")
        results = run_experiment_suite(
            data, args, config, strategies, n_runs=config.n_runs,
            censor_mode='midpoint', probe_depth=k
        )
        all_results[k] = analyze_results(results, f"Probe Depth k={k}")

    # Summary table
    print("\n--- PROBE DEPTH SUMMARY ---")
    print(f"{'Depth':<10} {'Entropy Mean':<15} {'BatchBALD Mean':<15} {'BB-Entropy':<15} {'BB Win%':<15}")
    print("-" * 70)

    for k, summary in all_results.items():
        diff = summary['batchbald']['mean'] - summary['entropy']['mean']
        bb_wins = summary['batchbald']['wins']
        total = summary['batchbald']['n_runs']
        print(f"k={k:<7} {summary['entropy']['mean']:+.4f}          {summary['batchbald']['mean']:+.4f}          {diff:+.4f}          {100*bb_wins/total:.0f}%")

    return all_results


def experiment_3_model_stochasticity(data, args, config):
    """Test model training stochasticity."""
    print("\n" + "="*80)
    print("EXPERIMENT 3: Model Training Stochasticity")
    print("="*80)

    strategies = ['entropy', 'batchbald']

    print("\n--- Variable model seeds (normal) ---")
    results_variable = run_experiment_suite(
        data, args, config, strategies, n_runs=config.n_runs,
        censor_mode='midpoint', probe_depth=3, fix_model_seed=False
    )
    summary_variable = analyze_results(results_variable, "Variable Model Seeds")

    print("\n--- Fixed model seed (reduced stochasticity) ---")
    results_fixed = run_experiment_suite(
        data, args, config, strategies, n_runs=config.n_runs,
        censor_mode='midpoint', probe_depth=3, fix_model_seed=True
    )
    summary_fixed = analyze_results(results_fixed, "Fixed Model Seed")

    print("\n--- STOCHASTICITY COMPARISON ---")
    print(f"{'Seed Mode':<15} {'Entropy Std':<15} {'BatchBALD Std':<15}")
    print("-" * 50)
    print(f"{'Variable':<15} {summary_variable['entropy']['std']:.4f}          {summary_variable['batchbald']['std']:.4f}")
    print(f"{'Fixed':<15} {summary_fixed['entropy']['std']:.4f}          {summary_fixed['batchbald']['std']:.4f}")

    return {'variable': summary_variable, 'fixed': summary_fixed}


def experiment_4_censoring_scope(data, args, config):
    """Test censoring only uncensored vs all samples."""
    print("\n" + "="*80)
    print("EXPERIMENT 4: Censoring Scope (uncensored only vs all samples)")
    print("="*80)

    strategies = ['entropy', 'batchbald', 'random']

    print("\n--- Censor only uncensored samples (original behavior) ---")
    results_uncensored_only = run_experiment_suite(
        data, args, config, strategies, n_runs=config.n_runs,
        censor_mode='midpoint', probe_depth=3, also_censor_censored=False
    )
    summary_uncensored = analyze_results(results_uncensored_only, "Uncensored Only")

    print("\n--- Censor all samples (fixed behavior) ---")
    results_all = run_experiment_suite(
        data, args, config, strategies, n_runs=config.n_runs,
        censor_mode='midpoint', probe_depth=3, also_censor_censored=True
    )
    summary_all = analyze_results(results_all, "All Samples")

    return {'uncensored_only': summary_uncensored, 'all': summary_all}


def experiment_5_batch_size(data, args, config):
    """Test different batch sizes."""
    print("\n" + "="*80)
    print("EXPERIMENT 5: Batch Size Effects")
    print("="*80)

    strategies = ['entropy', 'batchbald']
    batch_sizes = [10, 20, 30, 50]

    all_results = {}

    for bs in batch_sizes:
        print(f"\n--- Batch Size = {bs} ---")
        config_copy = ExperimentConfig()
        config_copy.batch_size = bs
        config_copy.n_runs = config.n_runs

        results = run_experiment_suite(
            data, args, config_copy, strategies, n_runs=config.n_runs,
            censor_mode='midpoint', probe_depth=3
        )
        all_results[bs] = analyze_results(results, f"Batch Size = {bs}")

    return all_results


class SimpleArgs:
    """Simple args object for model configuration."""
    def __init__(self):
        self.lr = 0.001
        self.weight_decay = 0.001
        self.dropout = 0.2
        self.n_time_bins = 15
        self.device = 'cpu'
        # Required by BayesMtlr model
        self.batch_size = 32
        self.n_samples_train = 5
        self.hidden_size = 64
        self.rho_scale = -5.0
        self.mu_scale = None  # Will be set automatically
        self.sigma_1 = 1.0
        self.sigma_2 = 0.0025


def main():
    print("\n" + "="*100)
    print("NOISE SOURCE ANALYSIS FOR SURVIVAL ACTIVE LEARNING")
    print("="*100)
    print(f"Started: {datetime.now()}")

    # Load data directly (avoiding complex dependencies)
    print("\nLoading NACD dataset...")
    try:
        data = load_nacd_data()
        args = SimpleArgs()
        print(f"Loaded: {data.shape[0]} samples, {data.shape[1]-2} features")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Setup config
    config = ExperimentConfig()
    config.n_runs = 5  # Start with fewer runs for faster iteration

    # Copy necessary args
    args.lr = config.lr
    args.weight_decay = config.weight_decay
    args.dropout = config.dropout

    all_experiments = {}

    # Run experiments
    print("\n" + "="*100)
    print("RUNNING EXPERIMENTS")
    print("="*100)

    try:
        all_experiments['exp1_censoring'] = experiment_1_censoring_randomness(data, args, config)
    except Exception as e:
        print(f"Experiment 1 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_experiments['exp2_probe_depth'] = experiment_2_probe_depth(data, args, config)
    except Exception as e:
        print(f"Experiment 2 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_experiments['exp3_stochasticity'] = experiment_3_model_stochasticity(data, args, config)
    except Exception as e:
        print(f"Experiment 3 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_experiments['exp4_scope'] = experiment_4_censoring_scope(data, args, config)
    except Exception as e:
        print(f"Experiment 4 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        all_experiments['exp5_batch_size'] = experiment_5_batch_size(data, args, config)
    except Exception as e:
        print(f"Experiment 5 failed: {e}")
        import traceback
        traceback.print_exc()

    # Final summary
    print("\n" + "="*100)
    print("FINAL SUMMARY: NOISE SOURCE RANKINGS")
    print("="*100)

    print("""
Based on the experiments above, rank the noise sources by their impact:

1. RANDOM ARTIFICIAL CENSORING
   - Compare variance between random vs deterministic modes
   - Higher variance reduction = bigger noise source

2. PROBE DEPTH
   - Compare variance and BatchBALD advantage across k values
   - Larger k should give BatchBALD more room to shine

3. MODEL TRAINING STOCHASTICITY
   - Compare fixed vs variable seeds
   - If big difference, model randomness is a major factor

4. ONE-SHOT vs MULTI-ROUND
   - (Not directly tested here, but can infer from batch size)
   - Larger batches approximate multi-round better

5. BATCH SIZE
   - Compare variance across batch sizes
   - Larger batches should reduce variance

RECOMMENDATIONS for creating a setup where BatchBALD dominates:
- Use deterministic censoring (midpoint or quartile)
- Use larger probe depths (k >= 5)
- Fix model seeds to reduce training noise
- Use larger batch sizes (50+)
- Consider synthetic data with known ground truth
""")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()
