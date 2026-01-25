"""
Multi-Round Active Learning Experiment

This experiment tests whether multi-round AL (instead of one-shot)
provides enough signal to differentiate between acquisition strategies.

Key change: Instead of query once → evaluate, we do:
  for round in range(n_rounds):
      query batch → update labels → retrain
  then evaluate final model
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
from sklearn.preprocessing import StandardScaler

from model import BayesMtlr, mtlr_nll


def load_nacd_data():
    """Load NACD data directly from CSV."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'MIMIC', 'NACD', 'NACD_Full.csv')
    data = pd.read_csv(data_path)
    data = data.rename(columns={'SURVIVAL': 'time', 'CENSORED': 'event'})

    for col in data.columns:
        if col not in ['time', 'event']:
            data[col] = data[col].replace(-1, 0)
            data[col] = data[col].fillna(0)

    feature_cols = [c for c in data.columns if c not in ['time', 'event']]
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])

    return data.astype(float)


class SimpleArgs:
    """Model configuration."""
    def __init__(self):
        self.lr = 0.001
        self.weight_decay = 0.001
        self.dropout = 0.2
        self.device = 'cpu'
        self.batch_size = 32
        self.n_samples_train = 5
        self.hidden_size = 64
        self.rho_scale = -5.0
        self.mu_scale = None
        self.sigma_1 = 1.0
        self.sigma_2 = 0.0025


class ExperimentConfig:
    """Configuration for multi-round AL."""
    def __init__(self):
        self.n_runs = 10
        self.n_rounds = 5  # Number of AL rounds
        self.batch_size_per_round = 20  # Queries per round
        self.train_epochs = 30
        self.ensemble_size = 5
        self.artificial_censor_proportion = 0.6
        self.probe_depth = 3
        self.lr = 0.001
        self.weight_decay = 0.001
        self.dropout = 0.2


def artificially_censor(time, event, proportion=0.5, random_state=None):
    """Artificially censor data."""
    if random_state is not None:
        np.random.seed(random_state)

    n = len(time)
    artificial_time = time.copy().astype(float)
    artificial_event = event.copy()
    can_query = np.ones(n, dtype=bool)

    eligible = np.where(time > 0)[0]
    n_to_censor = int(len(eligible) * proportion)

    if n_to_censor == 0:
        for i in range(n):
            if artificial_event[i] == 1:
                can_query[i] = False
        return artificial_time, artificial_event, can_query

    idx_to_censor = np.random.choice(eligible, min(n_to_censor, len(eligible)), replace=False)

    for idx in idx_to_censor:
        if time[idx] > 0:
            # Midpoint censoring (deterministic)
            artificial_time[idx] = time[idx] / 2.0
        artificial_event[idx] = 0

    for i in range(n):
        if artificial_event[i] == 1:
            can_query[i] = False
        elif np.isclose(artificial_time[i], time[i]) and event[i] == 0:
            can_query[i] = False

    return artificial_time, artificial_event, can_query


class SimpleOracle:
    """Oracle with probe depth constraint."""
    def __init__(self, true_time, true_event, probe_depth):
        self.true_time = true_time
        self.true_event = true_event
        self.probe_depth = probe_depth

    def query(self, indices, current_time, current_event):
        updated_time = current_time.copy()
        updated_event = current_event.copy()

        for idx in indices:
            c = current_time[idx]
            max_observable = c + self.probe_depth

            if self.true_event[idx] == 1:
                if self.true_time[idx] <= max_observable:
                    updated_time[idx] = self.true_time[idx]
                    updated_event[idx] = 1
                else:
                    updated_time[idx] = max_observable
                    updated_event[idx] = 0
            else:
                updated_time[idx] = min(self.true_time[idx], max_observable)
                updated_event[idx] = 0

        return updated_time, updated_event


def make_time_bins(times, num_bins=15):
    """Create time bins."""
    bins = np.quantile(times[times > 0], np.linspace(0, 1, num_bins + 1))
    bins = np.unique(bins)
    return bins[1:]


def discretize_times(times, bins):
    """Convert continuous times to bin indices."""
    return np.searchsorted(bins, times)


def encode_survival(time_bins, event, num_bins):
    """Encode survival data for MTLR."""
    n = len(time_bins)
    y = np.zeros((n, num_bins))
    for i in range(n):
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


def train_model(model, X_train, y_train, config, epochs=30):
    """Train model."""
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

def entropy_score(preds, time_bins, can_query):
    """Standard entropy."""
    mean_preds = preds.mean(axis=0)
    entropy = -np.sum(xlogy(mean_preds, mean_preds), axis=1)
    entropy[~can_query] = -np.inf
    return entropy


def variance_score(preds, time_bins, can_query):
    """Variance-based."""
    variance = preds.var(axis=0).mean(axis=1)
    variance[~can_query] = -np.inf
    return variance


def batchbald_score(preds, time_bins, can_query, probe_depth):
    """BatchBALD with oracle-aware entropy."""
    K, N, T = preds.shape
    scores = np.zeros(N)

    for i in range(N):
        if not can_query[i]:
            scores[i] = -np.inf
            continue

        c = int(time_bins[i])
        oracle_probs_all = np.zeros((K, probe_depth + 1))

        for k in range(K):
            probs = preds[k, i].copy()
            probs[:c+1] = 0
            probs = probs / (probs.sum() + 1e-8)

            oracle_probs = np.zeros(probe_depth + 1)
            for j in range(probe_depth):
                bin_idx = c + 1 + j
                if bin_idx < T:
                    oracle_probs[j] = probs[bin_idx]

            max_obs = c + probe_depth
            if max_obs + 1 < T:
                oracle_probs[-1] = probs[max_obs + 1:].sum()

            oracle_probs = oracle_probs / (oracle_probs.sum() + 1e-8)
            oracle_probs_all[k] = oracle_probs

        mean_probs = oracle_probs_all.mean(axis=0)
        entropy_expected = -np.sum(xlogy(mean_probs, mean_probs))
        conditional_entropies = -np.sum(xlogy(oracle_probs_all, oracle_probs_all), axis=1)
        expected_conditional = conditional_entropies.mean()

        scores[i] = entropy_expected - expected_conditional

    return scores


def random_score(n, can_query):
    """Random scores."""
    scores = np.random.rand(n)
    scores[~can_query] = -np.inf
    return scores


def update_can_query(can_query, current_time, current_event, true_time, true_event):
    """Update can_query mask after oracle query."""
    n = len(can_query)
    new_can_query = can_query.copy()

    for i in range(n):
        if current_event[i] == 1:
            # Now uncensored, can't query
            new_can_query[i] = False
        elif np.isclose(current_time[i], true_time[i]) and true_event[i] == 0:
            # At true censoring time, nothing more to learn
            new_can_query[i] = False

    return new_can_query


def run_multi_round_experiment(data, config, strategy_name, run_seed=42):
    """Run a multi-round AL experiment."""
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

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
    true_time = time_train.copy()
    true_event = event_train.copy()

    # Artificially censor
    current_time, current_event, can_query = artificially_censor(
        time_train, event_train,
        proportion=config.artificial_censor_proportion,
        random_state=run_seed
    )

    # Create time bins
    bins = make_time_bins(time_train, num_bins=15)
    num_bins = len(bins)

    # Discretize
    current_time_bins = discretize_times(current_time, bins)
    true_time_bins = discretize_times(true_time, bins)
    test_time_bins = discretize_times(time_test, bins)

    # Initial training
    y_train = encode_survival(current_time_bins, current_event, num_bins)
    y_test = encode_survival(test_time_bins, event_test, num_bins)

    args = SimpleArgs()
    model = BayesMtlr(in_features=X_train.shape[1], num_time_bins=num_bins, config=args)
    model = train_model(model, X_train, y_train, config, epochs=config.train_epochs)

    # Initial evaluation
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.FloatTensor(X_test), sample=True, n_samples=10)
        test_preds = torch.softmax(test_preds.mean(0), dim=1).numpy()
    initial_c_index = compute_c_index(test_preds, test_time_bins, event_test)

    # Oracle
    oracle = SimpleOracle(true_time_bins, true_event, config.probe_depth)

    # Track metrics per round
    round_metrics = []
    total_revealed = 0
    total_queried = 0

    # Multi-round AL loop
    for round_num in range(config.n_rounds):
        # Get predictions for acquisition
        train_preds = get_ensemble_predictions(model, X_train, n_samples=config.ensemble_size)

        # Compute scores
        if strategy_name == 'random':
            scores = random_score(len(time_train), can_query)
        elif strategy_name == 'entropy':
            scores = entropy_score(train_preds, current_time_bins, can_query)
        elif strategy_name == 'variance':
            scores = variance_score(train_preds, current_time_bins, can_query)
        elif strategy_name == 'batchbald':
            scores = batchbald_score(train_preds, current_time_bins, can_query, config.probe_depth)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        # Select batch
        valid_mask = scores > -np.inf
        n_valid = valid_mask.sum()
        batch_size_use = min(config.batch_size_per_round, n_valid)

        if batch_size_use == 0:
            break

        selected = np.argsort(scores)[-batch_size_use:]
        total_queried += batch_size_use

        # Query oracle
        new_time_bins, new_event = oracle.query(selected, current_time_bins, current_event)

        # Count reveals
        n_revealed = ((new_event - current_event) > 0).sum()
        total_revealed += n_revealed

        # Update current state
        current_time_bins = new_time_bins
        current_event = new_event

        # Update can_query
        can_query = update_can_query(can_query, current_time_bins, current_event,
                                      true_time_bins, true_event)

        # Retrain model
        y_train = encode_survival(current_time_bins, current_event, num_bins)
        model = BayesMtlr(in_features=X_train.shape[1], num_time_bins=num_bins, config=args)
        model = train_model(model, X_train, y_train, config, epochs=config.train_epochs)

        # Evaluate after this round
        model.eval()
        with torch.no_grad():
            test_preds = model(torch.FloatTensor(X_test), sample=True, n_samples=10)
            test_preds = torch.softmax(test_preds.mean(0), dim=1).numpy()
        round_c_index = compute_c_index(test_preds, test_time_bins, event_test)

        round_metrics.append({
            'round': round_num + 1,
            'c_index': round_c_index,
            'n_revealed': n_revealed,
            'n_queryable': can_query.sum()
        })

    # Final evaluation
    final_c_index = round_metrics[-1]['c_index'] if round_metrics else initial_c_index

    return {
        'initial_c_index': initial_c_index,
        'final_c_index': final_c_index,
        'improvement': final_c_index - initial_c_index,
        'total_revealed': total_revealed,
        'total_queried': total_queried,
        'round_metrics': round_metrics
    }


def main():
    print("\n" + "="*100)
    print("MULTI-ROUND ACTIVE LEARNING EXPERIMENT")
    print("="*100)
    print(f"Started: {datetime.now()}")

    # Load data
    print("\nLoading NACD dataset...")
    data = load_nacd_data()
    print(f"Loaded: {data.shape[0]} samples, {data.shape[1]-2} features")

    config = ExperimentConfig()

    print(f"\nConfiguration:")
    print(f"  AL Rounds: {config.n_rounds}")
    print(f"  Batch size per round: {config.batch_size_per_round}")
    print(f"  Total queries: {config.n_rounds * config.batch_size_per_round}")
    print(f"  Probe depth: {config.probe_depth}")
    print(f"  Runs: {config.n_runs}")

    strategies = ['random', 'entropy', 'variance', 'batchbald']
    all_results = {s: [] for s in strategies}

    print(f"\nRunning experiments...")

    for run in range(config.n_runs):
        print(f"\n--- Run {run+1}/{config.n_runs} ---")
        for strategy in strategies:
            result = run_multi_round_experiment(data, config, strategy, run_seed=42 + run)
            all_results[strategy].append(result)
            print(f"  {strategy:<12}: Δ={result['improvement']:+.4f}, revealed={result['total_revealed']}")

    # Analyze results
    print("\n" + "="*100)
    print("FINAL RESULTS")
    print("="*100)

    print(f"\n{'Strategy':<12} {'Mean Δ':<12} {'Std':<10} {'Wins':<8} {'Revealed':<10} {'SNR':<8}")
    print("-" * 70)

    summaries = {}
    for strategy in strategies:
        improvements = [r['improvement'] for r in all_results[strategy]]
        revealed = [r['total_revealed'] for r in all_results[strategy]]

        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        wins = sum(1 for imp in improvements if imp > 0)
        mean_rev = np.mean(revealed)
        snr = abs(mean_imp) / (std_imp + 1e-10)

        summaries[strategy] = {
            'mean': mean_imp, 'std': std_imp, 'wins': wins,
            'revealed': mean_rev, 'snr': snr, 'n': len(improvements)
        }

        print(f"{strategy:<12} {mean_imp:+.4f}      {std_imp:.4f}    {wins}/{len(improvements):<4} {mean_rev:.1f}       {snr:.3f}")

    # Statistical tests
    print("\nStatistical Tests (paired t-test):")
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i+1:]:
            imp1 = [r['improvement'] for r in all_results[s1]]
            imp2 = [r['improvement'] for r in all_results[s2]]
            t_stat, p_val = ttest_rel(imp1, imp2)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            print(f"  {s1} vs {s2}: t={t_stat:.3f}, p={p_val:.4f} {sig}")

    # Round-by-round analysis
    print("\n" + "="*100)
    print("ROUND-BY-ROUND IMPROVEMENT")
    print("="*100)

    print(f"\n{'Round':<8}", end="")
    for s in strategies:
        print(f"{s:<12}", end="")
    print()
    print("-" * 60)

    for r in range(config.n_rounds):
        print(f"{r+1:<8}", end="")
        for strategy in strategies:
            round_improvements = []
            for result in all_results[strategy]:
                if r < len(result['round_metrics']):
                    round_improvements.append(
                        result['round_metrics'][r]['c_index'] - result['initial_c_index']
                    )
            if round_improvements:
                mean_imp = np.mean(round_improvements)
                print(f"{mean_imp:+.4f}      ", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()

    # Compare to one-shot
    print("\n" + "="*100)
    print("COMPARISON: MULTI-ROUND vs ONE-SHOT")
    print("="*100)

    print("\nMulti-round (5 rounds × 20 queries = 100 total):")
    for s in strategies:
        print(f"  {s}: mean={summaries[s]['mean']:+.4f}, std={summaries[s]['std']:.4f}, SNR={summaries[s]['snr']:.3f}")

    print("\nKey Question: Is SNR > 1.0 (meaningful signal)?")
    for s in strategies:
        status = "YES ✓" if summaries[s]['snr'] > 1.0 else "NO ✗"
        print(f"  {s}: SNR={summaries[s]['snr']:.3f} → {status}")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()
