"""
Controlled Experiment to Create BatchBALD Advantage

This experiment creates a synthetic setting where BatchBALD SHOULD have
an advantage over entropy/variance:

1. Synthetic data with known ground truth
2. Clear cluster structure (BatchBALD should prefer diverse clusters)
3. Higher probe depths (more information to gain)
4. More training epochs (stable models)
5. More runs for statistical power
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

from model import BayesMtlr


class ExperimentConfig:
    """Configuration for controlled experiment."""
    def __init__(self):
        self.n_samples = 500  # Larger dataset
        self.n_features = 10
        self.n_clusters = 5  # Clear cluster structure
        self.n_time_bins = 10
        self.train_epochs = 100  # More epochs for stable training
        self.batch_size = 50
        self.ensemble_size = 10  # More ensemble members
        self.artificial_censor_proportion = 0.6
        self.probe_depth = 5  # Larger probe depth
        self.lr = 0.001
        self.weight_decay = 0.001
        self.dropout = 0.2
        self.n_runs = 20  # More runs for statistical power


class SimpleArgs:
    """Model configuration."""
    def __init__(self, config):
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.dropout = config.dropout
        self.device = 'cpu'
        self.batch_size = 32
        self.n_samples_train = 5
        self.hidden_size = 64
        self.rho_scale = -5.0
        self.mu_scale = None
        self.sigma_1 = 1.0
        self.sigma_2 = 0.0025


def generate_clustered_survival_data(config, random_state=42):
    """
    Generate synthetic survival data with clear cluster structure.

    Each cluster has different baseline hazards, making it important
    to have diverse samples from each cluster in AL.
    """
    np.random.seed(random_state)

    n_per_cluster = config.n_samples // config.n_clusters
    X_list = []
    time_list = []
    event_list = []
    cluster_list = []

    for c in range(config.n_clusters):
        # Cluster center
        center = np.random.randn(config.n_features) * 2

        # Cluster-specific baseline hazard
        base_hazard = 0.05 * (1 + c)  # Higher hazard for higher cluster indices

        # Generate samples for this cluster
        X_cluster = center + np.random.randn(n_per_cluster, config.n_features) * 0.5

        # Generate survival times (exponential with cluster-specific hazard)
        feature_effect = X_cluster.sum(axis=1) * 0.1
        hazards = base_hazard * np.exp(feature_effect)
        survival_times = np.random.exponential(1.0 / hazards)

        # Discretize to bins
        max_time = config.n_time_bins - 1
        time_bins = np.floor(np.clip(survival_times / np.percentile(survival_times, 95) * max_time, 0, max_time)).astype(int)

        # Random censoring (30% baseline censoring rate)
        events = np.ones(n_per_cluster, dtype=int)
        n_censored = int(n_per_cluster * 0.3)
        censor_idx = np.random.choice(n_per_cluster, n_censored, replace=False)
        events[censor_idx] = 0

        # For censored samples, censoring time is before death time
        for idx in censor_idx:
            if time_bins[idx] > 0:
                time_bins[idx] = np.random.randint(0, time_bins[idx] + 1)

        X_list.append(X_cluster)
        time_list.append(time_bins)
        event_list.append(events)
        cluster_list.append(np.full(n_per_cluster, c))

    X = np.vstack(X_list)
    time = np.concatenate(time_list)
    event = np.concatenate(event_list)
    cluster = np.concatenate(cluster_list)

    # Shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    time = time[perm]
    event = event[perm]
    cluster = cluster[perm]

    return {
        'X': X.astype(np.float32),
        'time': time,
        'event': event,
        'cluster': cluster
    }


def artificially_censor(time, event, proportion=0.5, random_state=None):
    """Artificially censor data with deterministic midpoint censoring."""
    if random_state is not None:
        np.random.seed(random_state)

    n = len(time)
    artificial_time = time.copy().astype(float)
    artificial_event = event.copy()
    can_query = np.ones(n, dtype=bool)

    # Select samples to censor
    eligible = np.where(time > 0)[0]
    n_to_censor = int(len(eligible) * proportion)

    if n_to_censor == 0:
        return artificial_time, artificial_event, can_query

    idx_to_censor = np.random.choice(eligible, min(n_to_censor, len(eligible)), replace=False)

    for idx in idx_to_censor:
        # Midpoint censoring (deterministic)
        artificial_time[idx] = time[idx] / 2.0
        artificial_event[idx] = 0

    # Mark points that can't be queried
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

    # Mean survival time from predictions
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


def train_model(model, X_train, y_train, config, epochs=100, seed=None):
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


def get_ensemble_predictions(model, X, n_samples=10):
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


def batchbald_batch_score(preds, time_bins, can_query, probe_depth, batch_size):
    """
    BatchBALD with batch diversity consideration.

    Uses greedy selection that considers joint entropy.
    """
    K, N, T = preds.shape

    # Compute oracle probs for all points
    oracle_probs_all = np.zeros((K, N, probe_depth + 1))

    for i in range(N):
        if not can_query[i]:
            continue

        c = int(time_bins[i])

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
            oracle_probs_all[k, i] = oracle_probs

    # Greedy batch selection with joint MI
    selected = []
    remaining = set(np.where(can_query)[0])

    for _ in range(min(batch_size, len(remaining))):
        if len(remaining) == 0:
            break

        best_score = -np.inf
        best_idx = None

        for idx in remaining:
            if len(selected) == 0:
                # First point: just use MI
                probs_i = oracle_probs_all[:, idx, :]
                mean_p = probs_i.mean(axis=0)
                H_expected = -np.sum(xlogy(mean_p, mean_p))
                H_conditional = -np.sum(xlogy(probs_i, probs_i), axis=1).mean()
                score = H_expected - H_conditional
            else:
                # Joint MI approximation
                batch_probs = oracle_probs_all[:, selected + [idx], :]

                # H(E[Y_batch])
                mean_batch = batch_probs.mean(axis=0)  # (batch_size+1, probe_depth+1)
                H_expected = -np.sum(xlogy(mean_batch, mean_batch))

                # E[H(Y_batch | theta)]
                H_conditional = -np.sum(xlogy(batch_probs, batch_probs), axis=2).sum(axis=1).mean()

                score = H_expected - H_conditional

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.discard(best_idx)

    return np.array(selected)


def random_score(n, can_query):
    """Random scores."""
    scores = np.random.rand(n)
    scores[~can_query] = -np.inf
    return scores


def cluster_diversity_score(preds, cluster, can_query):
    """
    Score that explicitly encourages cluster diversity.
    This is a "cheating" baseline to see upper bound of diversity benefit.
    """
    scores = np.zeros(len(cluster))

    # Count samples from each cluster in queryable set
    queryable_clusters = cluster[can_query]
    cluster_counts = np.bincount(queryable_clusters.astype(int), minlength=cluster.max()+1)

    # Score inversely proportional to cluster representation
    for i in range(len(cluster)):
        if can_query[i]:
            c = int(cluster[i])
            scores[i] = 1.0 / (cluster_counts[c] + 1)
        else:
            scores[i] = -np.inf

    return scores


def run_experiment(config, run_seed=42):
    """Run a single AL experiment."""
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    # Generate data
    data = generate_clustered_survival_data(config, random_state=run_seed)

    # Split train/test
    n_total = len(data['X'])
    n_train = int(0.7 * n_total)

    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train = data['X'][train_idx]
    time_train = data['time'][train_idx]
    event_train = data['event'][train_idx]
    cluster_train = data['cluster'][train_idx]

    X_test = data['X'][test_idx]
    time_test = data['time'][test_idx]
    event_test = data['event'][test_idx]

    # Artificially censor
    true_time = time_train.copy()
    true_event = event_train.copy()

    artificial_time, artificial_event, can_query = artificially_censor(
        time_train, event_train,
        proportion=config.artificial_censor_proportion,
        random_state=run_seed
    )

    # Encode for MTLR
    y_train_init = encode_survival(artificial_time, artificial_event, config.n_time_bins)
    y_test = encode_survival(time_test, event_test, config.n_time_bins)

    # Create model args
    args = SimpleArgs(config)

    # Train initial model
    model_init = BayesMtlr(in_features=config.n_features, num_time_bins=config.n_time_bins, config=args)
    model_init = train_model(model_init, X_train, y_train_init, config, epochs=config.train_epochs, seed=run_seed)

    # Initial evaluation
    model_init.eval()
    with torch.no_grad():
        test_preds_init = model_init(torch.FloatTensor(X_test), sample=True, n_samples=10)
        test_preds_init = torch.softmax(test_preds_init.mean(0), dim=1).numpy()

    initial_c_index = compute_c_index(test_preds_init, time_test, event_test)

    # Get ensemble predictions
    train_preds = get_ensemble_predictions(model_init, X_train, n_samples=config.ensemble_size)

    results = {}

    # Test each strategy
    strategies = {
        'random': lambda: random_score(len(time_train), can_query),
        'entropy': lambda: entropy_score(train_preds, artificial_time, can_query),
        'variance': lambda: variance_score(train_preds, artificial_time, can_query),
        'batchbald': lambda: batchbald_score(train_preds, artificial_time, can_query, config.probe_depth),
        'cluster_oracle': lambda: cluster_diversity_score(train_preds, cluster_train, can_query),
    }

    for strategy_name, score_func in strategies.items():
        scores = score_func()

        # Select batch
        if strategy_name == 'batchbald_batch':
            # Use true batch selection
            selected = batchbald_batch_score(train_preds, artificial_time, can_query,
                                              config.probe_depth, config.batch_size)
        else:
            valid_mask = scores > -np.inf
            n_valid = valid_mask.sum()
            batch_size_use = min(config.batch_size, n_valid)

            if batch_size_use == 0:
                results[strategy_name] = {
                    'improvement': 0.0,
                    'n_revealed': 0,
                    'cluster_diversity': 0.0
                }
                continue

            selected = np.argsort(scores)[-batch_size_use:]

        # Measure cluster diversity of selection
        selected_clusters = cluster_train[selected]
        cluster_diversity = len(np.unique(selected_clusters)) / config.n_clusters

        # Query oracle
        oracle = SimpleOracle(true_time, true_event, config.probe_depth)
        updated_time, updated_event = oracle.query(selected, artificial_time, artificial_event)

        n_revealed = ((updated_event - artificial_event) > 0).sum()

        # Retrain
        y_train_updated = encode_survival(updated_time, updated_event, config.n_time_bins)

        model_updated = BayesMtlr(in_features=config.n_features, num_time_bins=config.n_time_bins, config=args)
        model_updated = train_model(model_updated, X_train, y_train_updated, config, epochs=config.train_epochs, seed=run_seed+1000)

        # Evaluate
        model_updated.eval()
        with torch.no_grad():
            test_preds_updated = model_updated(torch.FloatTensor(X_test), sample=True, n_samples=10)
            test_preds_updated = torch.softmax(test_preds_updated.mean(0), dim=1).numpy()

        final_c_index = compute_c_index(test_preds_updated, time_test, event_test)

        results[strategy_name] = {
            'improvement': final_c_index - initial_c_index,
            'n_revealed': n_revealed,
            'cluster_diversity': cluster_diversity,
            'initial_c_index': initial_c_index,
            'final_c_index': final_c_index
        }

    return results


def main():
    print("\n" + "="*100)
    print("CONTROLLED EXPERIMENT: Creating Conditions for BatchBALD Advantage")
    print("="*100)
    print(f"Started: {datetime.now()}")

    config = ExperimentConfig()

    print(f"\nConfiguration:")
    print(f"  Samples: {config.n_samples} (clustered into {config.n_clusters} groups)")
    print(f"  Features: {config.n_features}")
    print(f"  Time bins: {config.n_time_bins}")
    print(f"  Probe depth: {config.probe_depth}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Training epochs: {config.train_epochs}")
    print(f"  Ensemble size: {config.ensemble_size}")
    print(f"  Runs: {config.n_runs}")

    all_results = {s: [] for s in ['random', 'entropy', 'variance', 'batchbald', 'cluster_oracle']}

    print(f"\nRunning {config.n_runs} experiments...")

    for run in range(config.n_runs):
        print(f"  Run {run+1}/{config.n_runs}...", end=" ", flush=True)
        try:
            results = run_experiment(config, run_seed=42 + run)
            for s, r in results.items():
                all_results[s].append(r)
            print(f"OK (entropy: {results['entropy']['improvement']:+.4f}, batchbald: {results['batchbald']['improvement']:+.4f})")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Analyze results
    print("\n" + "="*100)
    print("RESULTS")
    print("="*100)

    print(f"\n{'Strategy':<15} {'Mean Δ':<12} {'Std':<10} {'Wins':<10} {'Cluster Div':<12} {'SNR':<10}")
    print("-" * 75)

    for strategy in ['random', 'entropy', 'variance', 'batchbald', 'cluster_oracle']:
        if len(all_results[strategy]) == 0:
            continue

        improvements = [r['improvement'] for r in all_results[strategy]]
        cluster_divs = [r['cluster_diversity'] for r in all_results[strategy]]

        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        wins = sum(1 for imp in improvements if imp > 0)
        mean_div = np.mean(cluster_divs)
        snr = abs(mean_imp) / (std_imp + 1e-10)

        print(f"{strategy:<15} {mean_imp:+.4f}      {std_imp:.4f}    {wins}/{len(improvements):<6} {mean_div:.2f}         {snr:.3f}")

    # Statistical tests
    print("\nStatistical Tests (paired t-test):")

    strategies = ['entropy', 'variance', 'batchbald', 'cluster_oracle']
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i+1:]:
            if len(all_results[s1]) > 0 and len(all_results[s2]) > 0:
                imp1 = [r['improvement'] for r in all_results[s1]]
                imp2 = [r['improvement'] for r in all_results[s2]]
                t_stat, p_val = ttest_rel(imp1, imp2)
                sig = "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                print(f"  {s1} vs {s2}: t={t_stat:.3f}, p={p_val:.4f} {sig}")

    # Cluster diversity analysis
    print("\n" + "="*100)
    print("CLUSTER DIVERSITY ANALYSIS")
    print("="*100)
    print("\nCorrelation between cluster diversity and improvement:")

    for strategy in ['entropy', 'variance', 'batchbald']:
        if len(all_results[strategy]) == 0:
            continue

        divs = [r['cluster_diversity'] for r in all_results[strategy]]
        imps = [r['improvement'] for r in all_results[strategy]]

        if len(divs) > 2:
            corr = np.corrcoef(divs, imps)[0, 1]
            print(f"  {strategy}: r = {corr:.3f}")

    print("\n" + "="*100)
    print("CONCLUSIONS")
    print("="*100)
    print("""
Key observations:
1. If 'cluster_oracle' significantly outperforms others → diversity matters
2. If BatchBALD matches cluster_oracle → it's capturing diversity
3. If SNR > 2 → we have a meaningful signal
4. If BatchBALD beats entropy with p < 0.05 → we've created the right conditions
""")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()
