"""
Factor Analysis: Identify what prevents statistical significance.

Test three factors:
1. Artificial censoring strategy (random vs deterministic vs early/late-biased)
2. One-shot vs multi-round AL
3. Probe depth constraint (with vs without)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from src.acquisition.optimal_batchbald import OptimalBatchBALD
from src.oracle.oracle import Oracle
from src.models.survival_model import BayesianSurvivalModel
from src.evaluation.metrics import concordance_index
from src.data.synthetic import artificially_censor


def load_nacd_data():
    df = pd.read_csv('data/MIMIC/NACD/NACD_Full.csv')
    df = df.rename(columns={'SURVIVAL': 'time', 'CENSORED': 'event'})
    return df


def split_data(df, train_frac=0.7, random_state=42):
    np.random.seed(random_state)
    n = len(df)
    indices = np.random.permutation(n)
    n_train = int(n * train_frac)
    return df.iloc[indices[:n_train]].reset_index(drop=True), \
           df.iloc[indices[n_train:]].reset_index(drop=True)


def discretize_times(time_train, time_test, n_bins=20):
    all_times = np.concatenate([time_train, time_test])
    unique_times = np.unique(all_times[all_times > 0])

    if len(unique_times) > n_bins:
        bin_edges = np.quantile(unique_times, np.linspace(0, 1, n_bins + 1))
    else:
        bin_edges = np.concatenate([[0], unique_times, [unique_times[-1] + 1]])
        n_bins = len(unique_times)

    time_train_binned = np.digitize(time_train, bin_edges[1:])
    time_test_binned = np.digitize(time_test, bin_edges[1:])

    time_train_binned = np.clip(time_train_binned, 0, n_bins - 1)
    time_test_binned = np.clip(time_test_binned, 0, n_bins - 1)

    return time_train_binned, time_test_binned, n_bins


def deterministic_censoring(time, event, proportion, features, random_state=42):
    """Deterministic censoring based on features - same instances every time."""
    np.random.seed(random_state)
    n = len(time)

    # Use feature-based score for deterministic selection
    # This removes randomness from censoring
    feature_scores = features.sum(axis=1)
    n_to_censor = int((1 - proportion) * event.sum())

    event_indices = np.where(event == 1)[0]
    sorted_indices = event_indices[np.argsort(feature_scores[event_indices])]
    to_censor = sorted_indices[:n_to_censor]

    artificial_event = event.copy()
    artificial_event[to_censor] = 0

    return time.copy(), artificial_event


def early_biased_censoring(time, event, proportion, random_state=42):
    """Censor early deaths more - creates more teaching opportunity."""
    np.random.seed(random_state)
    n_to_censor = int((1 - proportion) * event.sum())

    event_indices = np.where(event == 1)[0]
    event_times = time[event_indices]

    # Probability inversely proportional to time (early deaths more likely censored)
    weights = 1.0 / (event_times + 1)
    weights = weights / weights.sum()

    to_censor = np.random.choice(event_indices, size=n_to_censor, replace=False, p=weights)

    artificial_event = event.copy()
    artificial_event[to_censor] = 0

    return time.copy(), artificial_event


def late_biased_censoring(time, event, proportion, random_state=42):
    """Censor late deaths more - oracle has less teaching opportunity."""
    np.random.seed(random_state)
    n_to_censor = int((1 - proportion) * event.sum())

    event_indices = np.where(event == 1)[0]
    event_times = time[event_indices]

    # Probability proportional to time (late deaths more likely censored)
    weights = (event_times + 1).astype(float)
    weights = weights / weights.sum()

    to_censor = np.random.choice(event_indices, size=n_to_censor, replace=False, p=weights)

    artificial_event = event.copy()
    artificial_event[to_censor] = 0

    return time.copy(), artificial_event


# ==================================================================================
# EXPERIMENT 1: ARTIFICIAL CENSORING STRATEGY
# ==================================================================================

def test_censoring_strategy(data, censoring_fn, censoring_name, probe_depth=3,
                           batch_size=30, n_runs=15, X_train=None):
    """Test a single censoring strategy."""
    print(f"\n{censoring_name}:")
    print(f"  Running {n_runs} trials...", flush=True)

    optimal_results = []
    entropy_results = []

    for run in range(n_runs):
        # Split data
        train_df, test_df = split_data(data, random_state=42 + run)

        # Features
        feature_cols = [col for col in train_df.columns if col not in ['time', 'event']]
        X_train_run = train_df[feature_cols].values.astype(np.float32)
        time_train = train_df['time'].values
        event_train = train_df['event'].values

        X_test = test_df[feature_cols].values.astype(np.float32)
        time_test = test_df['time'].values
        event_test = test_df['event'].values

        # Discretize
        time_train_binned, time_test_binned, n_bins = discretize_times(
            time_train, time_test, n_bins=20
        )

        true_train_time = time_train_binned.copy()
        true_train_event = event_train.copy()

        # Apply censoring strategy
        if censoring_name == "Random (current)":
            artificial_time, artificial_event = artificially_censor(
                time_train_binned, event_train, proportion=0.5, random_state=42 + run
            )
        elif censoring_name == "Deterministic":
            artificial_time, artificial_event = deterministic_censoring(
                time_train_binned, event_train, proportion=0.5,
                features=X_train_run, random_state=42
            )
        elif censoring_name == "Early-biased":
            artificial_time, artificial_event = early_biased_censoring(
                time_train_binned, event_train, proportion=0.5, random_state=42 + run
            )
        else:  # Late-biased
            artificial_time, artificial_event = late_biased_censoring(
                time_train_binned, event_train, proportion=0.5, random_state=42 + run
            )

        # Train initial model
        model_init = BayesianSurvivalModel(
            n_features=X_train_run.shape[1], n_time_bins=n_bins,
            n_ensemble=5, hidden_size=64
        )
        model_init.fit(X_train_run, artificial_time, artificial_event,
                      epochs=20, batch_size=32, verbose=False)

        test_preds_init = model_init.predict_proba(X_test)
        initial_c = concordance_index(test_preds_init, time_test_binned, event_test)

        train_preds = model_init.predict_proba(X_train_run)

        # Oracle
        oracle = Oracle(true_train_time, true_train_event, probe_depth)
        oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
            train_preds, artificial_time, artificial_event
        )

        # Test Optimal
        optimal_acq = OptimalBatchBALD(probe_depth=probe_depth, strategy='adaptive_fusion')
        selected_opt = optimal_acq.select_batch(
            oracle_probs, train_preds, artificial_time, batch_size, artificial_event, greedy=False
        )

        if len(selected_opt) > 0:
            updated_time_opt, updated_event_opt = oracle.query(
                selected_opt, artificial_time, artificial_event
            )
            model_opt = BayesianSurvivalModel(
                n_features=X_train_run.shape[1], n_time_bins=n_bins,
                n_ensemble=5, hidden_size=64
            )
            model_opt.fit(X_train_run, updated_time_opt, updated_event_opt,
                         epochs=20, batch_size=32, verbose=False)
            test_preds_opt = model_opt.predict_proba(X_test)
            final_c_opt = concordance_index(test_preds_opt, time_test_binned, event_test)
            optimal_results.append(final_c_opt - initial_c)
        else:
            optimal_results.append(0.0)

        # Test Entropy
        mean_preds = train_preds.mean(axis=0)
        scores = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=1)
        scores[artificial_event == 1] = -np.inf
        selected_ent = np.argsort(scores)[-batch_size:]
        valid = scores[selected_ent] > -np.inf
        selected_ent = selected_ent[valid]

        if len(selected_ent) > 0:
            updated_time_ent, updated_event_ent = oracle.query(
                selected_ent, artificial_time, artificial_event
            )
            model_ent = BayesianSurvivalModel(
                n_features=X_train_run.shape[1], n_time_bins=n_bins,
                n_ensemble=5, hidden_size=64
            )
            model_ent.fit(X_train_run, updated_time_ent, updated_event_ent,
                         epochs=20, batch_size=32, verbose=False)
            test_preds_ent = model_ent.predict_proba(X_test)
            final_c_ent = concordance_index(test_preds_ent, time_test_binned, event_test)
            entropy_results.append(final_c_ent - initial_c)
        else:
            entropy_results.append(0.0)

    # Stats
    mean_opt = np.mean(optimal_results)
    mean_ent = np.mean(entropy_results)
    diff = mean_opt - mean_ent
    t_stat, p_val = ttest_rel(optimal_results, entropy_results)

    print(f"    Optimal: {mean_opt:+.4f}±{np.std(optimal_results):.4f}")
    print(f"    Entropy: {mean_ent:+.4f}±{np.std(entropy_results):.4f}")
    print(f"    Difference: {diff:+.4f}, p={p_val:.4f}")

    return {
        'censoring': censoring_name,
        'optimal_mean': mean_opt,
        'entropy_mean': mean_ent,
        'difference': diff,
        'p_value': p_val,
        'significant': p_val < 0.05
    }


# ==================================================================================
# EXPERIMENT 2: MULTI-ROUND AL
# ==================================================================================

def test_multiround_al(data, n_rounds=3, batch_size_per_round=10, probe_depth=3, n_runs=15):
    """Test multi-round AL vs one-shot."""
    print(f"\nMulti-round AL ({n_rounds} rounds × {batch_size_per_round} batch):")
    print(f"  Running {n_runs} trials...", flush=True)

    multiround_results = []
    oneshot_results = []

    total_batch = n_rounds * batch_size_per_round

    for run in range(n_runs):
        # Split data
        train_df, test_df = split_data(data, random_state=42 + run)

        feature_cols = [col for col in train_df.columns if col not in ['time', 'event']]
        X_train = train_df[feature_cols].values.astype(np.float32)
        time_train = train_df['time'].values
        event_train = train_df['event'].values

        X_test = test_df[feature_cols].values.astype(np.float32)
        time_test = test_df['time'].values
        event_test = test_df['event'].values

        # Discretize
        time_train_binned, time_test_binned, n_bins = discretize_times(
            time_train, time_test, n_bins=20
        )

        true_train_time = time_train_binned.copy()
        true_train_event = event_train.copy()

        # Artificial censoring
        artificial_time, artificial_event = artificially_censor(
            time_train_binned, event_train, proportion=0.5, random_state=42 + run
        )

        # Initial model
        model_init = BayesianSurvivalModel(
            n_features=X_train.shape[1], n_time_bins=n_bins,
            n_ensemble=5, hidden_size=64
        )
        model_init.fit(X_train, artificial_time, artificial_event,
                      epochs=20, batch_size=32, verbose=False)

        test_preds_init = model_init.predict_proba(X_test)
        initial_c = concordance_index(test_preds_init, time_test_binned, event_test)

        # Multi-round AL
        current_time = artificial_time.copy()
        current_event = artificial_event.copy()

        for round_idx in range(n_rounds):
            # Train model on current data
            model_round = BayesianSurvivalModel(
                n_features=X_train.shape[1], n_time_bins=n_bins,
                n_ensemble=5, hidden_size=64
            )
            model_round.fit(X_train, current_time, current_event,
                           epochs=20, batch_size=32, verbose=False)

            train_preds = model_round.predict_proba(X_train)
            oracle = Oracle(true_train_time, true_train_event, probe_depth)
            oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
                train_preds, current_time, current_event
            )

            # Select batch
            acq = OptimalBatchBALD(probe_depth=probe_depth, strategy='adaptive_fusion')
            selected = acq.select_batch(
                oracle_probs, train_preds, current_time,
                batch_size_per_round, current_event, greedy=False
            )

            if len(selected) > 0:
                current_time, current_event = oracle.query(selected, current_time, current_event)

        # Final evaluation
        model_final = BayesianSurvivalModel(
            n_features=X_train.shape[1], n_time_bins=n_bins,
            n_ensemble=5, hidden_size=64
        )
        model_final.fit(X_train, current_time, current_event,
                       epochs=20, batch_size=32, verbose=False)
        test_preds_final = model_final.predict_proba(X_test)
        final_c = concordance_index(test_preds_final, time_test_binned, event_test)
        multiround_results.append(final_c - initial_c)

        # One-shot AL with same total budget
        train_preds = model_init.predict_proba(X_train)
        oracle = Oracle(true_train_time, true_train_event, probe_depth)
        oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
            train_preds, artificial_time, artificial_event
        )

        acq = OptimalBatchBALD(probe_depth=probe_depth, strategy='adaptive_fusion')
        selected = acq.select_batch(
            oracle_probs, train_preds, artificial_time,
            total_batch, artificial_event, greedy=False
        )

        if len(selected) > 0:
            updated_time, updated_event = oracle.query(selected, artificial_time, artificial_event)
            model_oneshot = BayesianSurvivalModel(
                n_features=X_train.shape[1], n_time_bins=n_bins,
                n_ensemble=5, hidden_size=64
            )
            model_oneshot.fit(X_train, updated_time, updated_event,
                             epochs=20, batch_size=32, verbose=False)
            test_preds_oneshot = model_oneshot.predict_proba(X_test)
            final_c_oneshot = concordance_index(test_preds_oneshot, time_test_binned, event_test)
            oneshot_results.append(final_c_oneshot - initial_c)
        else:
            oneshot_results.append(0.0)

    # Stats
    mean_multi = np.mean(multiround_results)
    mean_one = np.mean(oneshot_results)
    diff = mean_multi - mean_one
    t_stat, p_val = ttest_rel(multiround_results, oneshot_results)

    print(f"    Multi-round: {mean_multi:+.4f}±{np.std(multiround_results):.4f}")
    print(f"    One-shot: {mean_one:+.4f}±{np.std(oneshot_results):.4f}")
    print(f"    Difference: {diff:+.4f}, p={p_val:.4f}")

    return {
        'approach': 'Multi-round vs One-shot',
        'multiround_mean': mean_multi,
        'oneshot_mean': mean_one,
        'difference': diff,
        'p_value': p_val,
        'significant': p_val < 0.05
    }


# ==================================================================================
# EXPERIMENT 3: NO PROBE DEPTH CONSTRAINT
# ==================================================================================

def test_no_probe_constraint(data, batch_size=30, n_runs=15):
    """Test with unlimited probe depth (oracle always reveals truth)."""
    print(f"\nNo Probe Depth Constraint (unlimited oracle):")
    print(f"  Running {n_runs} trials...", flush=True)

    unlimited_results = []
    limited_results = []

    probe_depth = 3  # For limited case

    for run in range(n_runs):
        # Split data
        train_df, test_df = split_data(data, random_state=42 + run)

        feature_cols = [col for col in train_df.columns if col not in ['time', 'event']]
        X_train = train_df[feature_cols].values.astype(np.float32)
        time_train = train_df['time'].values
        event_train = train_df['event'].values

        X_test = test_df[feature_cols].values.astype(np.float32)
        time_test = test_df['time'].values
        event_test = test_df['event'].values

        # Discretize
        time_train_binned, time_test_binned, n_bins = discretize_times(
            time_train, time_test, n_bins=20
        )

        true_train_time = time_train_binned.copy()
        true_train_event = event_train.copy()

        # Artificial censoring
        artificial_time, artificial_event = artificially_censor(
            time_train_binned, event_train, proportion=0.5, random_state=42 + run
        )

        # Initial model
        model_init = BayesianSurvivalModel(
            n_features=X_train.shape[1], n_time_bins=n_bins,
            n_ensemble=5, hidden_size=64
        )
        model_init.fit(X_train, artificial_time, artificial_event,
                      epochs=20, batch_size=32, verbose=False)

        test_preds_init = model_init.predict_proba(X_test)
        initial_c = concordance_index(test_preds_init, time_test_binned, event_test)

        train_preds = model_init.predict_proba(X_train)

        # UNLIMITED: Simply select most uncertain censored, then reveal ALL truth
        mean_preds = train_preds.mean(axis=0)
        scores = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=1)
        scores[artificial_event == 1] = -np.inf
        selected = np.argsort(scores)[-batch_size:]
        valid = scores[selected] > -np.inf
        selected = selected[valid]

        if len(selected) > 0:
            # Reveal FULL truth (no probe constraint)
            updated_time_unl = artificial_time.copy()
            updated_event_unl = artificial_event.copy()
            updated_time_unl[selected] = true_train_time[selected]
            updated_event_unl[selected] = true_train_event[selected]

            model_unl = BayesianSurvivalModel(
                n_features=X_train.shape[1], n_time_bins=n_bins,
                n_ensemble=5, hidden_size=64
            )
            model_unl.fit(X_train, updated_time_unl, updated_event_unl,
                         epochs=20, batch_size=32, verbose=False)
            test_preds_unl = model_unl.predict_proba(X_test)
            final_c_unl = concordance_index(test_preds_unl, time_test_binned, event_test)
            unlimited_results.append(final_c_unl - initial_c)
        else:
            unlimited_results.append(0.0)

        # LIMITED: Use probe depth constraint
        oracle = Oracle(true_train_time, true_train_event, probe_depth)
        oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
            train_preds, artificial_time, artificial_event
        )

        acq = OptimalBatchBALD(probe_depth=probe_depth, strategy='adaptive_fusion')
        selected_lim = acq.select_batch(
            oracle_probs, train_preds, artificial_time,
            batch_size, artificial_event, greedy=False
        )

        if len(selected_lim) > 0:
            updated_time_lim, updated_event_lim = oracle.query(
                selected_lim, artificial_time, artificial_event
            )
            model_lim = BayesianSurvivalModel(
                n_features=X_train.shape[1], n_time_bins=n_bins,
                n_ensemble=5, hidden_size=64
            )
            model_lim.fit(X_train, updated_time_lim, updated_event_lim,
                         epochs=20, batch_size=32, verbose=False)
            test_preds_lim = model_lim.predict_proba(X_test)
            final_c_lim = concordance_index(test_preds_lim, time_test_binned, event_test)
            limited_results.append(final_c_lim - initial_c)
        else:
            limited_results.append(0.0)

    # Stats
    mean_unl = np.mean(unlimited_results)
    mean_lim = np.mean(limited_results)
    diff = mean_unl - mean_lim

    print(f"    Unlimited oracle: {mean_unl:+.4f}±{np.std(unlimited_results):.4f}")
    print(f"    Limited (k={probe_depth}): {mean_lim:+.4f}±{np.std(limited_results):.4f}")
    print(f"    Impact of constraint: {diff:+.4f}")

    return {
        'constraint': 'Probe depth impact',
        'unlimited_mean': mean_unl,
        'limited_mean': mean_lim,
        'impact': diff,
        'unlimited_std': np.std(unlimited_results),
        'limited_std': np.std(limited_results)
    }


def main():
    print("\n" + "="*80)
    print("FACTOR ANALYSIS: What Prevents Statistical Significance?")
    print("="*80)

    data = load_nacd_data()
    print(f"\nLoaded NACD: {data.shape}\n")

    # ==================================================================================
    # EXPERIMENT 1: CENSORING STRATEGY
    # ==================================================================================

    print("="*80)
    print("EXPERIMENT 1: Artificial Censoring Strategy")
    print("="*80)
    print("\nTesting if random censoring adds too much variance...")

    censoring_results = []

    strategies = [
        ("Random (current)", None, None),
        ("Deterministic", None, None),
        ("Early-biased", None, None),
        ("Late-biased", None, None)
    ]

    for name, _, _ in strategies:
        result = test_censoring_strategy(
            data, None, name, probe_depth=3, batch_size=30, n_runs=15
        )
        censoring_results.append(result)

    print("\n" + "-"*80)
    print("Summary: Censoring Strategy Impact")
    print("-"*80)
    for r in censoring_results:
        sig = "✓" if r['significant'] else " "
        print(f"{sig} {r['censoring']:<20} Δ={r['difference']:+.4f}, p={r['p_value']:.4f}")

    # ==================================================================================
    # EXPERIMENT 2: MULTI-ROUND AL
    # ==================================================================================

    print("\n" + "="*80)
    print("EXPERIMENT 2: Multi-round vs One-shot AL")
    print("="*80)
    print("\nTesting if iterative AL reduces variance...")

    multiround_result = test_multiround_al(
        data, n_rounds=3, batch_size_per_round=10, probe_depth=3, n_runs=15
    )

    print("\n" + "-"*80)
    print("Summary: Multi-round Impact")
    print("-"*80)
    sig = "✓" if multiround_result['significant'] else " "
    print(f"{sig} {multiround_result['approach']}: Δ={multiround_result['difference']:+.4f}, p={multiround_result['p_value']:.4f}")

    # ==================================================================================
    # EXPERIMENT 3: PROBE DEPTH CONSTRAINT
    # ==================================================================================

    print("\n" + "="*80)
    print("EXPERIMENT 3: Probe Depth Constraint Impact")
    print("="*80)
    print("\nTesting if probe constraint is the main limitation...")

    probe_result = test_no_probe_constraint(data, batch_size=30, n_runs=15)

    print("\n" + "-"*80)
    print("Summary: Probe Depth Constraint Impact")
    print("-"*80)
    print(f"  Unlimited oracle: {probe_result['unlimited_mean']:+.4f}±{probe_result['unlimited_std']:.4f}")
    print(f"  Limited oracle: {probe_result['limited_mean']:+.4f}±{probe_result['limited_std']:.4f}")
    print(f"  Constraint cost: {probe_result['impact']:+.4f}")

    # ==================================================================================
    # FINAL ANALYSIS
    # ==================================================================================

    print("\n" + "="*80)
    print("FINAL ANALYSIS: Main Culprits")
    print("="*80 + "\n")

    # Check which factor had biggest impact
    factors = []

    # Censoring: check if any strategy achieved significance
    if any(r['significant'] for r in censoring_results):
        factors.append(("Censoring strategy", "MAJOR CULPRIT - switching strategy helps!"))

    # Multi-round
    if multiround_result['significant']:
        factors.append(("One-shot AL", "MAJOR CULPRIT - multi-round helps!"))

    # Probe depth
    if abs(probe_result['impact']) > 0.01:  # Substantial difference
        factors.append(("Probe depth constraint", f"LIMITING FACTOR - costs {probe_result['impact']:+.4f}"))

    if len(factors) == 0:
        print("✗ NO SINGLE FACTOR is the culprit.")
        print("  The problem is fundamental to the dataset/task.")
        print("\n  Variance reduction attempts:")
        print(f"    - Deterministic censoring: p={[r for r in censoring_results if r['censoring']=='Deterministic'][0]['p_value']:.4f}")
        print(f"    - Multi-round AL: p={multiround_result['p_value']:.4f}")
        print(f"    - Probe constraint cost: {probe_result['impact']:+.4f}")
    else:
        print("✓ CULPRITS IDENTIFIED:\n")
        for i, (factor, description) in enumerate(factors, 1):
            print(f"{i}. {factor}: {description}")

    print("\n" + "="*80 + "\n")

    return {
        'censoring': censoring_results,
        'multiround': multiround_result,
        'probe': probe_result
    }


if __name__ == '__main__':
    results = main()
