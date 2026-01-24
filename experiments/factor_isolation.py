"""
Factor Isolation: Test each factor independently to find the main culprit.

Factors to test:
1. Random artificial censoring (vs deterministic)
2. One-shot AL (vs multi-round)
3. Probe depth constraint (vs unlimited oracle)
4. Censoring proportion (0.5 vs 0.7 - more aggressive)
5. Model training epochs (20 vs 50 - better trained models)
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


# ==================================================================================
# FACTOR 1: CENSORING RANDOMNESS
# ==================================================================================

def test_factor_censoring(data, n_runs=12):
    """Test: Random vs Deterministic censoring."""
    print("\n" + "="*80)
    print("FACTOR 1: Random vs Deterministic Censoring")
    print("="*80)
    print("\nHypothesis: Random censoring adds unnecessary variance.\n")

    random_opt = []
    random_ent = []
    determ_opt = []
    determ_ent = []

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", flush=True)

        train_df, test_df = split_data(data, random_state=42 + run)
        feature_cols = [col for col in train_df.columns if col not in ['time', 'event']]
        X_train = train_df[feature_cols].values.astype(np.float32)
        time_train = train_df['time'].values
        event_train = train_df['event'].values
        X_test = test_df[feature_cols].values.astype(np.float32)
        time_test = test_df['time'].values
        event_test = test_df['event'].values

        time_train_binned, time_test_binned, n_bins = discretize_times(
            time_train, time_test, n_bins=20
        )

        # RANDOM censoring
        art_time_rand, art_event_rand = artificially_censor(
            time_train_binned, event_train, proportion=0.5, random_state=42 + run
        )

        # DETERMINISTIC censoring (same instances every run)
        np.random.seed(42)  # Fixed seed
        n_to_censor = int(0.5 * event_train.sum())
        event_indices = np.where(event_train == 1)[0]
        to_censor = np.random.choice(event_indices, size=n_to_censor, replace=False)
        art_time_det = time_train_binned.copy()
        art_event_det = event_train.copy()
        art_event_det[to_censor] = 0

        # Test both
        for censoring_type, art_time, art_event in [
            ('random', art_time_rand, art_event_rand),
            ('deterministic', art_time_det, art_event_det)
        ]:
            model = BayesianSurvivalModel(
                n_features=X_train.shape[1], n_time_bins=n_bins,
                n_ensemble=5, hidden_size=64
            )
            model.fit(X_train, art_time, art_event, epochs=20, batch_size=32, verbose=False)

            test_preds_init = model.predict_proba(X_test)
            initial_c = concordance_index(test_preds_init, time_test_binned, event_test)
            train_preds = model.predict_proba(X_train)

            oracle = Oracle(time_train_binned, event_train, probe_depth=3)
            oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
                train_preds, art_time, art_event
            )

            # Optimal
            acq = OptimalBatchBALD(probe_depth=3, strategy='adaptive_fusion')
            selected = acq.select_batch(oracle_probs, train_preds, art_time, 30, art_event, greedy=False)
            if len(selected) > 0:
                upd_time, upd_event = oracle.query(selected, art_time, art_event)
                model_opt = BayesianSurvivalModel(
                    n_features=X_train.shape[1], n_time_bins=n_bins,
                    n_ensemble=5, hidden_size=64
                )
                model_opt.fit(X_train, upd_time, upd_event, epochs=20, batch_size=32, verbose=False)
                test_preds = model_opt.predict_proba(X_test)
                final_c = concordance_index(test_preds, time_test_binned, event_test)
                if censoring_type == 'random':
                    random_opt.append(final_c - initial_c)
                else:
                    determ_opt.append(final_c - initial_c)

            # Entropy
            mean_preds = train_preds.mean(axis=0)
            scores = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=1)
            scores[art_event == 1] = -np.inf
            selected_ent = np.argsort(scores)[-30:]
            valid = scores[selected_ent] > -np.inf
            selected_ent = selected_ent[valid]
            if len(selected_ent) > 0:
                upd_time, upd_event = oracle.query(selected_ent, art_time, art_event)
                model_ent = BayesianSurvivalModel(
                    n_features=X_train.shape[1], n_time_bins=n_bins,
                    n_ensemble=5, hidden_size=64
                )
                model_ent.fit(X_train, upd_time, upd_event, epochs=20, batch_size=32, verbose=False)
                test_preds = model_ent.predict_proba(X_test)
                final_c = concordance_index(test_preds, time_test_binned, event_test)
                if censoring_type == 'random':
                    random_ent.append(final_c - initial_c)
                else:
                    determ_ent.append(final_c - initial_c)

    # Analyze
    print("\nRandom Censoring:")
    print(f"  Optimal: {np.mean(random_opt):+.4f}±{np.std(random_opt):.4f}")
    print(f"  Entropy: {np.mean(random_ent):+.4f}±{np.std(random_ent):.4f}")
    t, p_rand = ttest_rel(random_opt, random_ent)
    print(f"  p-value: {p_rand:.4f}")

    print("\nDeterministic Censoring:")
    print(f"  Optimal: {np.mean(determ_opt):+.4f}±{np.std(determ_opt):.4f}")
    print(f"  Entropy: {np.mean(determ_ent):+.4f}±{np.std(determ_ent):.4f}")
    t, p_det = ttest_rel(determ_opt, determ_ent)
    print(f"  p-value: {p_det:.4f}")

    print(f"\nImpact: Deterministic reduces p-value by {p_rand - p_det:+.4f}")

    return {
        'factor': 'Censoring randomness',
        'p_random': p_rand,
        'p_deterministic': p_det,
        'improvement': p_rand - p_det > 0,
        'significant_with_fix': p_det < 0.05
    }


# ==================================================================================
# FACTOR 2: ONE-SHOT vs MULTI-ROUND
# ==================================================================================

def test_factor_multiround(data, n_runs=12):
    """Test: One-shot vs Multi-round AL."""
    print("\n" + "="*80)
    print("FACTOR 2: One-shot vs Multi-round AL")
    print("="*80)
    print("\nHypothesis: Multi-round reduces variance via adaptive selection.\n")

    oneshot_diffs = []
    multiround_diffs = []

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", flush=True)

        train_df, test_df = split_data(data, random_state=42 + run)
        feature_cols = [col for col in train_df.columns if col not in ['time', 'event']]
        X_train = train_df[feature_cols].values.astype(np.float32)
        time_train = train_df['time'].values
        event_train = train_df['event'].values
        X_test = test_df[feature_cols].values.astype(np.float32)
        time_test = test_df['time'].values
        event_test = test_df['event'].values

        time_train_binned, time_test_binned, n_bins = discretize_times(
            time_train, time_test, n_bins=20
        )

        art_time, art_event = artificially_censor(
            time_train_binned, event_train, proportion=0.5, random_state=42 + run
        )

        # Initial model
        model_init = BayesianSurvivalModel(
            n_features=X_train.shape[1], n_time_bins=n_bins,
            n_ensemble=5, hidden_size=64
        )
        model_init.fit(X_train, art_time, art_event, epochs=20, batch_size=32, verbose=False)
        test_preds_init = model_init.predict_proba(X_test)
        initial_c = concordance_index(test_preds_init, time_test_binned, event_test)

        # ONE-SHOT: 30 instances at once
        train_preds = model_init.predict_proba(X_train)
        oracle = Oracle(time_train_binned, event_train, probe_depth=3)
        oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
            train_preds, art_time, art_event
        )
        acq = OptimalBatchBALD(probe_depth=3, strategy='adaptive_fusion')
        selected = acq.select_batch(oracle_probs, train_preds, art_time, 30, art_event, greedy=False)
        if len(selected) > 0:
            upd_time, upd_event = oracle.query(selected, art_time, art_event)
            model_one = BayesianSurvivalModel(
                n_features=X_train.shape[1], n_time_bins=n_bins,
                n_ensemble=5, hidden_size=64
            )
            model_one.fit(X_train, upd_time, upd_event, epochs=20, batch_size=32, verbose=False)
            test_preds_one = model_one.predict_proba(X_test)
            final_c_one = concordance_index(test_preds_one, time_test_binned, event_test)
            oneshot_diffs.append(final_c_one - initial_c)
        else:
            oneshot_diffs.append(0.0)

        # MULTI-ROUND: 3 rounds × 10 instances
        current_time = art_time.copy()
        current_event = art_event.copy()
        for round_idx in range(3):
            model_round = BayesianSurvivalModel(
                n_features=X_train.shape[1], n_time_bins=n_bins,
                n_ensemble=5, hidden_size=64
            )
            model_round.fit(X_train, current_time, current_event, epochs=20, batch_size=32, verbose=False)
            train_preds = model_round.predict_proba(X_train)
            oracle = Oracle(time_train_binned, event_train, probe_depth=3)
            oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
                train_preds, current_time, current_event
            )
            acq = OptimalBatchBALD(probe_depth=3, strategy='adaptive_fusion')
            selected = acq.select_batch(oracle_probs, train_preds, current_time, 10, current_event, greedy=False)
            if len(selected) > 0:
                current_time, current_event = oracle.query(selected, current_time, current_event)

        model_multi = BayesianSurvivalModel(
            n_features=X_train.shape[1], n_time_bins=n_bins,
            n_ensemble=5, hidden_size=64
        )
        model_multi.fit(X_train, current_time, current_event, epochs=20, batch_size=32, verbose=False)
        test_preds_multi = model_multi.predict_proba(X_test)
        final_c_multi = concordance_index(test_preds_multi, time_test_binned, event_test)
        multiround_diffs.append(final_c_multi - initial_c)

    print("\nOne-shot (30 at once):")
    print(f"  Mean: {np.mean(oneshot_diffs):+.4f}±{np.std(oneshot_diffs):.4f}")

    print("\nMulti-round (3×10):")
    print(f"  Mean: {np.mean(multiround_diffs):+.4f}±{np.std(multiround_diffs):.4f}")

    t, p_multi = ttest_rel(multiround_diffs, oneshot_diffs)
    print(f"\np-value (multi vs one): {p_multi:.4f}")
    print(f"Multi-round better: {np.mean(multiround_diffs) > np.mean(oneshot_diffs)}")

    return {
        'factor': 'One-shot AL',
        'p_multiround': p_multi,
        'multiround_better': np.mean(multiround_diffs) > np.mean(oneshot_diffs),
        'variance_reduction': np.std(oneshot_diffs) - np.std(multiround_diffs),
        'significant_with_fix': p_multi < 0.05
    }


# ==================================================================================
# FACTOR 3: PROBE DEPTH CONSTRAINT
# ==================================================================================

def test_factor_probe_constraint(data, n_runs=12):
    """Test: Limited probe depth vs unlimited oracle."""
    print("\n" + "="*80)
    print("FACTOR 3: Probe Depth Constraint")
    print("="*80)
    print("\nHypothesis: Probe constraint severely limits oracle teaching.\n")

    limited_imps = []
    unlimited_imps = []

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", flush=True)

        train_df, test_df = split_data(data, random_state=42 + run)
        feature_cols = [col for col in train_df.columns if col not in ['time', 'event']]
        X_train = train_df[feature_cols].values.astype(np.float32)
        time_train = train_df['time'].values
        event_train = train_df['event'].values
        X_test = test_df[feature_cols].values.astype(np.float32)
        time_test = test_df['time'].values
        event_test = test_df['event'].values

        time_train_binned, time_test_binned, n_bins = discretize_times(
            time_train, time_test, n_bins=20
        )

        art_time, art_event = artificially_censor(
            time_train_binned, event_train, proportion=0.5, random_state=42 + run
        )

        model_init = BayesianSurvivalModel(
            n_features=X_train.shape[1], n_time_bins=n_bins,
            n_ensemble=5, hidden_size=64
        )
        model_init.fit(X_train, art_time, art_event, epochs=20, batch_size=32, verbose=False)
        test_preds_init = model_init.predict_proba(X_test)
        initial_c = concordance_index(test_preds_init, time_test_binned, event_test)

        train_preds = model_init.predict_proba(X_train)

        # LIMITED (k=3)
        oracle_lim = Oracle(time_train_binned, event_train, probe_depth=3)
        oracle_probs = oracle_lim.get_oracle_outcome_probs_ensemble(
            train_preds, art_time, art_event
        )
        acq = OptimalBatchBALD(probe_depth=3, strategy='adaptive_fusion')
        selected = acq.select_batch(oracle_probs, train_preds, art_time, 30, art_event, greedy=False)
        if len(selected) > 0:
            upd_time, upd_event = oracle_lim.query(selected, art_time, art_event)
            model_lim = BayesianSurvivalModel(
                n_features=X_train.shape[1], n_time_bins=n_bins,
                n_ensemble=5, hidden_size=64
            )
            model_lim.fit(X_train, upd_time, upd_event, epochs=20, batch_size=32, verbose=False)
            test_preds_lim = model_lim.predict_proba(X_test)
            final_c_lim = concordance_index(test_preds_lim, time_test_binned, event_test)
            limited_imps.append(final_c_lim - initial_c)
        else:
            limited_imps.append(0.0)

        # UNLIMITED (reveal full truth)
        mean_preds = train_preds.mean(axis=0)
        scores = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=1)
        scores[art_event == 1] = -np.inf
        selected_unl = np.argsort(scores)[-30:]
        valid = scores[selected_unl] > -np.inf
        selected_unl = selected_unl[valid]
        if len(selected_unl) > 0:
            upd_time_unl = art_time.copy()
            upd_event_unl = art_event.copy()
            upd_time_unl[selected_unl] = time_train_binned[selected_unl]
            upd_event_unl[selected_unl] = event_train[selected_unl]
            model_unl = BayesianSurvivalModel(
                n_features=X_train.shape[1], n_time_bins=n_bins,
                n_ensemble=5, hidden_size=64
            )
            model_unl.fit(X_train, upd_time_unl, upd_event_unl, epochs=20, batch_size=32, verbose=False)
            test_preds_unl = model_unl.predict_proba(X_test)
            final_c_unl = concordance_index(test_preds_unl, time_test_binned, event_test)
            unlimited_imps.append(final_c_unl - initial_c)
        else:
            unlimited_imps.append(0.0)

    print("\nLimited (k=3):")
    print(f"  Mean: {np.mean(limited_imps):+.4f}±{np.std(limited_imps):.4f}")

    print("\nUnlimited (full truth):")
    print(f"  Mean: {np.mean(unlimited_imps):+.4f}±{np.std(unlimited_imps):.4f}")

    constraint_cost = np.mean(unlimited_imps) - np.mean(limited_imps)
    print(f"\nConstraint cost: {constraint_cost:.4f}")
    print(f"Variance reduction: {np.std(limited_imps) - np.std(unlimited_imps):+.4f}")

    return {
        'factor': 'Probe depth constraint',
        'constraint_cost': constraint_cost,
        'limited_mean': np.mean(limited_imps),
        'unlimited_mean': np.mean(unlimited_imps),
        'limited_std': np.std(limited_imps),
        'unlimited_std': np.std(unlimited_imps)
    }


# ==================================================================================
# FACTOR 4: CENSORING PROPORTION
# ==================================================================================

def test_factor_censoring_proportion(data, n_runs=12):
    """Test: 50% vs 70% censoring."""
    print("\n" + "="*80)
    print("FACTOR 4: Censoring Proportion")
    print("="*80)
    print("\nHypothesis: More censoring = more teaching opportunities.\n")

    prop50_diffs = []
    prop70_diffs = []

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", flush=True)

        train_df, test_df = split_data(data, random_state=42 + run)
        feature_cols = [col for col in train_df.columns if col not in ['time', 'event']]
        X_train = train_df[feature_cols].values.astype(np.float32)
        time_train = train_df['time'].values
        event_train = train_df['event'].values
        X_test = test_df[feature_cols].values.astype(np.float32)
        time_test = test_df['time'].values
        event_test = test_df['event'].values

        time_train_binned, time_test_binned, n_bins = discretize_times(
            time_train, time_test, n_bins=20
        )

        for prop, results_list in [(0.5, prop50_diffs), (0.7, prop70_diffs)]:
            art_time, art_event = artificially_censor(
                time_train_binned, event_train, proportion=prop, random_state=42 + run
            )

            model_init = BayesianSurvivalModel(
                n_features=X_train.shape[1], n_time_bins=n_bins,
                n_ensemble=5, hidden_size=64
            )
            model_init.fit(X_train, art_time, art_event, epochs=20, batch_size=32, verbose=False)
            test_preds_init = model_init.predict_proba(X_test)
            initial_c = concordance_index(test_preds_init, time_test_binned, event_test)

            train_preds = model_init.predict_proba(X_train)
            oracle = Oracle(time_train_binned, event_train, probe_depth=3)
            oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
                train_preds, art_time, art_event
            )
            acq = OptimalBatchBALD(probe_depth=3, strategy='adaptive_fusion')
            selected = acq.select_batch(oracle_probs, train_preds, art_time, 30, art_event, greedy=False)
            if len(selected) > 0:
                upd_time, upd_event = oracle.query(selected, art_time, art_event)
                model_upd = BayesianSurvivalModel(
                    n_features=X_train.shape[1], n_time_bins=n_bins,
                    n_ensemble=5, hidden_size=64
                )
                model_upd.fit(X_train, upd_time, upd_event, epochs=20, batch_size=32, verbose=False)
                test_preds_upd = model_upd.predict_proba(X_test)
                final_c = concordance_index(test_preds_upd, time_test_binned, event_test)
                results_list.append(final_c - initial_c)
            else:
                results_list.append(0.0)

    print("\n50% censored:")
    print(f"  Mean: {np.mean(prop50_diffs):+.4f}±{np.std(prop50_diffs):.4f}")

    print("\n70% censored:")
    print(f"  Mean: {np.mean(prop70_diffs):+.4f}±{np.std(prop70_diffs):.4f}")

    print(f"\nDifference: {np.mean(prop70_diffs) - np.mean(prop50_diffs):+.4f}")

    return {
        'factor': 'Censoring proportion',
        'prop50_mean': np.mean(prop50_diffs),
        'prop70_mean': np.mean(prop70_diffs),
        'prop70_better': np.mean(prop70_diffs) > np.mean(prop50_diffs)
    }


# ==================================================================================
# FACTOR 5: MODEL TRAINING QUALITY
# ==================================================================================

def test_factor_model_training(data, n_runs=12):
    """Test: 20 epochs vs 50 epochs."""
    print("\n" + "="*80)
    print("FACTOR 5: Model Training Quality")
    print("="*80)
    print("\nHypothesis: Better trained models = less noisy predictions.\n")

    epoch20_diffs = []
    epoch50_diffs = []

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", flush=True)

        train_df, test_df = split_data(data, random_state=42 + run)
        feature_cols = [col for col in train_df.columns if col not in ['time', 'event']]
        X_train = train_df[feature_cols].values.astype(np.float32)
        time_train = train_df['time'].values
        event_train = train_df['event'].values
        X_test = test_df[feature_cols].values.astype(np.float32)
        time_test = test_df['time'].values
        event_test = test_df['event'].values

        time_train_binned, time_test_binned, n_bins = discretize_times(
            time_train, time_test, n_bins=20
        )

        art_time, art_event = artificially_censor(
            time_train_binned, event_train, proportion=0.5, random_state=42 + run
        )

        for epochs, results_list in [(20, epoch20_diffs), (50, epoch50_diffs)]:
            model_init = BayesianSurvivalModel(
                n_features=X_train.shape[1], n_time_bins=n_bins,
                n_ensemble=5, hidden_size=64
            )
            model_init.fit(X_train, art_time, art_event, epochs=epochs, batch_size=32, verbose=False)
            test_preds_init = model_init.predict_proba(X_test)
            initial_c = concordance_index(test_preds_init, time_test_binned, event_test)

            train_preds = model_init.predict_proba(X_train)
            oracle = Oracle(time_train_binned, event_train, probe_depth=3)
            oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
                train_preds, art_time, art_event
            )
            acq = OptimalBatchBALD(probe_depth=3, strategy='adaptive_fusion')
            selected = acq.select_batch(oracle_probs, train_preds, art_time, 30, art_event, greedy=False)
            if len(selected) > 0:
                upd_time, upd_event = oracle.query(selected, art_time, art_event)
                model_upd = BayesianSurvivalModel(
                    n_features=X_train.shape[1], n_time_bins=n_bins,
                    n_ensemble=5, hidden_size=64
                )
                model_upd.fit(X_train, upd_time, upd_event, epochs=epochs, batch_size=32, verbose=False)
                test_preds_upd = model_upd.predict_proba(X_test)
                final_c = concordance_index(test_preds_upd, time_test_binned, event_test)
                results_list.append(final_c - initial_c)
            else:
                results_list.append(0.0)

    print("\n20 epochs:")
    print(f"  Mean: {np.mean(epoch20_diffs):+.4f}±{np.std(epoch20_diffs):.4f}")

    print("\n50 epochs:")
    print(f"  Mean: {np.mean(epoch50_diffs):+.4f}±{np.std(epoch50_diffs):.4f}")

    print(f"\nVariance reduction: {np.std(epoch20_diffs) - np.std(epoch50_diffs):+.4f}")

    return {
        'factor': 'Model training quality',
        'epoch20_std': np.std(epoch20_diffs),
        'epoch50_std': np.std(epoch50_diffs),
        'variance_reduction': np.std(epoch20_diffs) - np.std(epoch50_diffs)
    }


def main():
    print("\n" + "="*80)
    print("FACTOR ISOLATION: Find the Main Culprit")
    print("="*80)

    data = load_nacd_data()
    print(f"\nLoaded NACD: {data.shape}\n")

    results = {}

    # Test each factor
    results['censoring'] = test_factor_censoring(data, n_runs=12)
    results['multiround'] = test_factor_multiround(data, n_runs=12)
    results['probe'] = test_factor_probe_constraint(data, n_runs=12)
    results['proportion'] = test_factor_censoring_proportion(data, n_runs=12)
    results['training'] = test_factor_model_training(data, n_runs=12)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Main Culprits Identified")
    print("="*80 + "\n")

    culprits = []

    if results['censoring'].get('significant_with_fix'):
        culprits.append("✓ CENSORING RANDOMNESS - deterministic achieves p<0.05!")
    elif results['censoring']['improvement']:
        culprits.append(f"  Censoring randomness - reduces p by {results['censoring']['p_random'] - results['censoring']['p_deterministic']:.4f}")

    if results['multiround'].get('significant_with_fix'):
        culprits.append("✓ ONE-SHOT AL - multi-round achieves p<0.05!")
    elif results['multiround']['multiround_better']:
        culprits.append(f"  One-shot AL - multi-round improves by {results['multiround']['variance_reduction']:.4f} variance")

    if abs(results['probe']['constraint_cost']) > 0.01:
        culprits.append(f"  Probe depth constraint - costs {results['probe']['constraint_cost']:.4f} performance")

    if results['proportion']['prop70_better']:
        culprits.append(f"  Censoring proportion - 70% improves by {results['proportion']['prop70_mean'] - results['proportion']['prop50_mean']:.4f}")

    if results['training']['variance_reduction'] > 0.001:
        culprits.append(f"  Model training - 50 epochs reduces variance by {results['training']['variance_reduction']:.4f}")

    if len(culprits) == 0:
        print("✗ NO SINGLE FACTOR resolves the problem.")
        print("  The issue is fundamental to the dataset/task combination.")
    else:
        for culprit in culprits:
            print(culprit)

    print("\n" + "="*80 + "\n")

    return results


if __name__ == '__main__':
    results = main()
