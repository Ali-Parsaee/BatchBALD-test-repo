"""
Compare True C-BALD vs Old C-BALD (BatchBALD) on different budgets.

Run comparison on batch sizes: 10, 20, 30
"""

import numpy as np
from scipy.stats import ttest_rel

from src.data.synthetic import generate_survival_data, artificially_censor, split_data
from src.models.survival_model import BayesianSurvivalModel
from src.oracle.oracle import Oracle
from src.acquisition.batchbald import SurvivalBatchBALD
from src.acquisition.cbald_true import TrueCBALD
from src.evaluation.metrics import concordance_index, brier_score


def run_single_comparison(
    batch_size,
    n_samples=500,
    n_features=10,
    n_time_bins=10,
    probe_depth=2,
    n_ensemble=5,
    n_runs=10,
    start_seed=42
):
    """Run comparison for a single batch size."""

    print(f"\n{'='*80}")
    print(f"BATCH SIZE: {batch_size}")
    print(f"{'='*80}\n")

    results = {
        'batchbald': [],
        'cbald_true': []
    }

    for run in range(n_runs):
        seed = start_seed + run

        if run % 3 == 0:
            print(f"Run {run+1}/{n_runs}...", end=" ", flush=True)

        # Generate data
        data = generate_survival_data(
            n_samples=n_samples,
            n_features=n_features,
            n_time_bins=n_time_bins,
            censoring_rate=0.3,
            random_state=seed
        )

        train_data, test_data = split_data(data, train_size=0.7, random_state=seed)

        # Ground truth
        true_train_time = train_data['time'].copy()
        true_train_event = train_data['event'].copy()

        # Artificially censor
        artificial_time, artificial_event = artificially_censor(
            train_data['time'],
            train_data['event'],
            proportion=0.5,
            random_state=seed
        )

        # Train initial model
        initial_model = BayesianSurvivalModel(
            n_features=n_features,
            n_time_bins=n_time_bins,
            n_ensemble=n_ensemble,
            hidden_size=64
        )
        initial_model.fit(
            train_data['X'],
            artificial_time,
            artificial_event,
            epochs=50,
            batch_size=32,
            verbose=False
        )

        # Evaluate initial
        test_preds = initial_model.predict_proba(test_data['X'])
        initial_c_index = concordance_index(test_preds, test_data['time'], test_data['event'])

        # Oracle
        oracle = Oracle(true_train_time, true_train_event, probe_depth)

        # Train predictions
        train_preds = initial_model.predict_proba(train_data['X'])

        # Test both methods
        methods = {
            'batchbald': SurvivalBatchBALD(probe_depth=probe_depth),
            'cbald_true': TrueCBALD(probe_depth=probe_depth)
        }

        for method_name, acq_func in methods.items():
            # Select batch
            if method_name == 'batchbald':
                oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
                    train_preds, artificial_time, artificial_event
                )
                selected_indices = acq_func.select_batch(
                    oracle_probs,
                    batch_size=batch_size,
                    current_event=artificial_event
                )
            else:  # cbald_true
                selected_indices = acq_func.select_batch(
                    train_preds,
                    artificial_time,
                    batch_size=batch_size,
                    current_event=artificial_event
                )

            # Query oracle
            updated_time, updated_event = oracle.query(
                selected_indices, artificial_time, artificial_event
            )

            # Retrain
            updated_model = BayesianSurvivalModel(
                n_features=n_features,
                n_time_bins=n_time_bins,
                n_ensemble=n_ensemble,
                hidden_size=64
            )
            updated_model.fit(
                train_data['X'],
                updated_time,
                updated_event,
                epochs=50,
                batch_size=32,
                verbose=False
            )

            # Evaluate
            updated_test_preds = updated_model.predict_proba(test_data['X'])
            updated_c_index = concordance_index(
                updated_test_preds, test_data['time'], test_data['event']
            )

            improvement = updated_c_index - initial_c_index
            results[method_name].append(improvement)

        if run % 3 == 2 or run == n_runs - 1:
            print("done")

    # Print results
    print(f"\n{'Method':<15} {'Δ C-index':<20} {'Wins'}")
    print("-" * 50)

    for method_name in ['batchbald', 'cbald_true']:
        improvements = results[method_name]
        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        wins = sum(1 for imp in improvements if imp > 0)

        print(f"{method_name:<15} {mean_imp:+.4f} ± {std_imp:.4f}     {wins}/{n_runs}")

    # Statistical test
    if len(results['batchbald']) >= 3:
        t_stat, p_value = ttest_rel(results['cbald_true'], results['batchbald'])
        mean_diff = np.mean(results['cbald_true']) - np.mean(results['batchbald'])

        print(f"\nCBALD_true vs BatchBALD:")
        print(f"  Mean difference: {mean_diff:+.4f}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            winner = "CBALD_true" if mean_diff > 0 else "BatchBALD"
            print(f"  Result: {winner} is significantly better (p<0.05) {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*'}")
        else:
            print(f"  Result: No significant difference")

    return results


def main():
    print("\n" + "="*80)
    print("COMPARING TRUE C-BALD vs OLD C-BALD (BatchBALD)")
    print("="*80)
    print("\nExperimental setup:")
    print("  - Samples: 500")
    print("  - Features: 10")
    print("  - Time bins: 10")
    print("  - Probe depth: 2")
    print("  - Ensemble: 5")
    print("  - Runs per batch size: 10")
    print("  - Batch sizes: [10, 20, 30]")
    print("="*80)

    batch_sizes = [10, 20, 30]
    all_results = {}

    for batch_size in batch_sizes:
        results = run_single_comparison(
            batch_size=batch_size,
            n_runs=10,
            start_seed=42
        )
        all_results[batch_size] = results

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Batch Size':<12} {'BatchBALD':<20} {'CBALD_true':<20} {'Winner'}")
    print("-" * 70)

    for batch_size in batch_sizes:
        results = all_results[batch_size]

        batchbald_mean = np.mean(results['batchbald'])
        batchbald_std = np.std(results['batchbald'])

        cbald_mean = np.mean(results['cbald_true'])
        cbald_std = np.std(results['cbald_true'])

        # Determine winner
        t_stat, p_value = ttest_rel(results['cbald_true'], results['batchbald'])
        if p_value < 0.05:
            winner = "CBALD_true *" if cbald_mean > batchbald_mean else "BatchBALD *"
        else:
            winner = "Tie"

        print(f"{batch_size:<12} {batchbald_mean:+.4f}±{batchbald_std:.4f}     "
              f"{cbald_mean:+.4f}±{cbald_std:.4f}     {winner}")

    print("\n* = statistically significant (p<0.05)")
    print(f"\n{'='*80}\n")

    print("Comparison complete!")


if __name__ == '__main__':
    main()
