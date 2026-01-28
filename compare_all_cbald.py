"""
Compare Old CBALD, New C-BALD variants, and BatchBALD on different budgets.

Methods tested:
- BatchBALD: Greedy BatchBALD with conditional independence approximation
- Old CBALD: Simple top-k selection based on C-BALD scores
- New C-BALD Diverse: Hybrid C-BALD + diversity (winner at budget=20)
- New C-BALD Adaptive: Adaptive ratio (winner at budget=10)

Batch sizes: 10, 20, 30
"""

import numpy as np
from scipy.stats import ttest_rel

from src.data.synthetic import generate_survival_data, artificially_censor, split_data
from src.models.survival_model import BayesianSurvivalModel
from src.oracle.oracle import Oracle
from src.acquisition.batchbald import SurvivalBatchBALD
from src.acquisition.cbald_diverse import OldCBALD, CBaldDiverse, CBaldDiverseAdaptive
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

    methods = {
        'BatchBALD': SurvivalBatchBALD(probe_depth=probe_depth),
        'Old_CBALD': OldCBALD(probe_depth=probe_depth),
        'New_CBALD_Diverse': CBaldDiverse(probe_depth=probe_depth, diversity_ratio=0.3),
        'New_CBALD_Adaptive': CBaldDiverseAdaptive(probe_depth=probe_depth, start_ratio=0.1, end_ratio=0.5)
    }

    results = {name: [] for name in methods.keys()}

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

        # Test each method
        for method_name, acq_func in methods.items():
            # Select batch
            if method_name == 'BatchBALD':
                oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
                    train_preds, artificial_time, artificial_event
                )
                selected_indices = acq_func.select_batch(
                    oracle_probs,
                    batch_size=batch_size,
                    current_event=artificial_event
                )
            else:  # Old CBALD and New CBALD variants
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
    print(f"\n{'Method':<25} {'Δ C-index':<20} {'Wins'}")
    print("-" * 70)

    for method_name in methods.keys():
        improvements = results[method_name]
        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        wins = sum(1 for imp in improvements if imp > 0)

        print(f"{method_name:<25} {mean_imp:+.4f} ± {std_imp:.4f}     {wins}/{n_runs}")

    # Statistical tests
    print(f"\n{'='*80}")
    print("STATISTICAL TESTS (pairwise comparisons)")
    print(f"{'='*80}\n")

    method_names = list(methods.keys())
    for i in range(len(method_names)):
        for j in range(i+1, len(method_names)):
            method1 = method_names[i]
            method2 = method_names[j]

            improvements1 = results[method1]
            improvements2 = results[method2]

            if len(improvements1) >= 3:
                t_stat, p_value = ttest_rel(improvements1, improvements2)
                mean_diff = np.mean(improvements1) - np.mean(improvements2)

                sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""

                winner = method1 if mean_diff > 0 else method2
                if p_value >= 0.05:
                    winner = "Tie"

                print(f"{method1} vs {method2}:")
                print(f"  Δ={mean_diff:+.4f}, p={p_value:.4f} {sig_marker} → {winner}")

    return results


def main():
    print("\n" + "="*80)
    print("COMPARING OLD CBALD vs NEW CBALD vs BATCHBALD")
    print("="*80)
    print("\nMethods:")
    print("  - BatchBALD: Greedy BatchBALD with conditional independence")
    print("  - Old CBALD: Simple top-k selection (time_variance * (0.5 + 0.5 * death_prob))")
    print("  - New C-BALD Diverse: Hybrid C-BALD + diversity (winner at budget=20)")
    print("  - New C-BALD Adaptive: Adaptive ratio (winner at budget=10)")
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

    print(f"{'Batch':<8} {'BatchBALD':<18} {'Old_CBALD':<18} {'New_Diverse':<18} {'New_Adaptive':<18} {'Best'}")
    print("-" * 105)

    for batch_size in batch_sizes:
        results = all_results[batch_size]

        batchbald_mean = np.mean(results['BatchBALD'])
        batchbald_std = np.std(results['BatchBALD'])

        old_cbald_mean = np.mean(results['Old_CBALD'])
        old_cbald_std = np.std(results['Old_CBALD'])

        new_diverse_mean = np.mean(results['New_CBALD_Diverse'])
        new_diverse_std = np.std(results['New_CBALD_Diverse'])

        new_adaptive_mean = np.mean(results['New_CBALD_Adaptive'])
        new_adaptive_std = np.std(results['New_CBALD_Adaptive'])

        # Determine best
        means = {
            'BatchBALD': batchbald_mean,
            'Old_CBALD': old_cbald_mean,
            'New_Diverse': new_diverse_mean,
            'New_Adaptive': new_adaptive_mean
        }
        best_method = max(means, key=means.get)

        print(f"{batch_size:<8} "
              f"{batchbald_mean:+.4f}±{batchbald_std:.4f}    "
              f"{old_cbald_mean:+.4f}±{old_cbald_std:.4f}    "
              f"{new_diverse_mean:+.4f}±{new_diverse_std:.4f}    "
              f"{new_adaptive_mean:+.4f}±{new_adaptive_std:.4f}    "
              f"{best_method}")

    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}\n")

    # Rank methods by average performance across all budgets
    avg_improvements = {}
    for method_name in ['BatchBALD', 'Old_CBALD', 'New_CBALD_Diverse', 'New_CBALD_Adaptive']:
        all_improvements = []
        for batch_size in batch_sizes:
            all_improvements.extend(all_results[batch_size][method_name])
        avg_improvements[method_name] = np.mean(all_improvements)

    print("Average improvement across all budgets:")
    for method_name, avg_imp in sorted(avg_improvements.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method_name:<25} {avg_imp:+.4f}")

    print(f"\n{'='*80}\n")

    print("Comparison complete!")


if __name__ == '__main__':
    main()
