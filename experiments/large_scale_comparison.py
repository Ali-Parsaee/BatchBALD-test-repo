"""
Large-scale experimental comparison with varying parameters.

Tests:
- Different dataset sizes (500, 1000, 2000)
- Different batch sizes (25, 50, 100)
- Different probe depths (2, 5, 10)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.synthetic import generate_survival_data, artificially_censor, split_data
from src.models.survival_model import BayesianSurvivalModel
from src.oracle.oracle import Oracle
from src.acquisition.batchbald import SurvivalBatchBALD
from src.acquisition.entropy import EntropyAcquisition
from src.acquisition.variance import VarianceAcquisition
from src.evaluation.metrics import concordance_index, brier_score
from scipy.stats import ttest_rel


def run_experiment(
    n_samples=1000,
    n_features=10,
    n_time_bins=15,
    probe_depth=5,
    batch_size=50,
    artificial_censor_rate=0.5,
    n_ensemble=5,
    random_state=42
):
    """Run a single experiment with specified parameters."""

    # Generate data
    data = generate_survival_data(
        n_samples=n_samples,
        n_features=n_features,
        n_time_bins=n_time_bins,
        censoring_rate=0.3,
        random_state=random_state
    )

    train_data, test_data = split_data(data, train_size=0.7, random_state=random_state)

    true_train_time = train_data['time'].copy()
    true_train_event = train_data['event'].copy()

    # Artificially censor
    artificial_time, artificial_event = artificially_censor(
        train_data['time'],
        train_data['event'],
        proportion=artificial_censor_rate,
        random_state=random_state
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

    # Evaluate initial model
    test_preds = initial_model.predict_proba(test_data['X'])
    initial_c_index = concordance_index(test_preds, test_data['time'], test_data['event'])

    # Oracle
    oracle = Oracle(true_train_time, true_train_event, probe_depth)

    # Train predictions
    train_preds = initial_model.predict_proba(train_data['X'])

    results = {'initial': {'c_index': initial_c_index}}

    # Test each method
    methods = {
        'BatchBALD': SurvivalBatchBALD(probe_depth=probe_depth),
        'Entropy': EntropyAcquisition(),
        'Variance': VarianceAcquisition()
    }

    for method_name, acq_func in methods.items():
        # Select batch
        if method_name == 'BatchBALD':
            oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
                train_preds, artificial_time, artificial_event
            )
            selected_indices = acq_func.select_batch(
                oracle_probs, batch_size=batch_size, current_event=artificial_event
            )
        else:
            selected_indices = acq_func.select_batch(
                train_preds, batch_size=batch_size, current_event=artificial_event
            )

        # Query oracle
        updated_time, updated_event = oracle.query(
            selected_indices, artificial_time, artificial_event
        )

        n_revealed = ((updated_event - artificial_event) > 0).sum()

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

        results[method_name] = {
            'c_index': updated_c_index,
            'improvement': updated_c_index - initial_c_index,
            'n_revealed': n_revealed,
            'n_selected': len(selected_indices)
        }

    return results


def run_parameter_sweep():
    """Run experiments with different parameter configurations."""

    print("\n" + "="*100)
    print("LARGE-SCALE PARAMETER SWEEP")
    print("="*100)

    # Configuration matrix
    configs = [
        # (n_samples, batch_size, probe_depth, n_runs, description)
        (500, 25, 2, 5, "Baseline (small)"),
        (1000, 50, 2, 5, "Medium dataset, medium batch"),
        (1000, 100, 2, 5, "Medium dataset, large batch"),
        (1000, 50, 5, 5, "Medium dataset, large probe depth"),
        (1000, 50, 10, 5, "Medium dataset, very large probe depth"),
        (2000, 100, 5, 3, "Large dataset, large batch, large probe"),
    ]

    all_config_results = []

    for config_idx, (n_samples, batch_size, probe_depth, n_runs, description) in enumerate(configs):
        print(f"\n{'#'*100}")
        print(f"# CONFIG {config_idx + 1}/{len(configs)}: {description}")
        print(f"# Samples={n_samples}, Batch={batch_size}, ProbeDepth={probe_depth}, Runs={n_runs}")
        print(f"{'#'*100}\n")

        config_results = []

        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}...", end=" ", flush=True)

            results = run_experiment(
                n_samples=n_samples,
                n_features=10,
                n_time_bins=15,
                probe_depth=probe_depth,
                batch_size=batch_size,
                artificial_censor_rate=0.5,
                n_ensemble=5,
                random_state=42 + run
            )

            config_results.append(results)

            # Quick summary
            bald_imp = results['BatchBALD']['improvement']
            ent_imp = results['Entropy']['improvement']
            print(f"BatchBALD={bald_imp:+.4f}, Entropy={ent_imp:+.4f}")

        # Aggregate results for this config
        methods = ['BatchBALD', 'Entropy', 'Variance']

        print(f"\n  SUMMARY FOR CONFIG {config_idx + 1}:")
        print(f"  {'-'*80}")
        print(f"  {'Method':<12} {'Mean Δ':<12} {'Std':<10} {'Wins':<8} {'p-value':<10}")
        print(f"  {'-'*80}")

        summary = {}
        for method in methods:
            improvements = [r[method]['improvement'] for r in config_results]
            mean_imp = np.mean(improvements)
            std_imp = np.std(improvements)
            wins = sum(1 for imp in improvements if imp > 0)

            summary[method] = {
                'mean': mean_imp,
                'std': std_imp,
                'wins': wins,
                'improvements': improvements
            }

        # Paired t-test: BatchBALD vs Entropy
        bald_imps = summary['BatchBALD']['improvements']
        ent_imps = summary['Entropy']['improvements']

        if n_runs >= 3:
            t_stat, p_value = ttest_rel(bald_imps, ent_imps)
        else:
            p_value = 1.0

        for method in methods:
            s = summary[method]
            marker = "✓" if method == 'BatchBALD' and s['mean'] > summary['Entropy']['mean'] else " "
            print(f"  {marker} {method:<10} {s['mean']:+.4f}      {s['std']:.4f}    "
                  f"{s['wins']}/{n_runs}     {p_value if method=='BatchBALD' else '-':<10}")

        print(f"  {'-'*80}")

        if p_value < 0.05:
            print(f"  🎉 BatchBALD wins SIGNIFICANTLY (p={p_value:.4f} < 0.05)")
        elif summary['BatchBALD']['mean'] > summary['Entropy']['mean']:
            print(f"  ⚠️  BatchBALD wins marginally (p={p_value:.4f} >= 0.05, NOT significant)")
        else:
            print(f"  ❌ Entropy wins (p={p_value:.4f})")

        all_config_results.append({
            'config': (n_samples, batch_size, probe_depth),
            'description': description,
            'summary': summary,
            'p_value': p_value,
            'n_runs': n_runs
        })

    # Final summary across all configs
    print(f"\n\n{'='*100}")
    print("FINAL SUMMARY ACROSS ALL CONFIGURATIONS")
    print(f"{'='*100}\n")

    print(f"{'Config':<35} {'N':<6} {'Batch':<7} {'Probe':<7} {'BatchBALD Δ':<15} {'Entropy Δ':<15} {'Winner':<12} {'Sig?':<10}")
    print("-"*100)

    for config_data in all_config_results:
        desc = config_data['description']
        n_samples, batch_size, probe_depth = config_data['config']
        summary = config_data['summary']
        p_value = config_data['p_value']

        bald_mean = summary['BatchBALD']['mean']
        ent_mean = summary['Entropy']['mean']

        winner = "BatchBALD" if bald_mean > ent_mean else "Entropy"
        sig = "YES ✓" if p_value < 0.05 else "NO"

        print(f"{desc:<35} {n_samples:<6} {batch_size:<7} {probe_depth:<7} "
              f"{bald_mean:+.4f}         {ent_mean:+.4f}         "
              f"{winner:<12} {sig:<10}")

    print("="*100)

    # Count significant wins
    sig_wins = sum(1 for c in all_config_results
                   if c['p_value'] < 0.05 and c['summary']['BatchBALD']['mean'] > c['summary']['Entropy']['mean'])
    total_configs = len(all_config_results)

    print(f"\n🎯 BatchBALD SIGNIFICANTLY wins in {sig_wins}/{total_configs} configurations")

    if sig_wins >= total_configs / 2:
        print("✅ CONCLUSION: BatchBALD convincingly beats baselines at scale!")
    else:
        print("⚠️  CONCLUSION: BatchBALD advantage remains marginal even at scale")

    print("\n" + "="*100 + "\n")


if __name__ == '__main__':
    run_parameter_sweep()
