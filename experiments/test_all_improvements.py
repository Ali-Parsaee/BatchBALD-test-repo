"""
Comprehensive test of ALL BatchBALD improvements.

Tests:
1. Observable mass
2. MI * P(death)
3. MI * P(death)^2
4. MI * Observable mass
5. Filtered MI (filter low P(death))
6. Iterative batch selection
7. Combinations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.synthetic import generate_survival_data, artificially_censor, split_data
from src.models.survival_model import BayesianSurvivalModel
from src.oracle.oracle import Oracle
from src.acquisition.batchbald import SurvivalBatchBALD
from src.acquisition.improved_batchbald import ImprovedBatchBALD, IterativeBatchBALD
from src.acquisition.entropy import EntropyAcquisition
from src.evaluation.metrics import concordance_index
from scipy.stats import ttest_rel


def run_single_experiment(
    acq_name,
    acq_func,
    data,
    train_data,
    test_data,
    true_train_time,
    true_train_event,
    artificial_time,
    artificial_event,
    probe_depth=5,
    batch_size=50
):
    """Run experiment with a single acquisition function."""

    # Train initial model
    initial_model = BayesianSurvivalModel(
        n_features=10, n_time_bins=15, n_ensemble=5, hidden_size=64
    )
    initial_model.fit(
        train_data['X'], artificial_time, artificial_event,
        epochs=40, batch_size=32, verbose=False
    )

    test_preds = initial_model.predict_proba(test_data['X'])
    initial_c_index = concordance_index(test_preds, test_data['time'], test_data['event'])

    # Oracle
    oracle = Oracle(true_train_time, true_train_event, probe_depth)
    train_preds = initial_model.predict_proba(train_data['X'])
    oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
        train_preds, artificial_time, artificial_event
    )

    # Select batch
    if hasattr(acq_func, 'select_batch'):
        # Check if it needs predictions argument
        if acq_name.startswith('Improved') or acq_name.startswith('Iterative'):
            selected = acq_func.select_batch(
                oracle_probs, train_preds, artificial_time,
                batch_size, artificial_event
            )
        else:
            selected = acq_func.select_batch(
                oracle_probs, batch_size, artificial_event
            )
    else:
        # Entropy
        selected = acq_func.select_batch(
            train_preds, batch_size, artificial_event
        )

    # Query oracle
    updated_time, updated_event = oracle.query(selected, artificial_time, artificial_event)
    n_revealed = ((updated_event - artificial_event) > 0).sum()

    # Retrain
    updated_model = BayesianSurvivalModel(
        n_features=10, n_time_bins=15, n_ensemble=5, hidden_size=64
    )
    updated_model.fit(
        train_data['X'], updated_time, updated_event,
        epochs=40, batch_size=32, verbose=False
    )

    # Evaluate
    updated_preds = updated_model.predict_proba(test_data['X'])
    updated_c_index = concordance_index(updated_preds, test_data['time'], test_data['event'])

    return {
        'initial': initial_c_index,
        'updated': updated_c_index,
        'improvement': updated_c_index - initial_c_index,
        'n_revealed': n_revealed,
        'n_selected': len(selected)
    }


def test_all_improvements(n_runs=5):
    """Test all improvements."""

    print("\n" + "="*100)
    print("COMPREHENSIVE TEST: All BatchBALD Improvements")
    print("="*100 + "\n")

    probe_depth = 5
    batch_size = 50

    # Define all methods to test
    methods = {
        # Baselines
        'Entropy': EntropyAcquisition(),
        'BatchBALD (original)': SurvivalBatchBALD(probe_depth=probe_depth),

        # Improved variants
        'Improved: Observable Mass': ImprovedBatchBALD(probe_depth, variant='observable_mass'),
        'Improved: MI * P(death)': ImprovedBatchBALD(probe_depth, variant='mi_times_pdeath'),
        'Improved: MI * P(death)^2': ImprovedBatchBALD(probe_depth, variant='mi_times_pdeath_sq'),
        'Improved: MI * ObsMass': ImprovedBatchBALD(probe_depth, variant='combined'),
        'Improved: Filtered MI': ImprovedBatchBALD(probe_depth, variant='filtered'),

        # Iterative selection
        'Iterative: BatchBALD (5x10)': IterativeBatchBALD(
            SurvivalBatchBALD(probe_depth), sub_batch_size=10
        ),
        'Iterative: ObsMass (5x10)': IterativeBatchBALD(
            ImprovedBatchBALD(probe_depth, variant='observable_mass'), sub_batch_size=10
        ),
    }

    all_results = []

    for run in range(n_runs):
        print(f"\n{'='*100}")
        print(f"RUN {run+1}/{n_runs}")
        print(f"{'='*100}\n")

        # Generate data
        data = generate_survival_data(
            n_samples=1000, n_features=10, n_time_bins=15,
            censoring_rate=0.3, random_state=42 + run
        )

        train_data, test_data = split_data(data, train_size=0.7, random_state=42 + run)
        true_train_time = train_data['time'].copy()
        true_train_event = train_data['event'].copy()

        artificial_time, artificial_event = artificially_censor(
            train_data['time'], train_data['event'],
            proportion=0.5, random_state=42 + run
        )

        run_results = {}

        for method_name, acq_func in methods.items():
            print(f"  Testing {method_name}...", end=" ", flush=True)

            result = run_single_experiment(
                method_name, acq_func, data, train_data, test_data,
                true_train_time, true_train_event,
                artificial_time, artificial_event,
                probe_depth, batch_size
            )

            run_results[method_name] = result

            print(f"Δ={result['improvement']:+.4f}, Revealed={result['n_revealed']}/{batch_size}")

        all_results.append(run_results)

    # ===========================================================================================
    # SUMMARY
    # ===========================================================================================

    print(f"\n\n{'='*100}")
    print("SUMMARY ACROSS ALL RUNS")
    print(f"{'='*100}\n")

    print(f"{'Method':<35} {'Mean Δ':<12} {'Std':<10} {'Wins':<8} {'Avg Reveals':<12} {'vs Entropy p':<12}")
    print("-"*100)

    summary = {}
    for method_name in methods.keys():
        improvements = [r[method_name]['improvement'] for r in all_results]
        n_reveals = [r[method_name]['n_revealed'] for r in all_results]

        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        wins = sum(1 for imp in improvements if imp > 0)
        avg_reveals = np.mean(n_reveals)

        summary[method_name] = {
            'mean': mean_imp,
            'std': std_imp,
            'wins': wins,
            'improvements': improvements,
            'avg_reveals': avg_reveals
        }

        # Test vs Entropy
        entropy_imps = summary['Entropy']['improvements']
        if len(improvements) >= 3 and method_name != 'Entropy':
            t_stat, p_value = ttest_rel(improvements, entropy_imps)
        else:
            p_value = 1.0

        marker = "🏆" if mean_imp > 0.015 else ("★" if mean_imp > 0.010 else " ")

        print(f"{marker} {method_name:<33} {mean_imp:+.4f}      {std_imp:.4f}    {wins}/{n_runs}     "
              f"{avg_reveals:.1f}           {p_value:.4f}")

    # ===========================================================================================
    # BEST METHOD
    # ===========================================================================================

    print(f"\n{'='*100}")
    print("BEST METHODS")
    print(f"{'='*100}\n")

    # Sort by mean improvement
    sorted_methods = sorted(summary.items(), key=lambda x: x[1]['mean'], reverse=True)

    print("Top 5 by mean improvement:\n")
    for rank, (name, stats) in enumerate(sorted_methods[:5], 1):
        print(f"{rank}. {name:<35} Δ={stats['mean']:+.4f} ± {stats['std']:.4f}, "
              f"Reveals={stats['avg_reveals']:.1f}/{batch_size}")

    # Statistical test: Best vs Entropy
    best_name = sorted_methods[0][0]
    best_imps = summary[best_name]['improvements']
    ent_imps = summary['Entropy']['improvements']

    if best_name != 'Entropy':
        t_stat, p_value = ttest_rel(best_imps, ent_imps)

        print(f"\n{'='*100}")
        print(f"STATISTICAL TEST: {best_name} vs Entropy")
        print(f"{'='*100}")
        print(f"t-statistic: {t_stat:.3f}")
        print(f"p-value: {p_value:.4f}")

        if p_value < 0.05:
            if summary[best_name]['mean'] > summary['Entropy']['mean']:
                print(f"\n🎉 {best_name} SIGNIFICANTLY BEATS Entropy! (p < 0.05)")
            else:
                print(f"\n✗ Entropy significantly better")
        else:
            improvement_vs_entropy = summary[best_name]['mean'] - summary['Entropy']['mean']
            print(f"\n⚠️  Not statistically significant (p = {p_value:.4f})")
            print(f"But {best_name} is {improvement_vs_entropy:+.4f} better on average")

    # ===========================================================================================
    # INSIGHTS
    # ===========================================================================================

    print(f"\n{'='*100}")
    print("KEY INSIGHTS")
    print(f"{'='*100}\n")

    # Compare iterative vs non-iterative
    if 'Iterative: BatchBALD (5x10)' in summary and 'BatchBALD (original)' in summary:
        iterative_mean = summary['Iterative: BatchBALD (5x10)']['mean']
        original_mean = summary['BatchBALD (original)']['mean']
        iterative_reveals = summary['Iterative: BatchBALD (5x10)']['avg_reveals']
        original_reveals = summary['BatchBALD (original)']['avg_reveals']

        print(f"1. Iterative Selection (5x10 vs 1x50):")
        print(f"   Iterative BatchBALD: Δ={iterative_mean:+.4f}, Reveals={iterative_reveals:.1f}")
        print(f"   Original BatchBALD:  Δ={original_mean:+.4f}, Reveals={original_reveals:.1f}")
        print(f"   → Improvement: {iterative_mean - original_mean:+.4f}")

        if iterative_mean > original_mean + 0.005:
            print(f"   ✅ Iterative selection HELPS!")
        else:
            print(f"   ⚠️  Iterative selection doesn't help much")

    # Compare improved variants to original
    print(f"\n2. Improved Variants vs Original BatchBALD:")
    original_mean = summary['BatchBALD (original)']['mean']

    improved_variants = [k for k in summary.keys() if k.startswith('Improved:')]
    for variant in improved_variants:
        variant_mean = summary[variant]['mean']
        diff = variant_mean - original_mean

        marker = "✅" if diff > 0.005 else ("~" if abs(diff) < 0.005 else "❌")
        print(f"   {marker} {variant:40s} Δ={diff:+.4f}")

    # Observable mass analysis
    if 'Improved: Observable Mass' in summary:
        obs_mass_mean = summary['Improved: Observable Mass']['mean']
        obs_mass_reveals = summary['Improved: Observable Mass']['avg_reveals']

        print(f"\n3. Observable Mass Strategy:")
        print(f"   Mean improvement: {obs_mass_mean:+.4f}")
        print(f"   Deaths revealed: {obs_mass_reveals:.1f}/{batch_size} ({obs_mass_reveals/batch_size*100:.1f}%)")

        if obs_mass_reveals > original_reveals + 2:
            print(f"   ✅ Reveals {obs_mass_reveals - original_reveals:.1f} MORE deaths than original!")

    print(f"\n{'='*100}\n")

    return summary


if __name__ == '__main__':
    summary = test_all_improvements(n_runs=5)
