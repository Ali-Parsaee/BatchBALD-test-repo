"""
Test if Weighted BatchBALD fixes the problem.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.synthetic import generate_survival_data, artificially_censor, split_data
from src.models.survival_model import BayesianSurvivalModel
from src.oracle.oracle import Oracle
from src.acquisition.batchbald import SurvivalBatchBALD
from src.acquisition.weighted_batchbald import WeightedBatchBALD
from src.acquisition.entropy import EntropyAcquisition
from src.evaluation.metrics import concordance_index
from scipy.stats import ttest_rel


def test_weighted_batchbald(n_runs=5):
    """Compare original vs weighted BatchBALD."""

    print("\n" + "="*100)
    print("TESTING WEIGHTED BATCHBALD")
    print("="*100 + "\n")

    all_results = []

    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}...", end=" ", flush=True)

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
        probe_depth = 5
        batch_size = 50
        oracle = Oracle(true_train_time, true_train_event, probe_depth)
        train_preds = initial_model.predict_proba(train_data['X'])
        oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
            train_preds, artificial_time, artificial_event
        )

        results = {'initial': initial_c_index}

        # Test methods
        methods = {
            'BatchBALD': SurvivalBatchBALD(probe_depth=probe_depth),
            'WeightedBALD': WeightedBatchBALD(probe_depth=probe_depth, death_weight=1.0, censor_weight=0.3),
            'WeightedBALD_v2': WeightedBatchBALD(probe_depth=probe_depth, death_weight=1.0, censor_weight=0.1),
            'Entropy': EntropyAcquisition()
        }

        for method_name, acq_func in methods.items():
            # Select batch
            if 'BALD' in method_name or 'Weighted' in method_name:
                selected = acq_func.select_batch(oracle_probs, batch_size, artificial_event)
            else:
                selected = acq_func.select_batch(train_preds, batch_size, artificial_event)

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

            results[method_name] = {
                'c_index': updated_c_index,
                'improvement': updated_c_index - initial_c_index,
                'n_revealed': n_revealed
            }

        all_results.append(results)

        # Quick summary
        bald = results['BatchBALD']['improvement']
        weighted = results['WeightedBALD']['improvement']
        weighted_v2 = results['WeightedBALD_v2']['improvement']
        ent = results['Entropy']['improvement']

        print(f"BALD={bald:+.4f}, Weighted={weighted:+.4f}, Weighted_v2={weighted_v2:+.4f}, Ent={ent:+.4f}")

    # Aggregate results
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}\n")

    methods = ['BatchBALD', 'WeightedBALD', 'WeightedBALD_v2', 'Entropy']

    print(f"{'Method':<18} {'Mean Δ':<12} {'Std':<10} {'Wins':<8} {'Avg Reveals':<12}")
    print("-"*70)

    summary = {}
    for method in methods:
        improvements = [r[method]['improvement'] for r in all_results]
        n_reveals = [r[method]['n_revealed'] for r in all_results]

        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        wins = sum(1 for imp in improvements if imp > 0)
        avg_reveals = np.mean(n_reveals)

        summary[method] = {
            'mean': mean_imp,
            'std': std_imp,
            'wins': wins,
            'improvements': improvements,
            'avg_reveals': avg_reveals
        }

        print(f"{method:<18} {mean_imp:+.4f}      {std_imp:.4f}    {wins}/{n_runs}     {avg_reveals:.1f}")

    # Statistical tests
    print(f"\n{'='*100}")
    print("STATISTICAL SIGNIFICANCE")
    print(f"{'='*100}\n")

    bald_imps = summary['BatchBALD']['improvements']
    weighted_imps = summary['WeightedBALD']['improvements']
    weighted_v2_imps = summary['WeightedBALD_v2']['improvements']
    ent_imps = summary['Entropy']['improvements']

    # Weighted vs Original
    t_stat, p_value = ttest_rel(weighted_imps, bald_imps)
    print(f"WeightedBALD vs BatchBALD: t={t_stat:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        if summary['WeightedBALD']['mean'] > summary['BatchBALD']['mean']:
            print(f"  ✓ Weighted significantly BETTER")
        else:
            print(f"  ✗ Weighted significantly WORSE")
    else:
        print(f"  ~ No significant difference")

    # Weighted v2 vs Original
    t_stat, p_value = ttest_rel(weighted_v2_imps, bald_imps)
    print(f"\nWeightedBALD_v2 vs BatchBALD: t={t_stat:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        if summary['WeightedBALD_v2']['mean'] > summary['BatchBALD']['mean']:
            print(f"  ✓ Weighted_v2 significantly BETTER")
        else:
            print(f"  ✗ Weighted_v2 significantly WORSE")
    else:
        print(f"  ~ No significant difference")

    # Best weighted vs Entropy
    best_weighted = 'WeightedBALD' if summary['WeightedBALD']['mean'] > summary['WeightedBALD_v2']['mean'] else 'WeightedBALD_v2'
    best_weighted_imps = summary[best_weighted]['improvements']

    t_stat, p_value = ttest_rel(best_weighted_imps, ent_imps)
    print(f"\n{best_weighted} vs Entropy: t={t_stat:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        if summary[best_weighted]['mean'] > summary['Entropy']['mean']:
            print(f"  🎉 {best_weighted} significantly BEATS Entropy!")
        else:
            print(f"  ✗ Entropy wins")
    else:
        print(f"  ~ No significant difference")

    # Conclusion
    print(f"\n{'='*100}")
    print("CONCLUSION")
    print(f"{'='*100}\n")

    best_method = max(methods, key=lambda m: summary[m]['mean'])
    print(f"Best method: {best_method}")
    print(f"Mean improvement: {summary[best_method]['mean']:+.4f}")
    print(f"Average deaths revealed: {summary[best_method]['avg_reveals']:.1f}/{batch_size}")

    if best_method.startswith('Weighted'):
        print(f"\n✅ SUCCESS: Weighted BatchBALD improves over original!")
        print(f"   Improvement: {summary[best_method]['mean'] - summary['BatchBALD']['mean']:+.4f} over BatchBALD")
    else:
        print(f"\n⚠️  Weighting didn't help significantly")

    print("\n" + "="*100 + "\n")


if __name__ == '__main__':
    test_weighted_batchbald(n_runs=5)
