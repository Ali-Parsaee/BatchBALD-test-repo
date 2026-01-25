"""
Fast diagnostic experiment with smaller parameters.
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


def run_fast_experiment(random_state=42):
    """Quick experiment with smaller parameters."""

    print(f"\n{'='*80}")
    print(f"FAST DIAGNOSTIC EXPERIMENT (seed={random_state})")
    print(f"{'='*80}\n")

    # Smaller, faster parameters
    n_samples = 300
    n_features = 5
    n_time_bins = 8
    probe_depth = 2
    batch_size = 15
    artificial_censor_rate = 0.5
    n_ensemble = 3  # Reduced from 5
    epochs = 30  # Reduced from 50

    # Generate data
    data = generate_survival_data(
        n_samples=n_samples,
        n_features=n_features,
        n_time_bins=n_time_bins,
        censoring_rate=0.3,
        random_state=random_state
    )

    train_data, test_data = split_data(data, train_size=0.7, random_state=random_state)
    print(f"Train: {len(train_data['X'])}, Test: {len(test_data['X'])}")

    # Ground truth for oracle
    true_train_time = train_data['time'].copy()
    true_train_event = train_data['event'].copy()

    # Artificially censor
    artificial_time, artificial_event = artificially_censor(
        train_data['time'],
        train_data['event'],
        proportion=artificial_censor_rate,
        random_state=random_state
    )

    print(f"Artificially censored: {(artificial_event == 0).sum()}/{len(artificial_time)}")

    # Train initial model
    print("\nTraining initial model...")
    initial_model = BayesianSurvivalModel(
        n_features=n_features,
        n_time_bins=n_time_bins,
        n_ensemble=n_ensemble,
        hidden_size=32
    )
    initial_model.fit(
        train_data['X'],
        artificial_time,
        artificial_event,
        epochs=epochs,
        batch_size=32,
        verbose=False
    )

    # Evaluate
    test_preds = initial_model.predict_proba(test_data['X'])
    initial_c_index = concordance_index(test_preds, test_data['time'], test_data['event'])
    initial_brier = brier_score(test_preds, test_data['time'], test_data['event'])

    print(f"\nInitial: C-index={initial_c_index:.4f}, Brier={initial_brier:.4f}")

    # Oracle
    oracle = Oracle(true_train_time, true_train_event, probe_depth)

    # Train predictions
    train_preds = initial_model.predict_proba(train_data['X'])

    results = {'initial': {'c_index': initial_c_index, 'brier_score': initial_brier}}

    # Test each method
    methods = {
        'BatchBALD': SurvivalBatchBALD(probe_depth=probe_depth),
        'Entropy': EntropyAcquisition(),
        'Variance': VarianceAcquisition()
    }

    print(f"\n{'Method':<12} {'Selected':<10} {'Revealed':<10} {'C-index':<12} {'Δ C-idx':<10}")
    print("-" * 70)

    for method_name, acq_func in methods.items():
        # Select batch
        if method_name == 'BatchBALD':
            oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
                train_preds, artificial_time, artificial_event
            )
            selected_indices = acq_func.select_batch(
                oracle_probs, batch_size=batch_size, current_event=artificial_event,
                artificial_time=artificial_time, true_time=true_train_time,
                true_event=true_train_event
            )
        else:
            selected_indices = acq_func.select_batch(
                train_preds, batch_size=batch_size, current_event=artificial_event,
                artificial_time=artificial_time, true_time=true_train_time,
                true_event=true_train_event
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
            hidden_size=32
        )
        updated_model.fit(
            train_data['X'],
            updated_time,
            updated_event,
            epochs=epochs,
            batch_size=32,
            verbose=False
        )

        # Evaluate
        updated_test_preds = updated_model.predict_proba(test_data['X'])
        updated_c_index = concordance_index(
            updated_test_preds, test_data['time'], test_data['event']
        )
        updated_brier = brier_score(
            updated_test_preds, test_data['time'], test_data['event']
        )

        improvement = updated_c_index - initial_c_index

        print(f"{method_name:<12} {len(selected_indices):<10} {n_revealed:<10} "
              f"{updated_c_index:.4f}      {improvement:+.4f}")

        results[method_name] = {
            'c_index': updated_c_index,
            'brier_score': updated_brier,
            'improvement': improvement,
            'n_revealed': n_revealed
        }

    return results


if __name__ == '__main__':
    print("\nRunning 3 fast experiments...\n")

    all_results = []
    for seed in [42, 43, 44]:
        results = run_fast_experiment(seed)
        all_results.append(results)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS 3 RUNS")
    print(f"{'='*80}\n")

    methods = ['BatchBALD', 'Entropy', 'Variance']

    for method in methods:
        improvements = [r[method]['improvement'] for r in all_results]
        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)

        wins = sum(1 for imp in improvements if imp > 0)

        print(f"{method:<12}: Δ C-index = {mean_imp:+.4f} ± {std_imp:.4f}  (wins: {wins}/3)")

    # Determine winner
    best_method = max(methods, key=lambda m: np.mean([r[m]['improvement'] for r in all_results]))
    print(f"\n**Best method: {best_method}**")
