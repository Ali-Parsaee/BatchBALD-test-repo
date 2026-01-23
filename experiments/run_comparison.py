"""
Experimental comparison of acquisition functions for survival active learning.

Compares:
- SurvivalBatchBALD (proposed method)
- Entropy baseline
- Variance baseline
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


def run_single_experiment(
    n_samples: int = 500,
    n_features: int = 10,
    n_time_bins: int = 10,
    probe_depth: int = 2,
    batch_size: int = 20,
    artificial_censor_rate: float = 0.5,
    n_ensemble: int = 5,
    random_state: int = 42
):
    """
    Run a single active learning experiment.

    Args:
        n_samples: Total number of samples
        n_features: Number of features
        n_time_bins: Number of time bins
        probe_depth: Oracle probe depth
        batch_size: Number of samples to query
        artificial_censor_rate: Proportion to artificially censor
        n_ensemble: Number of ensemble members
        random_state: Random seed

    Returns:
        results: Dictionary with results for each method
    """
    print(f"\n{'='*80}")
    print(f"Running experiment with seed {random_state}")
    print(f"Samples: {n_samples}, Features: {n_features}, Time bins: {n_time_bins}")
    print(f"Probe depth: {probe_depth}, Batch size: {batch_size}")
    print(f"Artificial censoring rate: {artificial_censor_rate}")
    print(f"{'='*80}\n")

    # Generate data
    print("Generating synthetic survival data...")
    data = generate_survival_data(
        n_samples=n_samples,
        n_features=n_features,
        n_time_bins=n_time_bins,
        censoring_rate=0.3,
        random_state=random_state
    )

    # Split into train and test
    train_data, test_data = split_data(data, train_size=0.7, random_state=random_state)
    print(f"Train size: {len(train_data['X'])}, Test size: {len(test_data['X'])}")

    # Store original train data (ground truth for oracle)
    true_train_time = train_data['time'].copy()
    true_train_event = train_data['event'].copy()

    # Artificially censor training data
    print(f"\nArtificially censoring {artificial_censor_rate*100}% of training data...")
    artificial_time, artificial_event = artificially_censor(
        train_data['time'],
        train_data['event'],
        proportion=artificial_censor_rate,
        random_state=random_state
    )

    n_artificially_censored = (artificial_event == 0).sum()
    print(f"Artificially censored: {n_artificially_censored}/{len(artificial_time)}")

    # Train initial model on artificially censored data
    print("\nTraining initial Bayesian model...")
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
    print("\nEvaluating initial model on test set...")
    test_preds = initial_model.predict_proba(test_data['X'])
    initial_c_index = concordance_index(test_preds, test_data['time'], test_data['event'])
    initial_brier = brier_score(test_preds, test_data['time'], test_data['event'])
    print(f"Initial C-index: {initial_c_index:.4f}, Brier score: {initial_brier:.4f}")

    # Create oracle
    oracle = Oracle(
        true_time=true_train_time,
        true_event=true_train_event,
        probe_depth=probe_depth
    )

    # Get predictions on training data for acquisition functions
    print("\nComputing predictions on training data...")
    train_preds = initial_model.predict_proba(train_data['X'])  # (K, N, T)

    results = {
        'initial': {
            'c_index': initial_c_index,
            'brier_score': initial_brier
        }
    }

    # Test each acquisition function
    acquisition_functions = {
        'BatchBALD': SurvivalBatchBALD(probe_depth=probe_depth),
        'Entropy': EntropyAcquisition(),
        'Variance': VarianceAcquisition()
    }

    for method_name, acq_func in acquisition_functions.items():
        print(f"\n{'-'*80}")
        print(f"Testing {method_name}")
        print(f"{'-'*80}")

        # Select batch
        if method_name == 'BatchBALD':
            # Get oracle outcome probabilities
            oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
                train_preds,
                artificial_time,
                artificial_event
            )  # (K, N, k+1)

            selected_indices = acq_func.select_batch(
                oracle_probs,
                batch_size=batch_size,
                current_event=artificial_event
            )
        else:
            selected_indices = acq_func.select_batch(
                train_preds,
                batch_size=batch_size,
                current_event=artificial_event
            )

        print(f"Selected {len(selected_indices)} samples")

        if len(selected_indices) == 0:
            print("No samples selected!")
            results[method_name] = results['initial'].copy()
            continue

        # Query oracle
        print("Querying oracle...")
        updated_time, updated_event = oracle.query(
            selected_indices,
            artificial_time,
            artificial_event
        )

        # Check how many were revealed
        n_revealed = ((updated_event - artificial_event) > 0).sum()
        n_extended = ((updated_time - artificial_time) > 0).sum()
        print(f"Oracle revealed {n_revealed} deaths, extended {n_extended} censoring times")

        # Retrain model with updated data
        print("Retraining model with oracle feedback...")
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

        # Evaluate updated model
        print("Evaluating updated model on test set...")
        updated_test_preds = updated_model.predict_proba(test_data['X'])
        updated_c_index = concordance_index(
            updated_test_preds,
            test_data['time'],
            test_data['event']
        )
        updated_brier = brier_score(
            updated_test_preds,
            test_data['time'],
            test_data['event']
        )

        print(f"Updated C-index: {updated_c_index:.4f} (Δ={updated_c_index-initial_c_index:+.4f})")
        print(f"Updated Brier: {updated_brier:.4f} (Δ={updated_brier-initial_brier:+.4f})")

        results[method_name] = {
            'c_index': updated_c_index,
            'brier_score': updated_brier,
            'c_index_improvement': updated_c_index - initial_c_index,
            'brier_improvement': initial_brier - updated_brier,  # Lower is better
            'n_revealed': n_revealed,
            'n_extended': n_extended
        }

    return results


def run_multiple_experiments(
    n_runs: int = 5,
    **kwargs
):
    """
    Run multiple experiments with different random seeds.

    Args:
        n_runs: Number of experimental runs
        **kwargs: Arguments for run_single_experiment

    Returns:
        all_results: List of results from each run
        summary: Aggregated statistics
    """
    all_results = []

    for run in range(n_runs):
        print(f"\n{'#'*80}")
        print(f"# RUN {run + 1}/{n_runs}")
        print(f"{'#'*80}")

        results = run_single_experiment(
            random_state=42 + run,
            **kwargs
        )
        all_results.append(results)

    # Compute summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS ACROSS RUNS")
    print(f"{'='*80}\n")

    methods = ['initial', 'BatchBALD', 'Entropy', 'Variance']

    summary = {}
    for method in methods:
        c_indices = [r[method]['c_index'] for r in all_results]
        brier_scores = [r[method]['brier_score'] for r in all_results]

        summary[method] = {
            'c_index_mean': np.mean(c_indices),
            'c_index_std': np.std(c_indices),
            'brier_mean': np.mean(brier_scores),
            'brier_std': np.std(brier_scores)
        }

        if method != 'initial':
            improvements = [r[method].get('c_index_improvement', 0) for r in all_results]
            summary[method]['c_index_improvement_mean'] = np.mean(improvements)
            summary[method]['c_index_improvement_std'] = np.std(improvements)

            brier_improvements = [r[method].get('brier_improvement', 0) for r in all_results]
            summary[method]['brier_improvement_mean'] = np.mean(brier_improvements)
            summary[method]['brier_improvement_std'] = np.std(brier_improvements)

    # Print results
    print(f"{'Method':<15} {'C-index':<20} {'Brier':<20} {'C-idx Δ':<15}")
    print("-" * 75)

    for method in methods:
        c_idx_str = f"{summary[method]['c_index_mean']:.4f} ± {summary[method]['c_index_std']:.4f}"
        brier_str = f"{summary[method]['brier_mean']:.4f} ± {summary[method]['brier_std']:.4f}"

        if method == 'initial':
            print(f"{method:<15} {c_idx_str:<20} {brier_str:<20} {'-':<15}")
        else:
            c_imp_str = f"{summary[method]['c_index_improvement_mean']:+.4f}"
            print(f"{method:<15} {c_idx_str:<20} {brier_str:<20} {c_imp_str:<15}")

    # Determine winner
    print(f"\n{'='*80}")
    acq_methods = ['BatchBALD', 'Entropy', 'Variance']
    best_method = max(
        acq_methods,
        key=lambda m: summary[m]['c_index_improvement_mean']
    )
    print(f"Best method by C-index improvement: {best_method}")
    print(f"Improvement: {summary[best_method]['c_index_improvement_mean']:+.4f} ± {summary[best_method]['c_index_improvement_std']:.4f}")
    print(f"{'='*80}\n")

    return all_results, summary


if __name__ == '__main__':
    # Run experiments
    all_results, summary = run_multiple_experiments(
        n_runs=5,
        n_samples=500,
        n_features=10,
        n_time_bins=10,
        probe_depth=2,
        batch_size=20,
        artificial_censor_rate=0.5,
        n_ensemble=5
    )

    print("\nExperiments completed!")
