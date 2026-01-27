"""
Main entry point for running survival active learning experiments.

Run any experiment with configurable:
- Budget (n_samples, batch_size, n_runs)
- Oracle settings (probe_depth)
- Acquisition functions (batchbald, improved_batchbald, entropy, variance, weighted_batchbald)

Usage:
    python main.py --acquisition batchbald --probe_depth 5 --batch_size 50 --n_runs 10
    python main.py --acquisition entropy --n_samples 1000 --batch_size 30
    python main.py --acquisition all --n_runs 5
"""

import argparse
import sys
import os
import numpy as np
from scipy.stats import ttest_rel

from src.data.synthetic import generate_survival_data, artificially_censor, split_data
from src.models.survival_model import BayesianSurvivalModel
from src.oracle.oracle import Oracle
from src.acquisition.batchbald import SurvivalBatchBALD
from src.acquisition.improved_batchbald import ImprovedBatchBALD
from src.acquisition.weighted_batchbald import WeightedBatchBALD
from src.acquisition.cbald_true import TrueCBALD
from src.acquisition.entropy import EntropyAcquisition
from src.acquisition.variance import VarianceAcquisition
from src.evaluation.metrics import concordance_index, brier_score


def run_single_experiment(
    acquisition_name,
    acq_func,
    n_samples=500,
    n_features=10,
    n_time_bins=10,
    probe_depth=2,
    batch_size=20,
    artificial_censor_rate=0.5,
    n_ensemble=5,
    hidden_size=64,
    epochs=50,
    random_state=42,
    verbose=True
):
    """
    Run a single active learning experiment.

    Args:
        acquisition_name: Name of acquisition function
        acq_func: Acquisition function instance
        n_samples: Total number of samples
        n_features: Number of features
        n_time_bins: Number of time bins
        probe_depth: Oracle probe depth
        batch_size: Number of samples to query
        artificial_censor_rate: Proportion to artificially censor
        n_ensemble: Number of ensemble members
        hidden_size: Hidden layer size
        epochs: Training epochs
        random_state: Random seed
        verbose: Print progress

    Returns:
        results: Dictionary with experiment results
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running {acquisition_name} with seed {random_state}")
        print(f"Samples: {n_samples}, Features: {n_features}, Time bins: {n_time_bins}")
        print(f"Probe depth: {probe_depth}, Batch size: {batch_size}")
        print(f"{'='*80}\n")

    # Generate data
    data = generate_survival_data(
        n_samples=n_samples,
        n_features=n_features,
        n_time_bins=n_time_bins,
        censoring_rate=0.3,
        random_state=random_state
    )

    # Split into train and test
    train_data, test_data = split_data(data, train_size=0.7, random_state=random_state)

    # Store original train data (ground truth for oracle)
    true_train_time = train_data['time'].copy()
    true_train_event = train_data['event'].copy()

    # Artificially censor training data
    artificial_time, artificial_event = artificially_censor(
        train_data['time'],
        train_data['event'],
        proportion=artificial_censor_rate,
        random_state=random_state
    )

    # Train initial model on artificially censored data
    initial_model = BayesianSurvivalModel(
        n_features=n_features,
        n_time_bins=n_time_bins,
        n_ensemble=n_ensemble,
        hidden_size=hidden_size
    )
    initial_model.fit(
        train_data['X'],
        artificial_time,
        artificial_event,
        epochs=epochs,
        batch_size=32,
        verbose=False
    )

    # Evaluate initial model
    test_preds = initial_model.predict_proba(test_data['X'])
    initial_c_index = concordance_index(test_preds, test_data['time'], test_data['event'])
    initial_brier = brier_score(test_preds, test_data['time'], test_data['event'])

    # Create oracle
    oracle = Oracle(
        true_time=true_train_time,
        true_event=true_train_event,
        probe_depth=probe_depth
    )

    # Get predictions on training data for acquisition functions
    train_preds = initial_model.predict_proba(train_data['X'])

    # Select batch based on acquisition function type
    if acquisition_name in ['batchbald', 'improved_batchbald', 'weighted_batchbald']:
        # Get oracle outcome probabilities
        oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
            train_preds,
            artificial_time,
            artificial_event
        )

        # Different acquisition functions have different signatures
        if acquisition_name == 'improved_batchbald':
            selected_indices = acq_func.select_batch(
                oracle_probs,
                train_preds,
                artificial_time,
                batch_size=batch_size,
                current_event=artificial_event
            )
        else:
            selected_indices = acq_func.select_batch(
                oracle_probs,
                batch_size=batch_size,
                current_event=artificial_event
            )
    elif acquisition_name == 'cbald_true':
        # True C-BALD uses predictions directly
        selected_indices = acq_func.select_batch(
            train_preds,
            artificial_time,
            batch_size=batch_size,
            current_event=artificial_event
        )
    else:
        # Entropy and variance baselines
        selected_indices = acq_func.select_batch(
            train_preds,
            batch_size=batch_size,
            current_event=artificial_event
        )

    # Query oracle
    updated_time, updated_event = oracle.query(
        selected_indices,
        artificial_time,
        artificial_event
    )

    # Count oracle reveals
    n_revealed = ((updated_event - artificial_event) > 0).sum()
    n_extended = ((updated_time - artificial_time) > 0).sum()

    if verbose:
        print(f"Selected {len(selected_indices)} samples")
        print(f"Oracle revealed {n_revealed} deaths, extended {n_extended} censoring times")

    # Retrain model with updated data
    updated_model = BayesianSurvivalModel(
        n_features=n_features,
        n_time_bins=n_time_bins,
        n_ensemble=n_ensemble,
        hidden_size=hidden_size
    )
    updated_model.fit(
        train_data['X'],
        updated_time,
        updated_event,
        epochs=epochs,
        batch_size=32,
        verbose=False
    )

    # Evaluate updated model
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

    if verbose:
        print(f"Initial C-index: {initial_c_index:.4f}")
        print(f"Updated C-index: {updated_c_index:.4f} (Δ={updated_c_index-initial_c_index:+.4f})")

    return {
        'initial_c_index': initial_c_index,
        'initial_brier': initial_brier,
        'updated_c_index': updated_c_index,
        'updated_brier': updated_brier,
        'c_index_improvement': updated_c_index - initial_c_index,
        'brier_improvement': initial_brier - updated_brier,
        'n_revealed': n_revealed,
        'n_extended': n_extended,
        'n_selected': len(selected_indices)
    }


def run_experiments(
    acquisition_functions,
    n_runs=5,
    n_samples=500,
    n_features=10,
    n_time_bins=10,
    probe_depth=2,
    batch_size=20,
    artificial_censor_rate=0.5,
    n_ensemble=5,
    hidden_size=64,
    epochs=50,
    start_seed=42
):
    """
    Run multiple experiments with different random seeds.

    Args:
        acquisition_functions: Dict mapping names to acquisition function instances
        n_runs: Number of experimental runs
        **kwargs: Other experiment parameters

    Returns:
        all_results: Dictionary mapping method names to lists of results
    """
    all_results = {name: [] for name in acquisition_functions.keys()}

    for run in range(n_runs):
        print(f"\n{'#'*80}")
        print(f"# RUN {run + 1}/{n_runs}")
        print(f"{'#'*80}")

        random_state = start_seed + run

        for method_name, acq_func in acquisition_functions.items():
            result = run_single_experiment(
                acquisition_name=method_name,
                acq_func=acq_func,
                n_samples=n_samples,
                n_features=n_features,
                n_time_bins=n_time_bins,
                probe_depth=probe_depth,
                batch_size=batch_size,
                artificial_censor_rate=artificial_censor_rate,
                n_ensemble=n_ensemble,
                hidden_size=hidden_size,
                epochs=epochs,
                random_state=random_state,
                verbose=(run == 0)  # Only verbose on first run
            )
            all_results[method_name].append(result)

    return all_results


def print_summary(all_results):
    """Print summary statistics across runs."""
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS ACROSS RUNS")
    print(f"{'='*80}\n")

    print(f"{'Method':<25} {'C-index':<20} {'Δ C-index':<20} {'Wins':<10}")
    print("-" * 80)

    for method_name, results in all_results.items():
        c_indices = [r['updated_c_index'] for r in results]
        improvements = [r['c_index_improvement'] for r in results]

        c_idx_mean = np.mean(c_indices)
        c_idx_std = np.std(c_indices)
        imp_mean = np.mean(improvements)
        imp_std = np.std(improvements)
        wins = sum(1 for imp in improvements if imp > 0)

        print(f"{method_name:<25} {c_idx_mean:.4f} ± {c_idx_std:.4f}    "
              f"{imp_mean:+.4f} ± {imp_std:.4f}    {wins}/{len(results)}")

    # Determine best method
    if len(all_results) > 1:
        best_method = max(
            all_results.keys(),
            key=lambda m: np.mean([r['c_index_improvement'] for r in all_results[m]])
        )
        print(f"\n{'='*80}")
        print(f"Best method by mean C-index improvement: {best_method}")
        best_imp = np.mean([r['c_index_improvement'] for r in all_results[best_method]])
        best_std = np.std([r['c_index_improvement'] for r in all_results[best_method]])
        print(f"Improvement: {best_imp:+.4f} ± {best_std:.4f}")

        # Statistical significance tests if multiple methods
        if len(all_results) >= 2:
            print(f"\n{'='*80}")
            print("STATISTICAL TESTS (vs best method)")
            print(f"{'='*80}\n")

            best_improvements = [r['c_index_improvement'] for r in all_results[best_method]]

            for method_name in all_results.keys():
                if method_name == best_method:
                    continue

                method_improvements = [r['c_index_improvement'] for r in all_results[method_name]]

                if len(method_improvements) >= 3:  # Need at least 3 samples for t-test
                    t_stat, p_value = ttest_rel(best_improvements, method_improvements)
                    mean_diff = np.mean(best_improvements) - np.mean(method_improvements)

                    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"{best_method} vs {method_name}: "
                          f"Δ={mean_diff:+.4f}, p={p_value:.4f} {sig_marker}")

        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run survival active learning experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run BatchBALD with default settings
  python main.py --acquisition batchbald

  # Compare all acquisition functions
  python main.py --acquisition all --n_runs 10

  # Run with larger budget
  python main.py --acquisition improved_batchbald --n_samples 1000 --batch_size 50 --probe_depth 5

  # Quick test
  python main.py --acquisition entropy --n_runs 3 --n_samples 300 --epochs 30
        """
    )

    # Acquisition function selection
    parser.add_argument('--acquisition', type=str, default='all',
                        choices=['all', 'batchbald', 'improved_batchbald', 'weighted_batchbald',
                                'cbald_true', 'entropy', 'variance'],
                        help='Acquisition function to use (default: all)')

    # Experiment settings
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Number of experimental runs with different seeds (default: 5)')
    parser.add_argument('--start_seed', type=int, default=42,
                        help='Starting random seed (default: 42)')

    # Data settings
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of samples (default: 500)')
    parser.add_argument('--n_features', type=int, default=10,
                        help='Number of features (default: 10)')
    parser.add_argument('--n_time_bins', type=int, default=10,
                        help='Number of time bins (default: 10)')
    parser.add_argument('--artificial_censor_rate', type=float, default=0.5,
                        help='Artificial censoring rate (default: 0.5)')

    # Oracle settings
    parser.add_argument('--probe_depth', type=int, default=2,
                        help='Oracle probe depth (default: 2)')

    # Active learning settings
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Query batch size (default: 20)')

    # Model settings
    parser.add_argument('--n_ensemble', type=int, default=5,
                        help='Number of ensemble members (default: 5)')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden layer size (default: 64)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (default: 50)')

    args = parser.parse_args()

    # Print configuration
    print("\n" + "="*80)
    print("SURVIVAL ACTIVE LEARNING EXPERIMENT")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Acquisition:   {args.acquisition}")
    print(f"  Runs:          {args.n_runs}")
    print(f"  Samples:       {args.n_samples}")
    print(f"  Features:      {args.n_features}")
    print(f"  Time bins:     {args.n_time_bins}")
    print(f"  Probe depth:   {args.probe_depth}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Ensemble:      {args.n_ensemble}")
    print(f"  Hidden size:   {args.hidden_size}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Censor rate:   {args.artificial_censor_rate}")
    print("="*80 + "\n")

    # Create acquisition functions
    acquisition_functions = {}

    if args.acquisition == 'all':
        acquisition_functions = {
            'batchbald': SurvivalBatchBALD(probe_depth=args.probe_depth),
            'improved_batchbald': ImprovedBatchBALD(probe_depth=args.probe_depth),
            'weighted_batchbald': WeightedBatchBALD(probe_depth=args.probe_depth),
            'cbald_true': TrueCBALD(probe_depth=args.probe_depth),
            'entropy': EntropyAcquisition(),
            'variance': VarianceAcquisition()
        }
    else:
        if args.acquisition == 'batchbald':
            acquisition_functions['batchbald'] = SurvivalBatchBALD(probe_depth=args.probe_depth)
        elif args.acquisition == 'improved_batchbald':
            acquisition_functions['improved_batchbald'] = ImprovedBatchBALD(probe_depth=args.probe_depth)
        elif args.acquisition == 'weighted_batchbald':
            acquisition_functions['weighted_batchbald'] = WeightedBatchBALD(probe_depth=args.probe_depth)
        elif args.acquisition == 'cbald_true':
            acquisition_functions['cbald_true'] = TrueCBALD(probe_depth=args.probe_depth)
        elif args.acquisition == 'entropy':
            acquisition_functions['entropy'] = EntropyAcquisition()
        elif args.acquisition == 'variance':
            acquisition_functions['variance'] = VarianceAcquisition()

    # Run experiments
    all_results = run_experiments(
        acquisition_functions=acquisition_functions,
        n_runs=args.n_runs,
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_time_bins=args.n_time_bins,
        probe_depth=args.probe_depth,
        batch_size=args.batch_size,
        artificial_censor_rate=args.artificial_censor_rate,
        n_ensemble=args.n_ensemble,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        start_seed=args.start_seed
    )

    # Print summary
    print_summary(all_results)

    print("\nExperiments completed!")


if __name__ == '__main__':
    main()
