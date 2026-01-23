"""
FINAL SHOWDOWN (Optimized): Test best strategies across multiple configs.

Optimized to complete all 4 configs with 10 runs each in reasonable time.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.synthetic import generate_survival_data, artificially_censor, split_data
from src.models.survival_model import BayesianSurvivalModel
from src.oracle.oracle import Oracle
from src.acquisition.improved_batchbald import ImprovedBatchBALD
from src.acquisition.entropy import EntropyAcquisition
from src.evaluation.metrics import concordance_index
from scipy.stats import ttest_rel


class AggressiveFilteredBatchBALD(ImprovedBatchBALD):
    """
    Best strategy: Filter aggressively, then use MI.
    """

    def __init__(self, probe_depth, p_death_threshold=0.3, obs_mass_threshold=0.2):
        super().__init__(probe_depth, variant='filtered')
        self.p_death_threshold = p_death_threshold
        self.obs_mass_threshold = obs_mass_threshold

    def compute_scores(self, oracle_probs, predictions, current_time, current_event=None):
        mi = self.compute_mutual_information(oracle_probs)
        p_death = self.compute_p_death_window(oracle_probs)
        obs_mass = self.compute_observable_mass(predictions, current_time, current_event)

        scores = mi.copy()

        # Aggressive filtering
        scores[p_death < self.p_death_threshold] = -np.inf
        scores[obs_mass < self.obs_mass_threshold] = -np.inf

        if current_event is not None:
            scores[current_event == 1] = -np.inf

        return scores


def run_single_experiment(
    method_name,
    acq_func,
    n_samples,
    probe_depth,
    batch_size,
    random_state
):
    """Run single experiment."""

    # Generate data
    data = generate_survival_data(
        n_samples=n_samples, n_features=10, n_time_bins=15,
        censoring_rate=0.3, random_state=random_state
    )

    train_data, test_data = split_data(data, train_size=0.7, random_state=random_state)
    true_train_time = train_data['time'].copy()
    true_train_event = train_data['event'].copy()

    artificial_time, artificial_event = artificially_censor(
        train_data['time'], train_data['event'],
        proportion=0.5, random_state=random_state
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
    oracle = Oracle(true_train_time, true_train_event, probe_depth)
    train_preds = initial_model.predict_proba(train_data['X'])
    oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
        train_preds, artificial_time, artificial_event
    )

    # Select batch
    if method_name == 'Entropy':
        selected = acq_func.select_batch(train_preds, batch_size, artificial_event)
    else:
        selected = acq_func.select_batch(
            oracle_probs, train_preds, artificial_time,
            batch_size, artificial_event, greedy=False
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
        'improvement': updated_c_index - initial_c_index,
        'n_revealed': n_revealed
    }


def main():
    """Final comprehensive test."""

    print("\n" + "="*100)
    print("FINAL SHOWDOWN (OPTIMIZED): Aggressive Filtered vs Entropy")
    print("="*100 + "\n")

    # Test multiple configurations
    configs = [
        # (n_samples, probe_depth, batch_size, name)
        (1000, 5, 50, "Standard"),
        (1000, 8, 50, "Large probe"),
        (1000, 5, 80, "Large batch"),
        (1500, 5, 75, "Large dataset"),
    ]

    n_runs = 10
    all_config_results = []

    for config_idx, (n_samples, probe_depth, batch_size, config_name) in enumerate(configs):
        print(f"\n{'#'*100}")
        print(f"CONFIG {config_idx+1}/{len(configs)}: {config_name}")
        print(f"N={n_samples}, Probe={probe_depth}, Batch={batch_size}")
        print(f"{'#'*100}\n")

        methods = {
            'Entropy': EntropyAcquisition(),
            'Aggressive Filtered (0.3, 0.2)': AggressiveFilteredBatchBALD(
                probe_depth, p_death_threshold=0.3, obs_mass_threshold=0.2
            ),
            'Ultra-Aggressive (0.4, 0.3)': AggressiveFilteredBatchBALD(
                probe_depth, p_death_threshold=0.4, obs_mass_threshold=0.3
            ),
        }

        config_results = {}

        # Run experiments
        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}...", end=" ", flush=True)

            for method_name, acq_func in methods.items():
                result = run_single_experiment(
                    method_name, acq_func, n_samples, probe_depth, batch_size,
                    random_state=42 + run
                )

                if method_name not in config_results:
                    config_results[method_name] = []
                config_results[method_name].append(result)

            print("Done")

        # Summarize results for this config
        print(f"\n  Results:")
        for method_name in methods.keys():
            improvements = [r['improvement'] for r in config_results[method_name]]
            n_reveals = [r['n_revealed'] for r in config_results[method_name]]

            mean_imp = np.mean(improvements)
            std_imp = np.std(improvements)
            wins = sum(1 for imp in improvements if imp > 0)
            avg_reveals = np.mean(n_reveals)

            print(f"    {method_name:35s}: Δ={mean_imp:+.4f}±{std_imp:.4f}, Wins={wins}/{n_runs}, Reveals={avg_reveals:.1f}")

        # Statistical tests
        print(f"\n  Statistical tests:")
        ent_imps = [r['improvement'] for r in config_results['Entropy']]

        for method_name in methods.keys():
            if method_name == 'Entropy':
                continue

            method_imps = [r['improvement'] for r in config_results[method_name]]
            t_stat, p_value = ttest_rel(method_imps, ent_imps)

            mean_ent = np.mean(ent_imps)
            mean_method = np.mean(method_imps)
            mean_diff = mean_method - mean_ent

            marker = "🎉" if p_value < 0.05 and mean_diff > 0 else ("★" if mean_diff > 0 else " ")

            print(f"    {marker} {method_name:40s} vs Entropy: Δ={mean_diff:+.4f}, p={p_value:.4f}")

        # Store results
        all_config_results.append({
            'config': (n_samples, probe_depth, batch_size, config_name),
            'results': config_results
        })

    # Overall summary
    print(f"\n\n{'='*100}")
    print("OVERALL SUMMARY")
    print(f"{'='*100}\n")

    print(f"{'Config':<20} {'Entropy Δ':<15} {'AggressFilt Δ':<15} {'Ultra Δ':<15} {'Winner':<15} {'Significant?':<15}")
    print("-"*100)

    n_wins_entropy = 0
    n_wins_aggr = 0
    n_wins_ultra = 0
    n_significant = 0

    for config_data in all_config_results:
        _, _, _, config_name = config_data['config']
        results = config_data['results']

        ent_mean = np.mean([r['improvement'] for r in results['Entropy']])
        aggr_mean = np.mean([r['improvement'] for r in results['Aggressive Filtered (0.3, 0.2)']])
        ultra_mean = np.mean([r['improvement'] for r in results['Ultra-Aggressive (0.4, 0.3)']])

        ent_imps = [r['improvement'] for r in results['Entropy']]
        aggr_imps = [r['improvement'] for r in results['Aggressive Filtered (0.3, 0.2)']]
        ultra_imps = [r['improvement'] for r in results['Ultra-Aggressive (0.4, 0.3)']]

        # Find winner
        best_mean = max(ent_mean, aggr_mean, ultra_mean)
        if best_mean == ent_mean:
            winner = "Entropy"
            n_wins_entropy += 1
            winner_imps = ent_imps
        elif best_mean == aggr_mean:
            winner = "AggressFilt"
            n_wins_aggr += 1
            winner_imps = aggr_imps
        else:
            winner = "Ultra"
            n_wins_ultra += 1
            winner_imps = ultra_imps

        # Test significance of winner vs Entropy
        if winner != "Entropy":
            _, p_value = ttest_rel(winner_imps, ent_imps)
            significant = "YES ✓" if p_value < 0.05 else "NO"
            if p_value < 0.05:
                n_significant += 1
        else:
            significant = "N/A"

        print(f"{config_name:<20} {ent_mean:+.4f}         {aggr_mean:+.4f}         {ultra_mean:+.4f}         {winner:<15} {significant:<15}")

    print(f"\n{'='*100}")
    print(f"Entropy wins:            {n_wins_entropy}/{len(configs)}")
    print(f"Aggressive Filtered wins: {n_wins_aggr}/{len(configs)}")
    print(f"Ultra-Aggressive wins:    {n_wins_ultra}/{len(configs)}")
    print(f"Statistically significant: {n_significant}/{len(configs)}")

    # Overall verdict
    if n_wins_aggr + n_wins_ultra > n_wins_entropy:
        print(f"\n🎉 SUCCESS! Aggressive strategies win {n_wins_aggr + n_wins_ultra}/{len(configs)} configs!")
        if n_significant > 0:
            print(f"   With {n_significant} statistically significant wins (p<0.05)")
        else:
            print(f"   ⚠️  But not statistically significant - need more runs")
    else:
        print(f"\n⚠️  Entropy still wins {n_wins_entropy}/{len(configs)} configs")

    print(f"{'='*100}\n")

    return all_config_results


if __name__ == '__main__':
    results = main()
