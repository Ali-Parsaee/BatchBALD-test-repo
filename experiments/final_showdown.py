"""
FINAL SHOWDOWN: Best strategies with 10 runs, multiple settings.

Aggressive Filtered is our best candidate: +0.0061 vs Entropy's +0.0046
Let's test it more thoroughly!
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
    Best strategy so far: Filter aggressively, then use MI.

    Only consider samples with:
    - P(death in window) > threshold
    - Observable mass > threshold
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


def run_single_config(
    method_name,
    acq_func,
    n_samples,
    probe_depth,
    batch_size,
    n_runs=10
):
    """Run experiments for a single configuration."""

    all_results = []

    for run in range(n_runs):
        # Generate data
        data = generate_survival_data(
            n_samples=n_samples, n_features=10, n_time_bins=15,
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

        all_results.append({
            'improvement': updated_c_index - initial_c_index,
            'n_revealed': n_revealed
        })

    return all_results


def main():
    """Final comprehensive test."""

    print("\n" + "="*100)
    print("FINAL SHOWDOWN: Aggressive Filtered vs Entropy")
    print("="*100 + "\n")

    # Test multiple configurations
    configs = [
        # (n_samples, probe_depth, batch_size, name)
        (1000, 5, 50, "Standard"),
        (1000, 8, 50, "Large probe"),
        (1000, 5, 80, "Large batch"),
        (1500, 5, 75, "Large dataset"),
    ]

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
            'Aggressive Filtered (0.2, 0.15)': AggressiveFilteredBatchBALD(
                probe_depth, p_death_threshold=0.2, obs_mass_threshold=0.15
            ),
            'Ultra-Aggressive (0.4, 0.3)': AggressiveFilteredBatchBALD(
                probe_depth, p_death_threshold=0.4, obs_mass_threshold=0.3
            ),
        }

        config_results = {}

        for method_name, acq_func in methods.items():
            print(f"  {method_name}...", end=" ", flush=True)

            results = run_single_config(
                method_name, acq_func, n_samples, probe_depth, batch_size, n_runs=10
            )

            improvements = [r['improvement'] for r in results]
            n_reveals = [r['n_revealed'] for r in results]

            mean_imp = np.mean(improvements)
            std_imp = np.std(improvements)
            wins = sum(1 for imp in improvements if imp > 0)
            avg_reveals = np.mean(n_reveals)

            config_results[method_name] = {
                'improvements': improvements,
                'mean': mean_imp,
                'std': std_imp,
                'wins': wins,
                'avg_reveals': avg_reveals
            }

            print(f"Δ={mean_imp:+.4f}±{std_imp:.4f}, Wins={wins}/10, Reveals={avg_reveals:.1f}")

        # Statistical tests
        print(f"\n  Statistical tests:")
        ent_imps = config_results['Entropy']['improvements']

        for method_name in methods.keys():
            if method_name == 'Entropy':
                continue

            method_imps = config_results[method_name]['improvements']
            t_stat, p_value = ttest_rel(method_imps, ent_imps)

            mean_diff = config_results[method_name]['mean'] - config_results['Entropy']['mean']

            marker = "🎉" if p_value < 0.05 and mean_diff > 0 else ("★" if mean_diff > 0 else " ")

            print(f"  {marker} {method_name:40s} vs Entropy: Δ={mean_diff:+.4f}, p={p_value:.4f}")

        all_config_results.append({
            'config': (n_samples, probe_depth, batch_size, config_name),
            'results': config_results
        })

    # Overall summary
    print(f"\n\n{'='*100}")
    print("OVERALL SUMMARY")
    print(f"{'='*100}\n")

    print(f"{'Config':<20} {'Entropy Δ':<15} {'AggressFilt Δ':<15} {'Winner':<15} {'Significant?':<15}")
    print("-"*90)

    n_wins_entropy = 0
    n_wins_aggr = 0
    n_significant = 0

    for config_data in all_config_results:
        _, _, _, config_name = config_data['config']
        results = config_data['results']

        ent_mean = results['Entropy']['mean']
        aggr_mean = results['Aggressive Filtered (0.3, 0.2)']['mean']

        ent_imps = results['Entropy']['improvements']
        aggr_imps = results['Aggressive Filtered (0.3, 0.2)']['improvements']

        t_stat, p_value = ttest_rel(aggr_imps, ent_imps)

        winner = "AggressFilt" if aggr_mean > ent_mean else "Entropy"
        significant = "YES ✓" if p_value < 0.05 else "NO"

        if winner == "AggressFilt":
            n_wins_aggr += 1
            if p_value < 0.05:
                n_significant += 1
        else:
            n_wins_entropy += 1

        print(f"{config_name:<20} {ent_mean:+.4f}         {aggr_mean:+.4f}         {winner:<15} {significant:<15}")

    print(f"\n{'='*100}")
    print(f"Aggressive Filtered wins: {n_wins_aggr}/{len(configs)}")
    print(f"Statistically significant: {n_significant}/{len(configs)}")

    if n_wins_aggr >= len(configs) / 2 and n_significant > 0:
        print(f"\n🎉 SUCCESS! Aggressive Filtered BatchBALD BEATS Entropy!")
    elif n_wins_aggr >= len(configs) / 2:
        print(f"\n⚠️  Aggressive Filtered wins most configs but not statistically significant")
    else:
        print(f"\n❌ Entropy still wins overall")

    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()
