"""
Quick significance test: Run ultra-aggressive with 30 runs.
Prints intermediate results every 5 runs.
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
    """Aggressive filtering strategy."""

    def __init__(self, probe_depth, p_death_threshold=0.3, obs_mass_threshold=0.2):
        super().__init__(probe_depth, variant='filtered')
        self.p_death_threshold = p_death_threshold
        self.obs_mass_threshold = obs_mass_threshold

    def compute_scores(self, oracle_probs, predictions, current_time, current_event=None):
        mi = self.compute_mutual_information(oracle_probs)
        p_death = self.compute_p_death_window(oracle_probs)
        obs_mass = self.compute_observable_mass(predictions, current_time, current_event)

        scores = mi.copy()
        scores[p_death < self.p_death_threshold] = -np.inf
        scores[obs_mass < self.obs_mass_threshold] = -np.inf

        if current_event is not None:
            scores[current_event == 1] = -np.inf

        return scores


def run_experiment(method_name, acq_func, probe_depth, batch_size, random_state):
    """Run single experiment."""

    data = generate_survival_data(
        n_samples=1000, n_features=10, n_time_bins=15,
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

    # Oracle and predictions
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
    print("\n" + "="*100)
    print("QUICK SIGNIFICANCE TEST: 30 Runs for Statistical Power")
    print("="*100 + "\n")

    n_runs = 30
    probe_depth = 5
    batch_size = 50

    methods = {
        'Entropy': EntropyAcquisition(),
        'Aggressive Filtered (0.3, 0.2)': AggressiveFilteredBatchBALD(
            probe_depth, p_death_threshold=0.3, obs_mass_threshold=0.2
        ),
        'Ultra-Aggressive (0.4, 0.3)': AggressiveFilteredBatchBALD(
            probe_depth, p_death_threshold=0.4, obs_mass_threshold=0.3
        ),
    }

    all_results = {name: [] for name in methods.keys()}

    for run in range(n_runs):
        print(f"\rRun {run+1}/{n_runs}...", end="", flush=True)

        for method_name, acq_func in methods.items():
            result = run_experiment(
                method_name, acq_func, probe_depth, batch_size, random_state=42 + run
            )
            all_results[method_name].append(result)

        # Print intermediate summary every 5 runs
        if (run + 1) % 5 == 0:
            print(f"\n\n--- INTERMEDIATE RESULTS (after {run+1} runs) ---")
            for method_name in methods.keys():
                improvements = [r['improvement'] for r in all_results[method_name]]
                mean_imp = np.mean(improvements)
                std_imp = np.std(improvements)
                wins = sum(1 for imp in improvements if imp > 0)

                print(f"  {method_name:35s}: Δ={mean_imp:+.4f}±{std_imp:.4f}, Wins={wins}/{run+1}")

            # Quick p-value check
            ent_imps = [r['improvement'] for r in all_results['Entropy']]
            aggr_imps = [r['improvement'] for r in all_results['Aggressive Filtered (0.3, 0.2)']]
            ultra_imps = [r['improvement'] for r in all_results['Ultra-Aggressive (0.4, 0.3)']]

            _, p_aggr = ttest_rel(aggr_imps, ent_imps)
            _, p_ultra = ttest_rel(ultra_imps, ent_imps)

            print(f"\n  p-values vs Entropy:")
            print(f"    Aggressive Filtered: p={p_aggr:.4f} {'✓ SIGNIFICANT!' if p_aggr < 0.05 else ''}")
            print(f"    Ultra-Aggressive:    p={p_ultra:.4f} {'✓ SIGNIFICANT!' if p_ultra < 0.05 else ''}")
            print()

    # Final results
    print(f"\n\n{'='*100}")
    print("FINAL RESULTS (30 runs)")
    print(f"{'='*100}\n")

    summary = {}
    for method_name in methods.keys():
        improvements = [r['improvement'] for r in all_results[method_name]]
        n_reveals = [r['n_revealed'] for r in all_results[method_name]]

        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        wins = sum(1 for imp in improvements if imp > 0)
        avg_reveals = np.mean(n_reveals)

        summary[method_name] = {
            'improvements': improvements,
            'mean': mean_imp,
            'std': std_imp,
            'wins': wins,
            'avg_reveals': avg_reveals
        }

    print(f"{'Method':<40} {'Mean Δ':<15} {'Std':<10} {'Wins':<10} {'Avg Reveals':<12}")
    print("-"*100)

    for method_name in methods.keys():
        s = summary[method_name]
        marker = "🏆" if s['mean'] > 0.010 else ("★" if s['mean'] > 0.005 else " ")
        print(f"{marker} {method_name:<38} {s['mean']:+.4f}         {s['std']:.4f}    "
              f"{s['wins']}/{n_runs}     {s['avg_reveals']:.1f}")

    # Statistical tests
    print(f"\n{'='*100}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*100}\n")

    ent_imps = summary['Entropy']['improvements']

    for method_name in ['Aggressive Filtered (0.3, 0.2)', 'Ultra-Aggressive (0.4, 0.3)']:
        method_imps = summary[method_name]['improvements']
        t_stat, p_value = ttest_rel(method_imps, ent_imps)

        mean_diff = summary[method_name]['mean'] - summary['Entropy']['mean']
        pct_better = (mean_diff / abs(summary['Entropy']['mean'])) * 100 if summary['Entropy']['mean'] != 0 else 0

        print(f"{method_name}:")
        print(f"  Mean improvement: {summary[method_name]['mean']:+.4f}")
        print(f"  vs Entropy:       {mean_diff:+.4f} ({pct_better:+.1f}% {'better' if mean_diff > 0 else 'worse'})")
        print(f"  t-statistic:      {t_stat:.3f}")
        print(f"  p-value:          {p_value:.4f}")

        if p_value < 0.05:
            if mean_diff > 0:
                print(f"  🎉 STATISTICALLY SIGNIFICANT! Beats Entropy at p < 0.05!")
            else:
                print(f"  ✗ Significantly worse than Entropy")
        else:
            print(f"  ⚠️  Not statistically significant (p ≥ 0.05)")
        print()

    # Final verdict
    print(f"{'='*100}")
    best_name = max(summary.keys(), key=lambda k: summary[k]['mean'])
    best_vs_ent = summary[best_name]['mean'] - summary['Entropy']['mean']

    if best_name != 'Entropy':
        best_imps = summary[best_name]['improvements']
        _, p_best = ttest_rel(best_imps, ent_imps)

        if p_best < 0.05 and best_vs_ent > 0:
            print(f"✅ SUCCESS! {best_name} SIGNIFICANTLY BEATS Entropy!")
            print(f"   Improvement: {best_vs_ent:+.4f} (p={p_best:.4f})")
        elif best_vs_ent > 0:
            print(f"⚠️  {best_name} beats Entropy but NOT significant")
            print(f"   Improvement: {best_vs_ent:+.4f} (p={p_best:.4f})")
            print(f"   Recommendation: Run more experiments or try larger datasets")
        else:
            print(f"❌ Entropy still wins")
    else:
        print(f"❌ Entropy wins overall")

    print(f"{'='*100}\n")

    return summary


if __name__ == '__main__':
    summary = main()
