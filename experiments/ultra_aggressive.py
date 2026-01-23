"""
ULTRA-AGGRESSIVE BatchBALD improvements.

Current problem: Improvements help but Entropy still wins.
- Entropy: +0.0090
- Best BatchBALD variant (MI * P(death)): +0.0058

New strategy: Be MUCH more aggressive!
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


class UltraAggressiveBatchBALD(ImprovedBatchBALD):
    """
    Ultra-aggressive: Almost entirely use observable mass, just use MI for tie-breaking.

    Score = Observable_mass^2 * (1 + 0.1 * MI)

    This heavily weights samples with high probability in observable window.
    """

    def compute_scores(self, oracle_probs, predictions, current_time, current_event=None):
        mi = self.compute_mutual_information(oracle_probs)
        obs_mass = self.compute_observable_mass(predictions, current_time, current_event)

        # Normalize MI to [0, 1]
        mi_norm = (mi - mi.min()) / (mi.max() - mi.min() + 1e-8)

        # Ultra-aggressive: Square observable mass, add tiny bit of MI
        scores = (obs_mass ** 2) * (1 + 0.1 * mi_norm)

        if current_event is not None:
            scores[current_event == 1] = -np.inf

        return scores


class PureObservableMass(ImprovedBatchBALD):
    """
    Completely ignore MI - only use observable mass.
    """

    def compute_scores(self, oracle_probs, predictions, current_time, current_event=None):
        scores = self.compute_observable_mass(predictions, current_time, current_event)

        if current_event is not None:
            scores[current_event == 1] = -np.inf

        return scores


class HybridPDeathObsMass(ImprovedBatchBALD):
    """
    Hybrid: P(death) * Observable_mass * (1 + MI)

    Focuses on samples where:
    1. High P(death in window)
    2. High observable mass
    3. Some model uncertainty
    """

    def compute_scores(self, oracle_probs, predictions, current_time, current_event=None):
        mi = self.compute_mutual_information(oracle_probs)
        p_death = self.compute_p_death_window(oracle_probs)
        obs_mass = self.compute_observable_mass(predictions, current_time, current_event)

        # Normalize MI
        mi_norm = (mi - mi.min()) / (mi.max() - mi.min() + 1e-8)

        # Hybrid score
        scores = p_death * obs_mass * (1 + mi_norm)

        if current_event is not None:
            scores[current_event == 1] = -np.inf

        return scores


class AggressiveFilteredBatchBALD(ImprovedBatchBALD):
    """
    Aggressive filtering: Only consider samples with P(death) > 0.3 and obs_mass > 0.2
    Then use MI on filtered set.
    """

    def compute_scores(self, oracle_probs, predictions, current_time, current_event=None):
        mi = self.compute_mutual_information(oracle_probs)
        p_death = self.compute_p_death_window(oracle_probs)
        obs_mass = self.compute_observable_mass(predictions, current_time, current_event)

        # Start with MI
        scores = mi.copy()

        # Aggressive filtering
        scores[p_death < 0.3] = -np.inf  # Must have at least 30% chance of death
        scores[obs_mass < 0.2] = -np.inf  # Must have at least 20% mass in window

        if current_event is not None:
            scores[current_event == 1] = -np.inf

        return scores


def test_ultra_aggressive(n_runs=5):
    """Test ultra-aggressive strategies."""

    print("\n" + "="*100)
    print("ULTRA-AGGRESSIVE BATCH BALD STRATEGIES")
    print("="*100 + "\n")

    probe_depth = 5
    batch_size = 50

    methods = {
        'Entropy (baseline)': EntropyAcquisition(),
        'Pure Observable Mass': PureObservableMass(probe_depth, variant='observable_mass'),
        'Ultra-Aggressive (ObsMass^2)': UltraAggressiveBatchBALD(probe_depth, variant='observable_mass'),
        'Hybrid: P(death)*ObsMass*MI': HybridPDeathObsMass(probe_depth, variant='observable_mass'),
        'Aggressive Filtered': AggressiveFilteredBatchBALD(probe_depth, variant='observable_mass'),
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

        run_results = {}

        for method_name, acq_func in methods.items():
            print(f"  {method_name}...", end=" ", flush=True)

            # Select batch
            if method_name == 'Entropy (baseline)':
                selected = acq_func.select_batch(train_preds, batch_size, artificial_event)
            else:
                selected = acq_func.select_batch(
                    oracle_probs, train_preds, artificial_time,
                    batch_size, artificial_event, greedy=False  # Use simple top-k
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

            improvement = updated_c_index - initial_c_index
            run_results[method_name] = {
                'improvement': improvement,
                'n_revealed': n_revealed
            }

            print(f"Δ={improvement:+.4f}, Revealed={n_revealed}/50")

        all_results.append(run_results)

    # Summary
    print(f"\n\n{'='*100}")
    print("FINAL RESULTS")
    print(f"{'='*100}\n")

    print(f"{'Method':<40} {'Mean Δ':<12} {'Std':<10} {'Wins':<8} {'Avg Reveals':<12} {'p vs Ent':<10}")
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
            'improvements': improvements,
            'avg_reveals': avg_reveals
        }

        # Test vs Entropy
        if method_name != 'Entropy (baseline)':
            ent_imps = summary['Entropy (baseline)']['improvements']
            t_stat, p_value = ttest_rel(improvements, ent_imps)
        else:
            p_value = 1.0

        marker = "🏆" if mean_imp > 0.012 else ("★" if mean_imp > 0.008 else " ")
        print(f"{marker} {method_name:<38} {mean_imp:+.4f}      {std_imp:.4f}    {wins}/{n_runs}     "
              f"{avg_reveals:.1f}           {p_value:.4f}")

    # Best method
    print(f"\n{'='*100}")
    sorted_methods = sorted(summary.items(), key=lambda x: x[1]['mean'], reverse=True)
    best_name = sorted_methods[0][0]
    best_mean = summary[best_name]['mean']
    ent_mean = summary['Entropy (baseline)']['mean']

    print(f"BEST: {best_name}")
    print(f"  Mean Δ: {best_mean:+.4f}")
    print(f"  vs Entropy: {best_mean - ent_mean:+.4f} ({'BETTER' if best_mean > ent_mean else 'worse'})")

    if best_name != 'Entropy (baseline)':
        best_imps = summary[best_name]['improvements']
        ent_imps = summary['Entropy (baseline)']['improvements']
        t_stat, p_value = ttest_rel(best_imps, ent_imps)

        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05 and best_mean > ent_mean:
            print(f"\n  🎉 {best_name} SIGNIFICANTLY BEATS Entropy!")
        elif best_mean > ent_mean:
            print(f"\n  ⚠️  Beats Entropy but not significant (p={p_value:.4f})")
        else:
            print(f"\n  ❌ Entropy still wins")

    print("="*100 + "\n")

    return summary


if __name__ == '__main__':
    summary = test_ultra_aggressive(n_runs=5)
