"""
Systematic Strategy Search: Find what actually works in survival active learning.

Tests multiple strategies from literature and analyzes WHY they work or don't.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.synthetic import generate_survival_data, artificially_censor, split_data
from src.models.survival_model import BayesianSurvivalModel
from src.oracle.oracle import Oracle
from src.evaluation.metrics import concordance_index
from scipy.stats import ttest_rel
from scipy.spatial.distance import cdist


class StrategyTester:
    """Test different active learning strategies."""

    def __init__(self, probe_depth=5):
        self.probe_depth = probe_depth

    def compute_entropy(self, predictions):
        """Standard entropy (what baseline uses)."""
        mean_preds = predictions.mean(axis=0)
        entropy = -np.sum(mean_preds * np.log(mean_preds + 1e-10), axis=1)
        return entropy

    def compute_mutual_information(self, oracle_probs):
        """BatchBALD MI."""
        mean_probs = oracle_probs.mean(axis=0)
        H_expected = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

        H_conditional = -np.sum(oracle_probs * np.log(oracle_probs + 1e-10), axis=2)
        E_H_conditional = H_conditional.mean(axis=0)

        return H_expected - E_H_conditional

    def compute_observable_mass(self, predictions, current_time):
        """Probability in observable window [c+1, c+k]."""
        mean_preds = predictions.mean(axis=0)
        N, T = mean_preds.shape

        obs_mass = np.zeros(N)
        for i in range(N):
            c = current_time[i]
            max_obs = min(c + self.probe_depth, T - 1)
            if c + 1 <= max_obs:
                obs_mass[i] = mean_preds[i, c+1:max_obs+1].sum()

        return obs_mass

    def compute_p_death_window(self, oracle_probs):
        """P(death revealed in window)."""
        # Sum over death outcomes (exclude last bin which is extended censoring)
        p_death = oracle_probs[:, :, :-1].sum(axis=2).mean(axis=0)
        return p_death

    def compute_variance(self, predictions):
        """Variance across ensemble (measures disagreement)."""
        mean_preds = predictions.mean(axis=0)
        variance = ((predictions - mean_preds[np.newaxis, :, :]) ** 2).mean(axis=(0, 2))
        return variance

    def compute_expected_model_change(self, predictions, X):
        """Expected gradient magnitude (proxy for model change)."""
        # For each point, estimate how much model would change
        # Use prediction variance * feature magnitude as proxy
        variance = self.compute_variance(predictions)
        feature_magnitude = np.linalg.norm(X, axis=1)
        return variance * feature_magnitude

    def compute_density(self, X, k=10):
        """1 / average distance to k nearest neighbors."""
        distances = cdist(X, X)
        # For each point, find k+1 nearest (including itself)
        sorted_distances = np.sort(distances, axis=1)
        avg_dist = sorted_distances[:, 1:k+1].mean(axis=1)
        density = 1.0 / (avg_dist + 1e-10)
        return density

    def compute_diversity_penalty(self, X, already_selected):
        """Penalty based on distance to already selected points."""
        if len(already_selected) == 0:
            return np.zeros(len(X))

        selected_X = X[already_selected]
        distances = cdist(X, selected_X)
        min_distances = distances.min(axis=1)

        # Higher penalty for points close to already selected
        penalty = 1.0 / (min_distances + 1e-3)
        return penalty


def test_strategy(strategy_name, score_func, n_runs=5):
    """Test a single strategy across multiple runs."""

    probe_depth = 5
    batch_size = 50
    results = []

    for run in range(n_runs):
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
        model = BayesianSurvivalModel(
            n_features=10, n_time_bins=15, n_ensemble=5, hidden_size=64
        )
        model.fit(
            train_data['X'], artificial_time, artificial_event,
            epochs=40, batch_size=32, verbose=False
        )

        test_preds = model.predict_proba(test_data['X'])
        initial_c_index = concordance_index(test_preds, test_data['time'], test_data['event'])

        # Get predictions and oracle probs
        train_preds = model.predict_proba(train_data['X'])
        oracle = Oracle(true_train_time, true_train_event, probe_depth)
        oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
            train_preds, artificial_time, artificial_event
        )

        # Compute scores with strategy
        scores = score_func(
            train_preds, oracle_probs, artificial_time, artificial_event, train_data['X']
        )

        # Filter out already uncensored
        scores[artificial_event == 1] = -np.inf

        # Select top batch_size
        selected = np.argsort(scores)[-batch_size:]

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

        results.append({
            'improvement': updated_c_index - initial_c_index,
            'n_revealed': n_revealed,
            'selected_indices': selected
        })

    return results


def analyze_selections(strategy_name, results, all_data):
    """Analyze what points a strategy selects."""

    print(f"\n  Analysis of {strategy_name} selections:")

    # Aggregate selected points characteristics
    avg_obs_mass = []
    avg_p_death = []
    avg_entropy = []
    avg_censoring_time = []

    for run_idx, result in enumerate(results):
        selected = result['selected_indices']

        # Get data for this run
        data = all_data[run_idx]
        predictions = data['predictions']
        oracle_probs = data['oracle_probs']
        current_time = data['current_time']

        tester = StrategyTester(probe_depth=5)

        obs_mass = tester.compute_observable_mass(predictions, current_time)
        p_death = tester.compute_p_death_window(oracle_probs)
        entropy = tester.compute_entropy(predictions)

        avg_obs_mass.append(obs_mass[selected].mean())
        avg_p_death.append(p_death[selected].mean())
        avg_entropy.append(entropy[selected].mean())
        avg_censoring_time.append(current_time[selected].mean())

    print(f"    Avg observable mass:  {np.mean(avg_obs_mass):.4f}")
    print(f"    Avg P(death):         {np.mean(avg_p_death):.4f}")
    print(f"    Avg entropy:          {np.mean(avg_entropy):.4f}")
    print(f"    Avg censoring time:   {np.mean(avg_censoring_time):.2f}")


def main():
    """Systematic strategy search."""

    print("\n" + "="*100)
    print("SYSTEMATIC STRATEGY SEARCH: What Actually Works?")
    print("="*100 + "\n")

    tester = StrategyTester(probe_depth=5)

    # Define strategies to test
    strategies = {}

    # 1. BASELINE: Entropy (what we're trying to beat)
    strategies['Entropy (baseline)'] = lambda pred, oracle_p, time, event, X: tester.compute_entropy(pred)

    # 2. BATCHBALD: Mutual Information
    strategies['BatchBALD MI'] = lambda pred, oracle_p, time, event, X: tester.compute_mutual_information(oracle_p)

    # 3. VARIANCE: Ensemble disagreement
    strategies['Variance'] = lambda pred, oracle_p, time, event, X: tester.compute_variance(pred)

    # 4. OBSERVABLE MASS: P(observable window)
    strategies['Observable Mass'] = lambda pred, oracle_p, time, event, X: tester.compute_observable_mass(pred, time)

    # 5. P(DEATH): Probability of death reveal
    strategies['P(death in window)'] = lambda pred, oracle_p, time, event, X: tester.compute_p_death_window(oracle_p)

    # 6. HYBRID: Entropy * Observable Mass
    def entropy_times_obs_mass(pred, oracle_p, time, event, X):
        entropy = tester.compute_entropy(pred)
        obs_mass = tester.compute_observable_mass(pred, time)
        return entropy * obs_mass
    strategies['Entropy × ObsMass'] = entropy_times_obs_mass

    # 7. HYBRID: Entropy * P(death)
    def entropy_times_pdeath(pred, oracle_p, time, event, X):
        entropy = tester.compute_entropy(pred)
        p_death = tester.compute_p_death_window(oracle_p)
        return entropy * p_death
    strategies['Entropy × P(death)'] = entropy_times_pdeath

    # 8. HYBRID: MI * Observable Mass
    def mi_times_obs_mass(pred, oracle_p, time, event, X):
        mi = tester.compute_mutual_information(oracle_p)
        obs_mass = tester.compute_observable_mass(pred, time)
        return mi * obs_mass
    strategies['MI × ObsMass'] = mi_times_obs_mass

    # 9. DENSITY-WEIGHTED: Entropy * Density
    def entropy_times_density(pred, oracle_p, time, event, X):
        entropy = tester.compute_entropy(pred)
        density = tester.compute_density(X)
        return entropy * density
    strategies['Entropy × Density'] = entropy_times_density

    # 10. ANTI-DENSITY: Entropy / Density (prefer outliers)
    def entropy_div_density(pred, oracle_p, time, event, X):
        entropy = tester.compute_entropy(pred)
        density = tester.compute_density(X)
        return entropy / (density + 1e-3)
    strategies['Entropy / Density'] = entropy_div_density

    # 11. EXPECTED MODEL CHANGE
    strategies['Expected Model Change'] = lambda pred, oracle_p, time, event, X: tester.compute_expected_model_change(pred, X)

    # 12. FILTERED ENTROPY: Only high observable mass
    def filtered_entropy(pred, oracle_p, time, event, X):
        entropy = tester.compute_entropy(pred)
        obs_mass = tester.compute_observable_mass(pred, time)
        scores = entropy.copy()
        scores[obs_mass < 0.2] = -np.inf  # Filter threshold
        return scores
    strategies['Filtered Entropy (obs>0.2)'] = filtered_entropy

    # 13. FILTERED ENTROPY: Only high P(death)
    def filtered_entropy_pdeath(pred, oracle_p, time, event, X):
        entropy = tester.compute_entropy(pred)
        p_death = tester.compute_p_death_window(oracle_p)
        scores = entropy.copy()
        scores[p_death < 0.3] = -np.inf  # Filter threshold
        return scores
    strategies['Filtered Entropy (pd>0.3)'] = filtered_entropy_pdeath

    n_runs = 10

    print(f"Testing {len(strategies)} strategies with {n_runs} runs each...\n")

    # Store all data for analysis
    all_run_data = []

    # First, collect all data (to enable analysis later)
    probe_depth = 5
    for run in range(n_runs):
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

        model = BayesianSurvivalModel(
            n_features=10, n_time_bins=15, n_ensemble=5, hidden_size=64
        )
        model.fit(
            train_data['X'], artificial_time, artificial_event,
            epochs=40, batch_size=32, verbose=False
        )

        train_preds = model.predict_proba(train_data['X'])
        oracle = Oracle(true_train_time, true_train_event, probe_depth)
        oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
            train_preds, artificial_time, artificial_event
        )

        all_run_data.append({
            'predictions': train_preds,
            'oracle_probs': oracle_probs,
            'current_time': artificial_time.copy(),
            'X': train_data['X']
        })

    # Test all strategies
    all_results = {}

    for strategy_name, score_func in strategies.items():
        print(f"  Testing {strategy_name}...", end=" ", flush=True)
        results = test_strategy(strategy_name, score_func, n_runs=n_runs)
        all_results[strategy_name] = results

        improvements = [r['improvement'] for r in results]
        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        wins = sum(1 for imp in improvements if imp > 0)

        print(f"Δ={mean_imp:+.4f}±{std_imp:.4f}, Wins={wins}/{n_runs}")

    # Compare to baseline
    print(f"\n{'='*100}")
    print("RESULTS RANKED BY PERFORMANCE")
    print(f"{'='*100}\n")

    baseline_imps = [r['improvement'] for r in all_results['Entropy (baseline)']]

    ranked_results = []
    for strategy_name, results in all_results.items():
        improvements = [r['improvement'] for r in results]
        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        wins = sum(1 for imp in improvements if imp > 0)

        if strategy_name != 'Entropy (baseline)':
            t_stat, p_value = ttest_rel(improvements, baseline_imps)
            diff = mean_imp - np.mean(baseline_imps)
            pct_diff = (diff / np.mean(baseline_imps)) * 100 if np.mean(baseline_imps) != 0 else 0
        else:
            p_value = 1.0
            pct_diff = 0.0
            diff = 0.0

        ranked_results.append({
            'name': strategy_name,
            'mean': mean_imp,
            'std': std_imp,
            'wins': wins,
            'p_value': p_value,
            'diff': diff,
            'pct_diff': pct_diff
        })

    # Sort by mean improvement
    ranked_results.sort(key=lambda x: x['mean'], reverse=True)

    print(f"{'Rank':<5} {'Strategy':<35} {'Mean Δ':<15} {'Wins':<10} {'vs Baseline':<20} {'p-value':<10}")
    print("-" * 100)

    for rank, result in enumerate(ranked_results, 1):
        marker = "🏆" if rank == 1 else ("✓" if result['p_value'] < 0.05 and result['diff'] > 0 else " ")

        print(f"{marker} {rank:<3} {result['name']:<35} {result['mean']:+.4f}±{result['std']:.4f}   "
              f"{result['wins']}/{n_runs}      {result['pct_diff']:+.1f}%              "
              f"{result['p_value']:.4f}")

    # Analyze top 3 strategies
    print(f"\n{'='*100}")
    print("DETAILED ANALYSIS OF TOP 3 STRATEGIES")
    print(f"{'='*100}")

    for rank, result in enumerate(ranked_results[:3], 1):
        strategy_name = result['name']
        analyze_selections(strategy_name, all_results[strategy_name], all_run_data)

    # KEY INSIGHTS
    print(f"\n{'='*100}")
    print("KEY INSIGHTS")
    print(f"{'='*100}\n")

    winner = ranked_results[0]
    baseline = [r for r in ranked_results if r['name'] == 'Entropy (baseline)'][0]

    print(f"Best strategy: {winner['name']}")
    print(f"  Performance: {winner['mean']:+.4f} (vs Entropy {baseline['mean']:+.4f})")
    print(f"  Improvement over baseline: {winner['pct_diff']:+.1f}%")
    print(f"  Statistical significance: p={winner['p_value']:.4f} {'✓ SIGNIFICANT' if winner['p_value'] < 0.05 else '✗ Not significant'}")
    print(f"  Win rate: {winner['wins']}/{n_runs} ({winner['wins']/n_runs*100:.0f}%)")

    # Find what type of strategy works
    print(f"\nStrategy type analysis:")
    hybrid_strategies = [r for r in ranked_results if '×' in r['name'] or '/' in r['name']]
    if hybrid_strategies:
        avg_hybrid = np.mean([r['mean'] for r in hybrid_strategies])
        print(f"  Hybrid strategies avg: {avg_hybrid:+.4f}")

    filtered_strategies = [r for r in ranked_results if 'Filtered' in r['name']]
    if filtered_strategies:
        avg_filtered = np.mean([r['mean'] for r in filtered_strategies])
        print(f"  Filtered strategies avg: {avg_filtered:+.4f}")

    print(f"\n{'='*100}\n")


if __name__ == '__main__':
    main()
