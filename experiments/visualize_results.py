"""
Visualization of BatchBALD improvements.

Creates plots showing:
1. C-index improvements across methods
2. Statistical significance
3. Correlation between observable mass and informativeness
4. Ensemble diversity improvements
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from src.data.synthetic import generate_survival_data, artificially_censor, split_data
from src.models.survival_model import BayesianSurvivalModel
from src.oracle.oracle import Oracle
from src.acquisition.improved_batchbald import ImprovedBatchBALD
from src.acquisition.entropy import EntropyAcquisition
from src.evaluation.metrics import concordance_index

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


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


def generate_sample_results(n_runs=10):
    """Generate sample results for visualization."""

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

    print(f"Generating results for {n_runs} runs...")

    for run in range(n_runs):
        print(f"\r  Run {run+1}/{n_runs}...", end="", flush=True)

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

        for method_name, acq_func in methods.items():
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

            all_results[method_name].append({
                'improvement': updated_c_index - initial_c_index,
                'n_revealed': n_revealed
            })

    print("\n")
    return all_results


def plot_improvements(results, save_path='results/improvements_comparison.png'):
    """Plot C-index improvements."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Box plot
    data_for_box = []
    labels = []
    for method_name, method_results in results.items():
        improvements = [r['improvement'] for r in method_results]
        data_for_box.append(improvements)
        labels.append(method_name.replace(' (0.3, 0.2)', '').replace(' (0.4, 0.3)', ''))

    bp = ax1.boxplot(data_for_box, labels=labels, patch_artist=True)

    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_ylabel('C-index Improvement', fontsize=12)
    ax1.set_title('Distribution of C-index Improvements', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    ax1.grid(True, alpha=0.3)

    # Bar plot with error bars
    means = [np.mean([r['improvement'] for r in results[m]]) for m in results.keys()]
    stds = [np.std([r['improvement'] for r in results[m]]) for m in results.keys()]

    x_pos = np.arange(len(labels))
    bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors)

    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Mean C-index Improvement', fontsize=12)
    ax2.set_title('Mean Improvements with Standard Deviation', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.001,
                f'{mean:+.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_statistical_significance(results, save_path='results/statistical_significance.png'):
    """Plot statistical significance tests."""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Compute p-values
    entropy_imps = [r['improvement'] for r in results['Entropy']]

    method_names = []
    p_values = []
    mean_diffs = []

    for method_name in results.keys():
        if method_name == 'Entropy':
            continue

        method_imps = [r['improvement'] for r in results[method_name]]
        t_stat, p_value = ttest_rel(method_imps, entropy_imps)

        mean_diff = np.mean(method_imps) - np.mean(entropy_imps)

        method_names.append(method_name.replace(' (0.3, 0.2)', '\n(0.3, 0.2)').replace(' (0.4, 0.3)', '\n(0.4, 0.3)'))
        p_values.append(p_value)
        mean_diffs.append(mean_diff)

    # Create bar plot
    x_pos = np.arange(len(method_names))
    colors = ['green' if p < 0.05 else 'orange' if p < 0.1 else 'red' for p in p_values]

    bars = ax.bar(x_pos, p_values, color=colors, alpha=0.7)

    # Add significance threshold line
    ax.axhline(y=0.05, color='green', linestyle='--', linewidth=2, label='p=0.05 (significant)')
    ax.axhline(y=0.10, color='orange', linestyle='--', linewidth=2, label='p=0.10 (marginal)')

    ax.set_ylabel('p-value (vs Entropy)', fontsize=12)
    ax.set_title('Statistical Significance Test Results\n(Paired t-test vs Entropy baseline)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, fontsize=10)
    ax.set_ylim(0, max(p_values) * 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add p-value labels and mean difference
    for i, (bar, p_val, mean_diff) in enumerate(zip(bars, p_values, mean_diffs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'p={p_val:.3f}\nΔ={mean_diff:+.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_win_rates(results, save_path='results/win_rates.png'):
    """Plot win rates (proportion of runs with positive improvement)."""

    fig, ax = plt.subplots(figsize=(10, 6))

    method_names = []
    win_rates = []
    n_runs = len(list(results.values())[0])

    for method_name, method_results in results.items():
        improvements = [r['improvement'] for r in method_results]
        wins = sum(1 for imp in improvements if imp > 0)
        win_rate = wins / n_runs

        method_names.append(method_name.replace(' (0.3, 0.2)', '\n(0.3, 0.2)').replace(' (0.4, 0.3)', '\n(0.4, 0.3)'))
        win_rates.append(win_rate)

    x_pos = np.arange(len(method_names))
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    bars = ax.bar(x_pos, win_rates, color=colors, alpha=0.7)

    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% (random)')
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.set_title('Win Rate: Proportion of Runs with Positive Improvement',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for bar, win_rate in zip(bars, win_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{win_rate*100:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_observable_mass_correlation(save_path='results/observable_mass_correlation.png'):
    """
    Plot correlation between observable mass and oracle informativeness.

    This recreates the key insight from analyze_what_matters.py.
    """

    print("Generating correlation data...")

    probe_depth = 5

    # Generate single dataset
    data = generate_survival_data(
        n_samples=1000, n_features=10, n_time_bins=15,
        censoring_rate=0.3, random_state=42
    )

    train_data, test_data = split_data(data, train_size=0.7, random_state=42)
    true_train_time = train_data['time'].copy()
    true_train_event = train_data['event'].copy()

    artificial_time, artificial_event = artificially_censor(
        train_data['time'], train_data['event'],
        proportion=0.5, random_state=42
    )

    # Train model
    model = BayesianSurvivalModel(
        n_features=10, n_time_bins=15, n_ensemble=5, hidden_size=64
    )
    model.fit(
        train_data['X'], artificial_time, artificial_event,
        epochs=40, batch_size=32, verbose=False
    )

    # Get predictions and oracle probs
    train_preds = model.predict_proba(train_data['X'])
    oracle = Oracle(true_train_time, true_train_event, probe_depth)
    oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
        train_preds, artificial_time, artificial_event
    )

    # Compute features for censored samples
    acq = ImprovedBatchBALD(probe_depth, variant='observable_mass')

    observable_mass = []
    mutual_info = []
    true_informativeness = []

    for i in range(len(artificial_time)):
        if artificial_event[i] == 1:
            continue  # Skip already uncensored

        # Observable mass
        c = artificial_time[i]
        max_obs = min(c + probe_depth, train_preds.shape[2] - 1)
        mean_preds = train_preds.mean(axis=0)
        if c + 1 <= max_obs:
            obs_mass = mean_preds[i, c+1:max_obs+1].sum()
        else:
            obs_mass = 0.0
        observable_mass.append(obs_mass)

        # Mutual information
        mi = acq.compute_mutual_information(oracle_probs[:, [i], :])
        mutual_info.append(mi[0])

        # True informativeness (entropy over true oracle outcomes)
        true_outcome_probs = oracle.get_true_outcome_probs(
            i, artificial_time[i], artificial_event[i]
        )
        entropy = -np.sum([p * np.log(p + 1e-10) for p in true_outcome_probs if p > 0])
        true_informativeness.append(entropy)

    # Create scatter plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Observable mass vs informativeness
    ax1.scatter(observable_mass, true_informativeness, alpha=0.5, s=20)
    corr1 = np.corrcoef(observable_mass, true_informativeness)[0, 1]
    z1 = np.polyfit(observable_mass, true_informativeness, 1)
    p1 = np.poly1d(z1)
    x_line = np.linspace(min(observable_mass), max(observable_mass), 100)
    ax1.plot(x_line, p1(x_line), "r--", linewidth=2, label=f'ρ = {corr1:+.4f}')

    ax1.set_xlabel('Observable Mass', fontsize=12)
    ax1.set_ylabel('True Oracle Informativeness (Entropy)', fontsize=12)
    ax1.set_title('Observable Mass vs Oracle Informativeness\n(KEY INSIGHT: Best predictor!)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # MI vs informativeness
    ax2.scatter(mutual_info, true_informativeness, alpha=0.5, s=20, color='orange')
    corr2 = np.corrcoef(mutual_info, true_informativeness)[0, 1]
    z2 = np.polyfit(mutual_info, true_informativeness, 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(min(mutual_info), max(mutual_info), 100)
    ax2.plot(x_line2, p2(x_line2), "r--", linewidth=2, label=f'ρ = {corr2:+.4f}')

    ax2.set_xlabel('Mutual Information (BatchBALD)', fontsize=12)
    ax2.set_ylabel('True Oracle Informativeness (Entropy)', fontsize=12)
    ax2.set_title('Mutual Information vs Oracle Informativeness\n(Weaker correlation than Observable Mass)',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    """Generate all visualizations."""

    print("\n" + "="*100)
    print("VISUALIZATION: BatchBALD Improvements")
    print("="*100 + "\n")

    # Generate results
    results = generate_sample_results(n_runs=10)

    print("\nGenerating visualizations...")

    # Create plots
    plot_improvements(results)
    plot_statistical_significance(results)
    plot_win_rates(results)
    plot_observable_mass_correlation()

    print("\n" + "="*100)
    print("All visualizations saved to results/ directory!")
    print("="*100 + "\n")

    # Print summary
    print("Summary:")
    for method_name, method_results in results.items():
        improvements = [r['improvement'] for r in method_results]
        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        wins = sum(1 for imp in improvements if imp > 0)
        print(f"  {method_name:40s}: Δ={mean_imp:+.4f}±{std_imp:.4f}, Wins={wins}/10")


if __name__ == '__main__':
    main()
