"""
SYSTEMATIC ANALYSIS: What Actually Predicts Good Oracle Revelations?

Step 1: Understand the data
Step 2: Identify what features correlate with informative revelations
Step 3: Build better acquisition functions based on insights
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.synthetic import generate_survival_data, artificially_censor, split_data
from src.models.survival_model import BayesianSurvivalModel
from src.oracle.oracle import Oracle
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cdist


def analyze_what_matters():
    """Systematically understand what predicts good revelations."""

    print("\n" + "="*100)
    print("SYSTEMATIC ANALYSIS: What Predicts Good Oracle Revelations?")
    print("="*100 + "\n")

    # Generate data
    np.random.seed(42)
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
    print("Training model...")
    model = BayesianSurvivalModel(
        n_features=10, n_time_bins=15, n_ensemble=5, hidden_size=64
    )
    model.fit(
        train_data['X'], artificial_time, artificial_event,
        epochs=50, batch_size=32, verbose=False
    )

    train_preds = model.predict_proba(train_data['X'])  # (K, N, T)
    K, N, T = train_preds.shape

    print(f"Data: K={K}, N={N}, T={T}")
    print(f"Censored: {(artificial_event == 0).sum()}/{N}\n")

    # Setup
    probe_depth = 5
    oracle = Oracle(true_train_time, true_train_event, probe_depth)
    oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
        train_preds, artificial_time, artificial_event
    )

    # Focus on censored samples
    censored_mask = artificial_event == 0
    censored_indices = np.where(censored_mask)[0]
    n_censored = len(censored_indices)

    print(f"Analyzing {n_censored} censored samples...\n")

    # ===========================================================================================
    # STEP 1: Compute features for each censored sample
    # ===========================================================================================

    print("="*100)
    print("STEP 1: Computing Features for Each Sample")
    print("="*100 + "\n")

    features = {}

    # 1. True informativeness (what we want to predict)
    true_info = np.zeros(n_censored)
    for i, idx in enumerate(censored_indices):
        c_time = artificial_time[idx]
        max_obs = c_time + probe_depth

        if true_train_event[idx] == 1 and true_train_time[idx] <= max_obs:
            true_info[i] = 1.0  # Death will be revealed
        elif true_train_time[idx] > c_time:
            true_info[i] = 0.3  # Censoring extended
        else:
            true_info[i] = 0.0  # No new info

    features['true_info'] = true_info

    # 2. Mutual information (current BatchBALD score)
    mean_oracle_probs = oracle_probs.mean(axis=0)
    H_expected = -np.sum(mean_oracle_probs * np.log(mean_oracle_probs + 1e-8), axis=1)
    H_conditional = -np.sum(oracle_probs * np.log(oracle_probs + 1e-8), axis=2).mean(axis=0)
    mutual_info = H_expected - H_conditional

    features['mutual_info'] = mutual_info[censored_mask]

    # 3. Probability of death within window
    p_death_in_window = oracle_probs[:, censored_mask, :-1].sum(axis=2).mean(axis=0)
    features['p_death_window'] = p_death_in_window

    # 4. Variance in "death within window" across ensemble
    death_window_probs = oracle_probs[:, censored_mask, :-1].sum(axis=2)  # (K, n_censored)
    death_window_variance = death_window_probs.var(axis=0)
    features['death_window_variance'] = death_window_variance

    # 5. Predictive entropy (standard entropy baseline)
    mean_preds = train_preds.mean(axis=0)
    pred_entropy = -np.sum(mean_preds * np.log(mean_preds + 1e-8), axis=1)
    features['pred_entropy'] = pred_entropy[censored_mask]

    # 6. Censoring time (how early are they censored?)
    features['censor_time'] = artificial_time[censored_mask]

    # 7. Mean predicted time of death
    time_bins = np.arange(T)
    mean_predicted_time = (mean_preds * time_bins).sum(axis=1)
    features['mean_pred_time'] = mean_predicted_time[censored_mask]

    # 8. Distance to predicted death from censoring time
    features['time_to_pred_death'] = features['mean_pred_time'] - features['censor_time']

    # 9. Probability mass in observable window
    observable_mass = np.zeros(n_censored)
    for i, idx in enumerate(censored_indices):
        c = artificial_time[idx]
        max_obs = min(c + probe_depth, T - 1)
        observable_mass[i] = mean_preds[idx, c+1:max_obs+1].sum()
    features['observable_mass'] = observable_mass

    # 10. Density p(x) - distance to nearest neighbors
    X_censored = train_data['X'][censored_mask]
    distances = cdist(X_censored, train_data['X'], metric='euclidean')
    # Average distance to 5 nearest neighbors
    k_neighbors = 5
    sorted_distances = np.sort(distances, axis=1)
    avg_nn_dist = sorted_distances[:, 1:k_neighbors+1].mean(axis=1)
    features['density'] = 1.0 / (avg_nn_dist + 1e-8)  # Higher = denser

    # 11. Model confidence (max probability)
    max_prob = mean_preds[censored_mask].max(axis=1)
    features['confidence'] = max_prob

    print("Computed features:")
    for name, values in features.items():
        if name != 'true_info':
            print(f"  {name:25s}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")

    # ===========================================================================================
    # STEP 2: Analyze correlations
    # ===========================================================================================

    print("\n" + "="*100)
    print("STEP 2: What Correlates with Informative Revelations?")
    print("="*100 + "\n")

    print(f"{'Feature':<30} {'Spearman ρ':<15} {'p-value':<12} {'Pearson r':<15}")
    print("-"*80)

    correlations = {}

    for name, values in features.items():
        if name == 'true_info':
            continue

        # Spearman (rank) correlation
        rho, p_spearman = spearmanr(values, true_info)

        # Pearson correlation
        r, p_pearson = pearsonr(values, true_info)

        correlations[name] = {
            'spearman': rho,
            'p_spearman': p_spearman,
            'pearson': r,
            'p_pearson': p_pearson
        }

        marker = "★" if abs(rho) > 0.15 or abs(r) > 0.15 else " "
        print(f"{marker} {name:<28} {rho:+.4f}          {p_spearman:.4f}      {r:+.4f}")

    # ===========================================================================================
    # STEP 3: Identify top predictors
    # ===========================================================================================

    print("\n" + "="*100)
    print("STEP 3: Top Predictive Features")
    print("="*100 + "\n")

    # Sort by absolute Spearman correlation
    sorted_features = sorted(correlations.items(),
                            key=lambda x: abs(x[1]['spearman']),
                            reverse=True)

    print("Ranking by correlation strength:\n")
    for rank, (name, corr) in enumerate(sorted_features[:10], 1):
        print(f"{rank:2d}. {name:<30} ρ={corr['spearman']:+.4f}, p={corr['p_spearman']:.4f}")

    # ===========================================================================================
    # STEP 4: Build composite scores
    # ===========================================================================================

    print("\n" + "="*100)
    print("STEP 4: Testing Composite Scores")
    print("="*100 + "\n")

    # Test different combinations
    composites = {}

    # Current BatchBALD
    composites['BatchBALD (MI)'] = features['mutual_info']

    # Death probability focus
    composites['P(death in window)'] = features['p_death_window']

    # Variance in death probability
    composites['Var(death window)'] = features['death_window_variance']

    # Observable mass
    composites['Observable mass'] = features['observable_mass']

    # Combination: MI * P(death)
    composites['MI * P(death)'] = features['mutual_info'] * features['p_death_window']

    # Combination: P(death) * Observable mass
    composites['P(death) * Obs mass'] = features['p_death_window'] * features['observable_mass']

    # Combination: Death variance + P(death)
    norm_var = (features['death_window_variance'] - features['death_window_variance'].min()) / \
               (features['death_window_variance'].max() - features['death_window_variance'].min() + 1e-8)
    norm_p = (features['p_death_window'] - features['p_death_window'].min()) / \
             (features['p_death_window'].max() - features['p_death_window'].min() + 1e-8)
    composites['Var + P(death)'] = norm_var + norm_p

    # With density weighting
    norm_density = (features['density'] - features['density'].min()) / \
                   (features['density'].max() - features['density'].min() + 1e-8)
    composites['P(death) * density'] = features['p_death_window'] * (1 + norm_density)

    # Expected information weighted by death prob
    composites['MI * P(death)^2'] = features['mutual_info'] * (features['p_death_window'] ** 2)

    print(f"{'Composite Score':<30} {'Correlation':<15} {'p-value':<12}")
    print("-"*60)

    composite_results = {}
    for name, score in composites.items():
        rho, p_val = spearmanr(score, true_info)
        composite_results[name] = {'rho': rho, 'p': p_val}

        marker = "🏆" if rho > 0.15 else ("★" if rho > 0.10 else " ")
        print(f"{marker} {name:<28} {rho:+.4f}          {p_val:.4f}")

    # ===========================================================================================
    # STEP 5: Find best composite
    # ===========================================================================================

    print("\n" + "="*100)
    print("STEP 5: Best Strategy")
    print("="*100 + "\n")

    best_name = max(composite_results.items(), key=lambda x: x[1]['rho'])[0]
    best_rho = composite_results[best_name]['rho']

    print(f"🎯 BEST COMPOSITE: {best_name}")
    print(f"   Correlation: ρ = {best_rho:+.4f}")
    print(f"   vs BatchBALD: ρ = {composite_results['BatchBALD (MI)']['rho']:+.4f}")
    print(f"   Improvement: {best_rho - composite_results['BatchBALD (MI)']['rho']:+.4f}")

    if best_rho > composite_results['BatchBALD (MI)']['rho'] + 0.05:
        print(f"\n✅ Found a BETTER strategy than pure MI!")
    else:
        print(f"\n⚠️  No major improvement found, but insights can still help.")

    # ===========================================================================================
    # STEP 6: Recommendations
    # ===========================================================================================

    print("\n" + "="*100)
    print("STEP 6: Recommendations for Improved BatchBALD")
    print("="*100 + "\n")

    top_features = sorted_features[:3]

    print("Based on analysis, prioritize:\n")
    for i, (name, corr) in enumerate(top_features, 1):
        print(f"{i}. {name} (ρ={corr['spearman']:+.4f})")

    print(f"\nProposed improvements:")
    print(f"1. Weight MI by P(death in window): MI * P(death)^α where α ∈ [1, 2]")
    print(f"2. Add observable mass term: ensure predicted mass in window")
    print(f"3. Filter samples where P(death window) < threshold (e.g., 0.1)")

    if composite_results['P(death) * density']['rho'] > composite_results['BatchBALD (MI)']['rho']:
        print(f"4. Density weighting HELPS: multiply by (1 + normalized_density)")

    print("\n" + "="*100 + "\n")

    return features, composite_results, best_name


if __name__ == '__main__':
    features, results, best = analyze_what_matters()
