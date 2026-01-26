"""
Test signal-to-noise ratio for acquisition function comparison.
Ensures there's enough signal to detect statistically significant differences.
"""

import sys
sys.path.insert(0, 'Model_stuff')

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import argparse
from scipy import stats

# Import from Model_stuff
from Model_stuff.data import make_nacd_data
from Model_stuff.model import BayesLinMtlr, mtlr_survival

# Import directly to avoid umap dependency
def encode_survival(time, event, bins):
    """Encodes survival time and event indicator for MTLR training."""
    if isinstance(time, (float, int, np.ndarray)):
        time = np.atleast_1d(time)
        time = torch.tensor(time)
    if isinstance(event, (int, bool, np.ndarray)):
        event = np.atleast_1d(event)
        event = torch.tensor(event)
    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins)

    bins = torch.as_tensor(bins, device=time.device, dtype=time.dtype)
    device = bins.device if hasattr(bins, 'device') else "cpu"
    time = np.clip(time, 0, bins.max())
    y = torch.zeros((time.shape[0], bins.shape[0] + 1), dtype=torch.float, device=device)
    bin_idxs = torch.bucketize(time, bins, right=True)
    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y.squeeze()


def reformat_survival(dataset, time_bins):
    """Reformat survival data for training."""
    x = torch.tensor(dataset.drop(["time", "event"], axis=1).values, dtype=torch.float)
    y = encode_survival(dataset["time"].values, dataset["event"].values, time_bins)
    return x, y


def artificially_censor_true(times, events, num_initial_samples=50):
    """Artificially censor data points."""
    censored_times = np.copy(times)
    new_events = np.copy(events)
    uncensored_indices = np.random.choice(len(times), num_initial_samples, replace=False)
    censored_indices = []

    for i in range(len(times)):
        if i in uncensored_indices:
            continue
        censoring_time = np.random.uniform(0, times[i])
        censored_times[i] = censoring_time
        new_events[i] = 0
        censored_indices.append(i)

    return censored_times, new_events, np.array(censored_indices)


def concordance(y_pred, y_test, cens):
    """Calculate the concordance index (C-index)."""
    n = 0
    n_concordant = 0

    for i in range(len(y_test)):
        for j in range(i + 1, len(y_test)):
            if (y_test[i] < y_test[j] and cens[i] == 1) or (y_test[j] < y_test[i] and cens[j] == 1):
                n += 1
                if y_test[i] < y_test[j]:
                    if y_pred[i] > y_pred[j]:
                        n_concordant += 1
                else:
                    if y_pred[j] > y_pred[i]:
                        n_concordant += 1

    if n == 0:
        return 0.0
    return n_concordant / n


def expected_times_from_survival(surv_np, tbins):
    """Convert survival probabilities to expected survival times."""
    if isinstance(tbins, torch.Tensor):
        tb = tbins.clone().detach().cpu().numpy()
    else:
        tb = np.array(tbins)
    left_edges = np.concatenate([[0.0], tb[:-1]])
    right_edges = tb
    mids = (left_edges + right_edges) / 2.0
    pdf = np.zeros_like(surv_np)
    pdf[:, 0] = 1.0 - surv_np[:, 0]
    pdf[:, 1:] = surv_np[:, :-1] - surv_np[:, 1:]
    pdf = np.clip(pdf, 0.0, 1.0)
    pdf = pdf / (pdf.sum(axis=1, keepdims=True) + 1e-12)
    return (pdf * mids.reshape(1, -1)).sum(axis=1)


def train_and_evaluate(X_train, y_train, e_train, X_test, y_test, e_test,
                       time_bins, config, verbose=False):
    """Train model and return c-index."""
    train_df = pd.DataFrame(X_train)
    train_df['time'] = y_train
    train_df['event'] = e_train
    x_formatted, y_formatted = reformat_survival(train_df, time_bins[1:])

    train_dataset = TensorDataset(x_formatted, y_formatted)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = BayesLinMtlr(
        in_features=X_train.shape[1],
        num_time_bins=len(time_bins) - 1,
        config=config
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)

    best_loss = float('inf')
    patience_counter = 0
    patience = 20

    for epoch in range(100):
        model.train()
        total_loss = 0.0

        for xi, yi in train_loader:
            xi, yi = xi.to(config.device), yi.to(config.device)
            optimizer.zero_grad()
            loss, _, _, _ = model.sample_elbo(xi, yi, len(X_train), see1=config.c1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(config.device)
        logits = model.forward(X_test_tensor, sample=True, n_samples=config.n_samples_test)
        survival_probs = mtlr_survival(logits, with_sample=True)
        mean_survival = survival_probs.mean(dim=0).cpu().numpy()

    pred_times = expected_times_from_survival(mean_survival, time_bins)
    c_index = concordance(-pred_times, y_test, e_test)

    return c_index


def test_signal_to_noise(n_trials=10):
    """Test if we have enough signal to detect differences."""

    print("="*70)
    print("SIGNAL-TO-NOISE RATIO TEST")
    print("="*70)
    print(f"\nTesting with {n_trials} trials to estimate variance and effect sizes\n")

    # Load data
    print("1. Loading NACD dataset...")

    # Load NACD dataset directly
    import os
    nacd_path = "data/MIMIC/NACD/NACD_Full.csv"
    if not os.path.exists(nacd_path):
        raise FileNotFoundError(f"NACD file not found at {nacd_path}")

    data = pd.read_csv(nacd_path)

    # Preprocess NACD data
    cols_to_drop = ['PERFORMANCE_STATUS', 'STAGE_NUMERICAL', 'AGE65']
    data = data.drop([c for c in cols_to_drop if c in data.columns], axis=1)

    if "CENSORED" in data.columns:
        data["event"] = 1 - data["CENSORED"]
        data = data.drop(columns=["CENSORED"])
    if "SURVIVAL" in data.columns:
        data = data.rename(columns={"SURVIVAL": "time"})

    cols_standardize = ['BOX1_SCORE', 'BOX2_SCORE', 'BOX3_SCORE', 'BMI', 'WEIGHT_CHANGEPOINT',
                        'AGE', 'GRANULOCYTES', 'LDH_SERUM', 'LYMPHOCYTES',
                        'PLATELET', 'WBC_COUNT', 'CALCIUM_SERUM', 'HGB', 'CREATININE_SERUM', 'ALBUMIN']
    cols_standardize = [c for c in cols_standardize if c in data.columns]
    data[cols_standardize] = data[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())

    print(f"   Dataset shape: {data.shape}")

    # Prepare data
    X = data.drop(["time", "event"], axis=1).values
    y = data["time"].values
    e = data["event"].values

    # Train/test split (fixed)
    np.random.seed(42)
    X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(
        X, y, e, test_size=0.1, random_state=42, stratify=e
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create time bins
    num_bins = 10
    event_times = y_train[e_train == 1]
    quantiles = np.linspace(0, 1, num_bins + 1)[1:]
    time_bins = np.quantile(event_times, quantiles)
    time_bins[-1] *= 1.05
    time_bins = np.array([0] + list(time_bins))

    # Create config
    config = argparse.Namespace()
    config.pi = 0.5
    config.sigma1 = 1.0
    config.sigma2 = 0.0025
    config.rho_scale = -3.0
    config.mu_scale = 0.1
    config.batch_size = 32
    config.c1 = 0.01
    config.n_samples_train = 10
    config.n_samples_test = 100
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n2. Testing baseline variance (no acquisition)...")
    baseline_scores = []

    for trial in range(n_trials):
        seed = 100 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Apply artificial censoring
        y_labeled, e_labeled, _ = artificially_censor_true(
            y_train.copy(), e_train.copy(), num_initial_samples=200
        )

        c_index = train_and_evaluate(
            X_train, y_labeled, e_labeled, X_test, y_test, e_test,
            time_bins, config
        )
        baseline_scores.append(c_index)
        print(f"   Trial {trial+1}: C-index = {c_index:.4f}")

    baseline_mean = np.mean(baseline_scores)
    baseline_std = np.std(baseline_scores)

    print(f"\n   Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")

    print("\n3. Testing with random acquisition (pessimistic baseline)...")
    random_scores = []

    for trial in range(n_trials):
        seed = 200 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Apply artificial censoring
        y_labeled, e_labeled, censored_indices = artificially_censor_true(
            y_train.copy(), e_train.copy(), num_initial_samples=200
        )

        # Random acquisition (30 samples, 60 time increment)
        budget = 30
        increment = 60
        censored_list = list(censored_indices)
        if len(censored_list) >= budget:
            acquired_indices = np.random.choice(censored_list, budget, replace=False)

            # Update labels
            for idx in acquired_indices:
                y_labeled[idx] = min(y_train[idx], y_labeled[idx] + increment)
                if y_labeled[idx] >= y_train[idx]:
                    e_labeled[idx] = e_train[idx]

        c_index = train_and_evaluate(
            X_train, y_labeled, e_labeled, X_test, y_test, e_test,
            time_bins, config
        )
        random_scores.append(c_index)
        print(f"   Trial {trial+1}: C-index = {c_index:.4f}")

    random_mean = np.mean(random_scores)
    random_std = np.std(random_scores)

    print(f"\n   Random: {random_mean:.4f} ± {random_std:.4f}")

    # Analysis
    print("\n" + "="*70)
    print("SIGNAL-TO-NOISE ANALYSIS")
    print("="*70)

    # 1. Within-method variance
    print(f"\n1. Within-Method Variance:")
    print(f"   Baseline std: {baseline_std:.4f}")
    print(f"   Random std: {random_std:.4f}")
    print(f"   Average std: {(baseline_std + random_std)/2:.4f}")

    # 2. Improvement from random acquisition
    improvement = random_mean - baseline_mean
    pooled_std = np.sqrt((baseline_std**2 + random_std**2) / 2)

    print(f"\n2. Random Acquisition Effect:")
    print(f"   Mean improvement: {improvement:+.4f}")
    print(f"   Improvement / Std: {improvement / pooled_std:.2f}")

    # 3. Statistical power analysis
    # For two-sample t-test, effect size d = (μ1 - μ2) / σ_pooled
    effect_size = improvement / pooled_std

    print(f"\n3. Effect Size (Cohen's d):")
    print(f"   Random vs Baseline: {effect_size:.4f}")

    if abs(effect_size) < 0.2:
        print("   ⚠️  Small effect - may need 30+ trials")
    elif abs(effect_size) < 0.5:
        print("   ✓ Medium effect - 15-20 trials should suffice")
    else:
        print("   ✓✓ Large effect - 10 trials should be sufficient")

    # 4. Detectable difference
    # For t-test with α=0.05, power=0.80, n=10
    # Minimum detectable effect size ≈ 0.91 for two-sample t-test
    min_detectable_diff = 0.91 * pooled_std / np.sqrt(n_trials)

    print(f"\n4. Minimum Detectable Difference (α=0.05, power=0.80):")
    print(f"   With {n_trials} trials: {min_detectable_diff:.4f}")
    print(f"   Observed improvement: {improvement:.4f}")

    if abs(improvement) >= min_detectable_diff:
        print(f"   ✓ Observed improvement is detectable!")
    else:
        needed_trials = int(np.ceil((0.91 * pooled_std / improvement)**2))
        print(f"   ⚠️  May need ~{needed_trials} trials to detect this difference")

    # 5. Signal-to-noise ratio
    snr = abs(improvement) / pooled_std
    print(f"\n5. Signal-to-Noise Ratio:")
    print(f"   SNR = {snr:.4f}")

    if snr < 0.2:
        print("   ❌ Very poor - consider different settings")
        recommendation = "Poor"
    elif snr < 0.5:
        print("   ⚠️  Low - use 20+ trials for reliable results")
        recommendation = "Marginal"
    elif snr < 1.0:
        print("   ✓ Moderate - 10-15 trials should work")
        recommendation = "Good"
    else:
        print("   ✓✓ Good - 10 trials should be sufficient")
        recommendation = "Excellent"

    # 6. Expected differences between methods
    # Assume better methods have 2-3x the improvement of random
    expected_best_improvement = improvement * 2.5
    expected_effect_size = expected_best_improvement / pooled_std

    print(f"\n6. Expected Results for Better Methods:")
    print(f"   If better methods have 2.5x improvement of random:")
    print(f"   Expected improvement: {expected_best_improvement:+.4f}")
    print(f"   Expected effect size: {expected_effect_size:.4f}")

    # Statistical test between baseline and random
    t_stat, p_value = stats.ttest_ind(baseline_scores, random_scores)
    print(f"\n7. Current Statistical Significance:")
    print(f"   Baseline vs Random t-test:")
    print(f"   t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

    if p_value < 0.05:
        print(f"   ✓ Already statistically significant (p < 0.05)")
    elif p_value < 0.10:
        print(f"   ✓ Marginally significant (p < 0.10)")
    else:
        print(f"   ⚠️  Not significant yet (p > 0.10)")

    # Final recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    print(f"\nSignal Quality: {recommendation}")

    if recommendation == "Excellent":
        print("✓✓ Excellent signal-to-noise ratio!")
        print("   → Proceed with 10 trials for main comparison")
        print("   → Should detect differences between acquisition functions")
    elif recommendation == "Good":
        print("✓ Good signal-to-noise ratio")
        print("   → Proceed with 10-15 trials for main comparison")
        print("   → Should detect meaningful differences")
    elif recommendation == "Marginal":
        print("⚠️ Marginal signal-to-noise ratio")
        print("   → Use 20+ trials for reliable results")
        print("   → May need larger budget or different settings")
    else:
        print("❌ Poor signal-to-noise ratio")
        print("   → Consider increasing budget (e.g., 50 samples)")
        print("   → Consider larger probe increment (e.g., 90-120)")
        print("   → May need 30+ trials")

    print()

    return {
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'random_mean': random_mean,
        'random_std': random_std,
        'improvement': improvement,
        'effect_size': effect_size,
        'snr': snr,
        'p_value': p_value,
        'recommendation': recommendation
    }


if __name__ == "__main__":
    import argparse as arg_parser

    parser = arg_parser.ArgumentParser(description='Test signal-to-noise ratio')
    parser.add_argument('--n_trials', type=int, default=10,
                       help='Number of trials for testing')

    args = parser.parse_args()

    results = test_signal_to_noise(n_trials=args.n_trials)
