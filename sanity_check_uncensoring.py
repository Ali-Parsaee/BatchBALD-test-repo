"""
Sanity check: Does uncensoring more data improve model performance?

Tests that acquiring more information (decensoring points) leads to better C-index.
If this doesn't work, there's no point comparing acquisition strategies.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from Model_stuff.model import BayesLinMtlr, mtlr_survival


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

        if verbose and (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"      Early stopping at epoch {epoch+1}")
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


def sanity_check(n_trials=5, increment=60):
    """
    Test that uncensoring more data improves performance.

    Strategy:
    1. Start with heavily censored data (200 initial samples)
    2. Progressively uncensor more random samples: 0, 30, 60, 100, 150, 200 additional
    3. Train model at each stage
    4. Verify C-index increases with more uncensored data
    """

    print("="*70)
    print("SANITY CHECK: Does Uncensoring Data Improve Performance?")
    print("="*70)
    print()
    print(f"Testing with {n_trials} trials")
    print(f"Progressive uncensoring: 0 → 30 → 60 → 100 → 150 → 200 additional samples")
    print()

    # Load data
    print("1. Loading NACD dataset...")
    import os
    nacd_path = "data/MIMIC/NACD/NACD_Full.csv"
    if not os.path.exists(nacd_path):
        raise FileNotFoundError(f"NACD file not found at {nacd_path}")

    data = pd.read_csv(nacd_path)

    # Preprocess
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

    # Standardize
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

    # Test different amounts of uncensored data
    n_additional = [0, 30, 60, 100, 150, 200]  # Additional samples to uncensor
    initial_samples = 200

    print(f"\n2. Running {n_trials} trials with progressive uncensoring...")
    print()

    # Store results
    results = {n: [] for n in n_additional}

    for trial in range(n_trials):
        print(f"  Trial {trial + 1}/{n_trials}")
        seed = 100 + trial
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Apply artificial censoring
        y_labeled, e_labeled, censored_indices = artificially_censor_true(
            y_train.copy(), e_train.copy(), num_initial_samples=initial_samples
        )

        # Store original censored state
        y_original = y_train.copy()
        e_original = e_train.copy()

        for n_add in n_additional:
            # Reset to censored state
            y_current = y_labeled.copy()
            e_current = e_labeled.copy()

            if n_add > 0:
                # Randomly select additional samples to uncensor
                censored_list = list(censored_indices)
                if len(censored_list) >= n_add:
                    additional_indices = np.random.choice(censored_list, n_add, replace=False)

                    # Uncensor by revealing more information
                    for idx in additional_indices:
                        y_current[idx] = min(y_original[idx], y_current[idx] + increment)
                        if y_current[idx] >= y_original[idx]:
                            e_current[idx] = e_original[idx]

            # Train and evaluate
            c_index = train_and_evaluate(
                X_train, y_current, e_current, X_test, y_test, e_test,
                time_bins, config, verbose=False
            )

            results[n_add].append(c_index)
            print(f"    +{n_add:3d} samples: C-index = {c_index:.4f}")

        print()

    # Analyze results
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()

    print("Summary Statistics:")
    print("-" * 70)
    print(f"{'Additional Samples':<20} {'Mean C-Index':<15} {'Std':<10} {'Improvement':<15}")
    print("-" * 70)

    baseline_mean = np.mean(results[0])
    for n_add in n_additional:
        mean_cindex = np.mean(results[n_add])
        std_cindex = np.std(results[n_add])
        improvement = mean_cindex - baseline_mean

        print(f"{n_add:<20} {mean_cindex:.4f}          {std_cindex:.4f}     {improvement:+.4f}")

    # Statistical tests
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE")
    print("="*70)
    print()

    from scipy import stats as scipy_stats

    print("One-tailed t-tests vs baseline (0 additional samples):")
    print("-" * 70)

    for n_add in n_additional[1:]:  # Skip baseline
        t_stat, p_value = scipy_stats.ttest_rel(results[n_add], results[0], alternative='greater')

        significance = ""
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"

        mean_improvement = np.mean(results[n_add]) - np.mean(results[0])

        print(f"+{n_add:3d} samples: t={t_stat:+.3f}, p={p_value:.4f} {significance:3s} "
              f"(improvement: {mean_improvement:+.4f})")

    # Trend test
    print("\n" + "="*70)
    print("TREND ANALYSIS")
    print("="*70)
    print()

    # Calculate correlation between n_additional and c-index
    all_n_add = []
    all_cindices = []
    for n_add in n_additional:
        for c_idx in results[n_add]:
            all_n_add.append(n_add)
            all_cindices.append(c_idx)

    from scipy.stats import pearsonr, spearmanr
    pearson_r, pearson_p = pearsonr(all_n_add, all_cindices)
    spearman_r, spearman_p = spearmanr(all_n_add, all_cindices)

    print(f"Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.6f}")
    print(f"Spearman correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.6f}")

    # Create plot
    print("\n3. Creating visualization...")
    plt.figure(figsize=(10, 6))

    means = [np.mean(results[n]) for n in n_additional]
    stds = [np.std(results[n]) for n in n_additional]

    plt.errorbar(n_additional, means, yerr=stds, marker='o', capsize=5, capthick=2,
                linewidth=2, markersize=8, label='Mean ± Std')

    # Add individual points
    for i, n_add in enumerate(n_additional):
        plt.scatter([n_add] * len(results[n_add]), results[n_add],
                   alpha=0.3, s=50, color='blue')

    plt.xlabel('Number of Additional Uncensored Samples', fontsize=12)
    plt.ylabel('C-Index', fontsize=12)
    plt.title('Sanity Check: Model Performance vs. Amount of Uncensored Data', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add text box with statistics
    textstr = f'Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})\n'
    textstr += f'Final improvement: {means[-1] - means[0]:+.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('sanity_check_results.png', dpi=150)
    print("   Saved plot to: sanity_check_results.png")

    # Final verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    print()

    final_improvement = means[-1] - means[0]
    all_positive = all(means[i] >= means[i-1] for i in range(1, len(means)))

    if pearson_p < 0.05 and pearson_r > 0:
        print("✅ SANITY CHECK PASSED")
        print()
        print(f"   • Statistically significant positive correlation (p = {pearson_p:.6f})")
        print(f"   • Pearson r = {pearson_r:.4f}")
        print(f"   • Final improvement: {final_improvement:+.4f} ({final_improvement/baseline_mean*100:+.2f}%)")
        print()
        if all_positive:
            print("   • Monotonic improvement observed ✓")
        print()
        print("   → Uncensoring data DOES improve model performance")
        print("   → Safe to proceed with acquisition function comparison")
    else:
        print("❌ SANITY CHECK FAILED")
        print()
        print(f"   • Correlation not significant (p = {pearson_p:.6f})")
        print(f"   • Pearson r = {pearson_r:.4f}")
        print()
        print("   → Something is wrong with the setup")
        print("   → DO NOT proceed with acquisition function comparison")

    print()

    return results, means, stds, pearson_r, pearson_p


if __name__ == "__main__":
    import argparse as arg_parser

    parser = arg_parser.ArgumentParser(description='Sanity check for uncensoring effect')
    parser.add_argument('--n_trials', type=int, default=5,
                       help='Number of trials (default: 5)')
    parser.add_argument('--increment', type=int, default=60,
                       help='Time increment for probe (default: 60)')

    args = parser.parse_args()

    results, means, stds, r, p = sanity_check(
        n_trials=args.n_trials,
        increment=args.increment
    )
