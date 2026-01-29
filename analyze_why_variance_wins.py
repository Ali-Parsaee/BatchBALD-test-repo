"""
Deep analysis: WHY does variance-based selection outperform BALD?

Hypotheses to test:
1. Oracle revelation rate: Does variance select samples where the oracle reveals more events (deaths)?
2. Diversity: Does variance select more diverse samples (spread in feature/prediction space)?
3. Label informativeness: Are the revealed labels more informative for the model?
4. Death probability accuracy: Is variance better at identifying high death-prob samples?
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'Model_stuff')

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from torch.utils.data import DataLoader, TensorDataset
import argparse

from Model_stuff.model import BayesLinMtlr, mtlr_survival
from Model_stuff.acquisition import _make_prediction, _map_indices, select_indices_from_scores
from Model_stuff.utils import ensemble_to_pdf


def encode_survival(time, event, bins):
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
    x = torch.tensor(dataset.drop(["time", "event"], axis=1).values, dtype=torch.float)
    y = encode_survival(dataset["time"].values, dataset["event"].values, time_bins)
    return x, y


def artificially_censor_true(times, events, num_initial_samples=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
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


def main():
    seed = 100
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load NACD
    nacd_csv = 'data/MIMIC/NACD/NACD_Full.csv'
    df = pd.read_csv(nacd_csv)
    cols_to_drop = ['PERFORMANCE_STATUS', 'STAGE_NUMERICAL', 'AGE65']
    df = df.drop([c for c in cols_to_drop if c in df.columns], axis=1)
    if "CENSORED" in df.columns:
        df["event"] = 1 - df["CENSORED"]
        df = df.drop(columns=["CENSORED"])
    if "SURVIVAL" in df.columns:
        df = df.rename(columns={"SURVIVAL": "time"})
    cols_standardize = ['BOX1_SCORE', 'BOX2_SCORE', 'BOX3_SCORE', 'BMI', 'WEIGHT_CHANGEPOINT',
                        'AGE', 'GRANULOCYTES', 'LDH_SERUM', 'LYMPHOCYTES',
                        'PLATELET', 'WBC_COUNT', 'CALCIUM_SERUM', 'HGB', 'CREATININE_SERUM', 'ALBUMIN']
    cols_standardize = [c for c in cols_standardize if c in df.columns]
    df[cols_standardize] = df[cols_standardize].apply(lambda x: (x - x.mean()) / x.std())

    X = df.drop(columns=['time', 'event'])
    y_time = df['time'].values
    y_event = df['event'].values

    X_train_val, X_test, y_time_train_val, y_time_test, y_event_train_val, y_event_test = train_test_split(
        X, y_time, y_event, test_size=0.2, random_state=seed, stratify=y_event
    )

    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)

    y_time_censored, y_event_censored, censored_indices = artificially_censor_true(
        y_time_train_val, y_event_train_val, num_initial_samples=500, seed=seed
    )

    num_bins = 20
    event_times = y_time_train_val[y_event_train_val == 1]
    quantiles = np.linspace(0, 1, num_bins + 1)[1:]
    time_bins = np.quantile(event_times, quantiles)
    time_bins[-1] *= 1.05
    time_bins = np.array([0] + list(time_bins))

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
    config.device = "cpu"
    config.patience = 10

    # Train model
    print("Training model...")
    train_df = pd.DataFrame(X_train_val_scaled)
    train_df['time'] = y_time_censored
    train_df['event'] = y_event_censored

    x_train, y_train = reformat_survival(train_df, time_bins)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = BayesLinMtlr(
        in_features=X_train_val_scaled.shape[1],
        num_time_bins=len(time_bins),
        config=config
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss, _, _, _ = model.sample_elbo(batch_x, batch_y, len(x_train), see1=config.c1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    print("Model trained. Computing scores...\n")

    # Get pool data
    X_censored = X_train_val_scaled[censored_indices]
    y_time_original = y_time_train_val[censored_indices]  # TRUE times
    y_event_original = y_event_train_val[censored_indices]  # TRUE events
    y_time_cens = y_time_censored[censored_indices]  # Censored times

    increment = 20
    budget = 50

    # Compute scores
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X_censored).to(config.device)
        _, _, ensemble_outputs = _make_prediction(model, x_tensor, time_bins, config)
        pdf = ensemble_to_pdf(ensemble_outputs, config.device)

    N, K, T = pdf.shape
    eps = 1e-10

    # === Variance of expected times ===
    bin_mids = (np.concatenate([[0], time_bins[:-1]]) + time_bins) / 2
    bin_mids_tensor = torch.tensor(bin_mids[:T], dtype=torch.float32)
    expected_times = (pdf[:, :, :len(bin_mids_tensor)] * bin_mids_tensor).sum(dim=2)
    time_variance = expected_times.var(dim=1).numpy()
    mean_expected_time = expected_times.mean(dim=1).numpy()

    # === BALD ===
    mean_pdf = pdf.mean(dim=1)
    H_y = -torch.sum(mean_pdf * torch.log(mean_pdf + eps), dim=1)
    individual_entropies = -torch.sum(pdf * torch.log(pdf + eps), dim=2)
    E_H_y_theta = individual_entropies.mean(dim=1)
    bald_score = (H_y - E_H_y_theta).numpy()

    # === Death probability in window ===
    time_bins_np = np.array(time_bins)
    temp_bins = np.array([0] + list(time_bins_np)[:-1])
    start_bins = _map_indices(y_time_cens, temp_bins).astype(int)
    end_bins = _map_indices(y_time_cens + increment, temp_bins).astype(int)

    death_prob = np.zeros(N)
    mean_pdf_np = mean_pdf.numpy()
    for i in range(N):
        s = int(start_bins[i])
        e = min(int(end_bins[i]) + 1, mean_pdf_np.shape[1])
        death_prob[i] = mean_pdf_np[i, s:e].sum()

    # Final scores
    variance_score = time_variance * (0.5 + 0.5 * death_prob)
    bald_final_score = bald_score * (0.5 + 0.5 * death_prob)

    # Select top samples
    variance_selected = np.argsort(variance_score)[-budget:]
    bald_selected = np.argsort(bald_final_score)[-budget:]

    # === ANALYSIS ===
    print("=" * 70)
    print("ANALYSIS: WHY DOES VARIANCE WIN?")
    print("=" * 70)

    # 1. Oracle Revelation Rate
    print("\n" + "-" * 70)
    print("1. ORACLE REVELATION RATE")
    print("-" * 70)
    print("   When oracle reveals labels (censor_time + increment), how many are actual deaths?")

    # For each selected sample, check if TRUE death time falls within window
    def get_revealed_events(selected_indices):
        """Count how many selected samples have TRUE event within oracle window."""
        revealed = 0
        for idx in selected_indices:
            censor_time = y_time_cens[idx]
            true_time = y_time_original[idx]
            true_event = y_event_original[idx]
            # Oracle reveals up to censor_time + increment
            if true_event == 1 and true_time <= censor_time + increment:
                revealed += 1
        return revealed

    var_revealed = get_revealed_events(variance_selected)
    bald_revealed = get_revealed_events(bald_selected)

    print(f"   Variance: {var_revealed}/{budget} samples have TRUE death revealed ({100*var_revealed/budget:.1f}%)")
    print(f"   BALD:     {bald_revealed}/{budget} samples have TRUE death revealed ({100*bald_revealed/budget:.1f}%)")

    # 2. Diversity in Feature Space
    print("\n" + "-" * 70)
    print("2. DIVERSITY IN FEATURE SPACE")
    print("-" * 70)
    print("   How spread out are the selected samples?")

    var_features = X_censored[variance_selected]
    bald_features = X_censored[bald_selected]

    var_distances = euclidean_distances(var_features)
    bald_distances = euclidean_distances(bald_features)

    # Mean pairwise distance (higher = more diverse)
    var_mean_dist = var_distances[np.triu_indices(budget, k=1)].mean()
    bald_mean_dist = bald_distances[np.triu_indices(budget, k=1)].mean()

    print(f"   Variance mean pairwise distance: {var_mean_dist:.4f}")
    print(f"   BALD mean pairwise distance:     {bald_mean_dist:.4f}")
    print(f"   {'Variance MORE diverse' if var_mean_dist > bald_mean_dist else 'BALD MORE diverse'}")

    # 3. Diversity in Prediction Space
    print("\n" + "-" * 70)
    print("3. DIVERSITY IN PREDICTION SPACE")
    print("-" * 70)
    print("   How spread out are the predicted survival times?")

    var_pred_times = mean_expected_time[variance_selected]
    bald_pred_times = mean_expected_time[bald_selected]

    print(f"   Variance predicted times: mean={var_pred_times.mean():.2f}, std={var_pred_times.std():.2f}")
    print(f"   BALD predicted times:     mean={bald_pred_times.mean():.2f}, std={bald_pred_times.std():.2f}")
    print(f"   {'Variance MORE diverse' if var_pred_times.std() > bald_pred_times.std() else 'BALD MORE diverse'} in predictions")

    # 4. True Time Distribution
    print("\n" + "-" * 70)
    print("4. TRUE SURVIVAL TIME DISTRIBUTION")
    print("-" * 70)
    print("   What are the TRUE survival times of selected samples?")

    var_true_times = y_time_original[variance_selected]
    bald_true_times = y_time_original[bald_selected]
    var_true_events = y_event_original[variance_selected]
    bald_true_events = y_event_original[bald_selected]

    print(f"   Variance TRUE times: mean={var_true_times.mean():.2f}, std={var_true_times.std():.2f}")
    print(f"   BALD TRUE times:     mean={bald_true_times.mean():.2f}, std={bald_true_times.std():.2f}")
    print(f"   Variance TRUE event rate: {var_true_events.mean():.2%}")
    print(f"   BALD TRUE event rate:     {bald_true_events.mean():.2%}")

    # 5. Death Probability Calibration
    print("\n" + "-" * 70)
    print("5. DEATH PROBABILITY IN WINDOW (Model's Estimate)")
    print("-" * 70)

    var_death_prob = death_prob[variance_selected]
    bald_death_prob = death_prob[bald_selected]

    print(f"   Variance death prob: mean={var_death_prob.mean():.4f}")
    print(f"   BALD death prob:     mean={bald_death_prob.mean():.4f}")

    # 6. Prediction Error Analysis
    print("\n" + "-" * 70)
    print("6. PREDICTION UNCERTAINTY vs ACTUAL ERROR")
    print("-" * 70)
    print("   Is variance better at identifying samples where model is WRONG?")

    # Prediction error = |predicted time - true time|
    pred_error = np.abs(mean_expected_time - y_time_original)

    var_pred_error = pred_error[variance_selected]
    bald_pred_error = pred_error[bald_selected]

    print(f"   Variance selects samples with avg error: {var_pred_error.mean():.2f}")
    print(f"   BALD selects samples with avg error:     {bald_pred_error.mean():.2f}")

    # Correlation between scores and actual errors
    corr_var_error = np.corrcoef(time_variance, pred_error)[0, 1]
    corr_bald_error = np.corrcoef(bald_score, pred_error)[0, 1]

    print(f"\n   Correlation(variance, prediction_error): {corr_var_error:.4f}")
    print(f"   Correlation(BALD, prediction_error):     {corr_bald_error:.4f}")
    print(f"   {'Variance' if corr_var_error > corr_bald_error else 'BALD'} better identifies wrong predictions!")

    # 7. Overlap Analysis
    print("\n" + "-" * 70)
    print("7. SELECTION OVERLAP")
    print("-" * 70)

    overlap = len(set(variance_selected) & set(bald_selected))
    var_only = set(variance_selected) - set(bald_selected)
    bald_only = set(bald_selected) - set(variance_selected)

    print(f"   Overlap: {overlap}/{budget} ({100*overlap/budget:.1f}%)")
    print(f"   Variance-only: {len(var_only)}")
    print(f"   BALD-only: {len(bald_only)}")

    # Analyze the difference
    if var_only:
        var_only_idx = list(var_only)
        bald_only_idx = list(bald_only)

        print(f"\n   Variance-only samples:")
        print(f"     - Variance score: {time_variance[var_only_idx].mean():.4f}")
        print(f"     - BALD score: {bald_score[var_only_idx].mean():.4f}")
        print(f"     - TRUE event revealed: {get_revealed_events(var_only_idx)}/{len(var_only_idx)}")
        print(f"     - Prediction error: {pred_error[var_only_idx].mean():.2f}")

        print(f"\n   BALD-only samples:")
        print(f"     - Variance score: {time_variance[bald_only_idx].mean():.4f}")
        print(f"     - BALD score: {bald_score[bald_only_idx].mean():.4f}")
        print(f"     - TRUE event revealed: {get_revealed_events(bald_only_idx)}/{len(bald_only_idx)}")
        print(f"     - Prediction error: {pred_error[bald_only_idx].mean():.2f}")

    # 8. KEY INSIGHT
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
    The variance of expected times measures how much ensemble members
    DISAGREE about the PREDICTED SURVIVAL TIME - the exact quantity
    used to compute C-index.

    BALD measures mutual information about the full distribution, which
    can be high even when predictions agree (e.g., different shapes
    but same mean).

    For C-index improvement, we want samples where:
    1. The model is UNCERTAIN about the expected time (high variance)
    2. The oracle will REVEAL information (high death prob in window)
    3. The revealed info will CHANGE predictions (high variance = disagreement)

    Variance directly captures #1 and #3, while BALD only partially captures #1.
    """)


if __name__ == '__main__':
    main()
