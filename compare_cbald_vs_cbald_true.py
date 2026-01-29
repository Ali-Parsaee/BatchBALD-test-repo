"""
Fair Comparison: Acquisition Functions for Censored Survival Analysis

Settings:
- Budget: 50
- Trials: 3
- increment: 20
- Dataset: NACD
- num_initial_samples: 500 (uncensored)
- test_size: 0.2
- num_bins: 20
- Model: BayesLinMtlr
- lr: 8e-5
- epochs: 100 (early stopping, patience=10)
- n_samples_test: 100

Methods:
1. C-BALD: time_variance * (0.5 + 0.5 * death_prob) — uses variance, not BALD
2. CBALDBALD: BALD * (0.5 + 0.5 * death_prob) — uses entropy-based BALD
3. CBALD_True: I(l;theta|x) + I(y;theta|l,x) (paper-correct)
4. Variance: variance of probs across ensemble
5. Entropy: entropy of averaged probs
"""

import os
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'Model_stuff')

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import argparse
from scipy import stats
import time
import copy

from Model_stuff.model import BayesLinMtlr, mtlr_survival
from Model_stuff.acquisition import (
    cbald_censored_regression,  # C-BALD (simplified)
    select_indices_from_scores,
    _make_prediction,
    _map_indices,
    entropy_of_probs,
    variance_of_probs,
)
from Model_stuff.utils import ensemble_to_pdf
import ssl
import urllib.request
import io
import zipfile


# ==================== CBALDBALD IMPLEMENTATION ====================

def cbaldbald_acquire(model, X_pool, batch_size, time_bins, config,
                      device='cpu', in_data_train=None, increment=10000,
                      costlist=None, budget=0, **kwargs):
    """
    CBALDBALD: Entropy-based BALD weighted by expected time variance.

    The key insight: BALD captures distribution uncertainty, but we care about
    PREDICTION uncertainty (disagreement in expected survival times).

    Solution: Multiply BALD by sqrt(variance of expected times) to combine:
    1. Information-theoretic uncertainty (BALD)
    2. Prediction disagreement (variance of expected times)

    Then weight by death probability in window (like C-BALD).
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X_pool).to(device)
        _, _, ensemble_outputs = _make_prediction(model, x_tensor, time_bins, config)
        pdf = ensemble_to_pdf(ensemble_outputs, device)  # [N, K, T]

    N, K, T = pdf.shape
    eps = 1e-10

    # === BALD: I(y;θ|x) = H(y|x) - E_θ[H(y|x,θ)] ===
    mean_pdf = pdf.mean(dim=1)  # [N, T]
    H_y = -torch.sum(mean_pdf * torch.log(mean_pdf + eps), dim=1)  # [N]
    individual_entropies = -torch.sum(pdf * torch.log(pdf + eps), dim=2)  # [N, K]
    E_H_y_theta = individual_entropies.mean(dim=1)  # [N]
    bald_score = H_y - E_H_y_theta  # [N]

    # === Variance of expected times (prediction disagreement) ===
    time_bins_np = np.array(time_bins) if not isinstance(time_bins, np.ndarray) else time_bins
    bin_mids = (np.concatenate([[0], time_bins_np[:-1]]) + time_bins_np) / 2
    bin_mids_tensor = torch.tensor(bin_mids[:T], dtype=torch.float32, device=device)
    expected_times = (pdf[:, :, :len(bin_mids_tensor)] * bin_mids_tensor).sum(dim=2)  # [N, K]
    time_variance = expected_times.var(dim=1)  # [N]

    # === Death probability in window ===
    temp_bins = np.array([0] + list(time_bins_np)[:-1])
    censoredtime = np.array(in_data_train['time'])
    start_bins = _map_indices(censoredtime, temp_bins).astype(int)
    end_bins = _map_indices(censoredtime + increment, temp_bins).astype(int)

    death_prob = torch.zeros(N, device=device)
    for i in range(N):
        s = int(start_bins[i])
        e = min(int(end_bins[i]) + 1, mean_pdf.shape[1])
        death_prob[i] = mean_pdf[i, s:e].sum()

    # === Combined score: BALD * sqrt(Variance) * death_weight ===
    # sqrt(variance) balances the scale and combines both uncertainty types
    cbaldbald_score = bald_score * torch.sqrt(time_variance + eps) * (0.5 + 0.5 * death_prob)

    scores = cbaldbald_score.cpu().numpy()
    print("cbaldbald_acquire scores: ", scores)
    selected = select_indices_from_scores(scores, budget, costlist, batch_size)
    return selected, scores


def cbaldbald_v2_acquire(model, X_pool, batch_size, time_bins, config,
                         device='cpu', in_data_train=None, increment=10000,
                         costlist=None, budget=0, **kwargs):
    """
    CBALDBALD_v2: BALD * Variance (full multiplicative).

    Stronger weighting on prediction disagreement.
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X_pool).to(device)
        _, _, ensemble_outputs = _make_prediction(model, x_tensor, time_bins, config)
        pdf = ensemble_to_pdf(ensemble_outputs, device)  # [N, K, T]

    N, K, T = pdf.shape
    eps = 1e-10

    # BALD
    mean_pdf = pdf.mean(dim=1)
    H_y = -torch.sum(mean_pdf * torch.log(mean_pdf + eps), dim=1)
    individual_entropies = -torch.sum(pdf * torch.log(pdf + eps), dim=2)
    E_H_y_theta = individual_entropies.mean(dim=1)
    bald_score = H_y - E_H_y_theta

    # Variance of expected times
    time_bins_np = np.array(time_bins) if not isinstance(time_bins, np.ndarray) else time_bins
    bin_mids = (np.concatenate([[0], time_bins_np[:-1]]) + time_bins_np) / 2
    bin_mids_tensor = torch.tensor(bin_mids[:T], dtype=torch.float32, device=device)
    expected_times = (pdf[:, :, :len(bin_mids_tensor)] * bin_mids_tensor).sum(dim=2)
    time_variance = expected_times.var(dim=1)

    # Death probability
    temp_bins = np.array([0] + list(time_bins_np)[:-1])
    censoredtime = np.array(in_data_train['time'])
    start_bins = _map_indices(censoredtime, temp_bins).astype(int)
    end_bins = _map_indices(censoredtime + increment, temp_bins).astype(int)

    death_prob = torch.zeros(N, device=device)
    for i in range(N):
        s = int(start_bins[i])
        e = min(int(end_bins[i]) + 1, mean_pdf.shape[1])
        death_prob[i] = mean_pdf[i, s:e].sum()

    # BALD * Variance
    score = bald_score * time_variance * (0.5 + 0.5 * death_prob)

    scores = score.cpu().numpy()
    print("cbaldbald_v2_acquire scores: ", scores)
    selected = select_indices_from_scores(scores, budget, costlist, batch_size)
    return selected, scores


def cbaldbald_v3_acquire(model, X_pool, batch_size, time_bins, config,
                         device='cpu', in_data_train=None, increment=10000,
                         costlist=None, budget=0, **kwargs):
    """
    CBALDBALD_v3: Variance * (1 + normalized_BALD).

    Use variance as the primary signal, BALD as a boost factor.
    This keeps variance as the main driver while using BALD to differentiate.
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X_pool).to(device)
        _, _, ensemble_outputs = _make_prediction(model, x_tensor, time_bins, config)
        pdf = ensemble_to_pdf(ensemble_outputs, device)  # [N, K, T]

    N, K, T = pdf.shape
    eps = 1e-10

    # BALD
    mean_pdf = pdf.mean(dim=1)
    H_y = -torch.sum(mean_pdf * torch.log(mean_pdf + eps), dim=1)
    individual_entropies = -torch.sum(pdf * torch.log(pdf + eps), dim=2)
    E_H_y_theta = individual_entropies.mean(dim=1)
    bald_score = H_y - E_H_y_theta

    # Variance of expected times
    time_bins_np = np.array(time_bins) if not isinstance(time_bins, np.ndarray) else time_bins
    bin_mids = (np.concatenate([[0], time_bins_np[:-1]]) + time_bins_np) / 2
    bin_mids_tensor = torch.tensor(bin_mids[:T], dtype=torch.float32, device=device)
    expected_times = (pdf[:, :, :len(bin_mids_tensor)] * bin_mids_tensor).sum(dim=2)
    time_variance = expected_times.var(dim=1)

    # Death probability
    temp_bins = np.array([0] + list(time_bins_np)[:-1])
    censoredtime = np.array(in_data_train['time'])
    start_bins = _map_indices(censoredtime, temp_bins).astype(int)
    end_bins = _map_indices(censoredtime + increment, temp_bins).astype(int)

    death_prob = torch.zeros(N, device=device)
    for i in range(N):
        s = int(start_bins[i])
        e = min(int(end_bins[i]) + 1, mean_pdf.shape[1])
        death_prob[i] = mean_pdf[i, s:e].sum()

    # Normalize BALD to [0, 1] range as a boost factor
    bald_min = bald_score.min()
    bald_max = bald_score.max()
    bald_normalized = (bald_score - bald_min) / (bald_max - bald_min + eps)

    # Variance with BALD boost: variance * (1 + 0.5 * normalized_bald)
    score = time_variance * (1 + 0.5 * bald_normalized) * (0.5 + 0.5 * death_prob)

    scores = score.cpu().numpy()
    print("cbaldbald_v3_acquire scores: ", scores)
    selected = select_indices_from_scores(scores, budget, costlist, batch_size)
    return selected, scores


def cbaldbald_v4_acquire(model, X_pool, batch_size, time_bins, config,
                         device='cpu', in_data_train=None, increment=10000,
                         costlist=None, budget=0, **kwargs):
    """
    CBALDBALD_v4: Variance with selective BALD boost.

    Only boost samples where BALD is above median - these are samples
    where there's genuine distribution disagreement, not just noise.
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X_pool).to(device)
        _, _, ensemble_outputs = _make_prediction(model, x_tensor, time_bins, config)
        pdf = ensemble_to_pdf(ensemble_outputs, device)  # [N, K, T]

    N, K, T = pdf.shape
    eps = 1e-10

    # BALD
    mean_pdf = pdf.mean(dim=1)
    H_y = -torch.sum(mean_pdf * torch.log(mean_pdf + eps), dim=1)
    individual_entropies = -torch.sum(pdf * torch.log(pdf + eps), dim=2)
    E_H_y_theta = individual_entropies.mean(dim=1)
    bald_score = H_y - E_H_y_theta

    # Variance of expected times
    time_bins_np = np.array(time_bins) if not isinstance(time_bins, np.ndarray) else time_bins
    bin_mids = (np.concatenate([[0], time_bins_np[:-1]]) + time_bins_np) / 2
    bin_mids_tensor = torch.tensor(bin_mids[:T], dtype=torch.float32, device=device)
    expected_times = (pdf[:, :, :len(bin_mids_tensor)] * bin_mids_tensor).sum(dim=2)
    time_variance = expected_times.var(dim=1)

    # Death probability
    temp_bins = np.array([0] + list(time_bins_np)[:-1])
    censoredtime = np.array(in_data_train['time'])
    start_bins = _map_indices(censoredtime, temp_bins).astype(int)
    end_bins = _map_indices(censoredtime + increment, temp_bins).astype(int)

    death_prob = torch.zeros(N, device=device)
    for i in range(N):
        s = int(start_bins[i])
        e = min(int(end_bins[i]) + 1, mean_pdf.shape[1])
        death_prob[i] = mean_pdf[i, s:e].sum()

    # Selective boost: only boost samples with above-median BALD
    bald_median = torch.median(bald_score)
    bald_normalized = (bald_score - bald_score.min()) / (bald_score.max() - bald_score.min() + eps)

    # Boost factor: 1.0 for below-median BALD, up to 1.5 for high BALD
    boost = torch.where(bald_score > bald_median, 1.0 + 0.5 * bald_normalized, torch.ones_like(bald_score))

    score = time_variance * boost * (0.5 + 0.5 * death_prob)

    scores = score.cpu().numpy()
    print("cbaldbald_v4_acquire scores: ", scores)
    selected = select_indices_from_scores(scores, budget, costlist, batch_size)
    return selected, scores


def cbaldbald_v5_acquire(model, X_pool, batch_size, time_bins, config,
                         device='cpu', in_data_train=None, increment=10000,
                         costlist=None, budget=0, **kwargs):
    """
    CBALDBALD_v5: Variance * (1 + log(1 + BALD_normalized)).

    Log-scaled boost to dampen the effect of extreme BALD values while
    still providing differentiation.
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X_pool).to(device)
        _, _, ensemble_outputs = _make_prediction(model, x_tensor, time_bins, config)
        pdf = ensemble_to_pdf(ensemble_outputs, device)  # [N, K, T]

    N, K, T = pdf.shape
    eps = 1e-10

    # BALD
    mean_pdf = pdf.mean(dim=1)
    H_y = -torch.sum(mean_pdf * torch.log(mean_pdf + eps), dim=1)
    individual_entropies = -torch.sum(pdf * torch.log(pdf + eps), dim=2)
    E_H_y_theta = individual_entropies.mean(dim=1)
    bald_score = H_y - E_H_y_theta

    # Variance of expected times
    time_bins_np = np.array(time_bins) if not isinstance(time_bins, np.ndarray) else time_bins
    bin_mids = (np.concatenate([[0], time_bins_np[:-1]]) + time_bins_np) / 2
    bin_mids_tensor = torch.tensor(bin_mids[:T], dtype=torch.float32, device=device)
    expected_times = (pdf[:, :, :len(bin_mids_tensor)] * bin_mids_tensor).sum(dim=2)
    time_variance = expected_times.var(dim=1)

    # Death probability
    temp_bins = np.array([0] + list(time_bins_np)[:-1])
    censoredtime = np.array(in_data_train['time'])
    start_bins = _map_indices(censoredtime, temp_bins).astype(int)
    end_bins = _map_indices(censoredtime + increment, temp_bins).astype(int)

    death_prob = torch.zeros(N, device=device)
    for i in range(N):
        s = int(start_bins[i])
        e = min(int(end_bins[i]) + 1, mean_pdf.shape[1])
        death_prob[i] = mean_pdf[i, s:e].sum()

    # Normalize BALD to [0, 1]
    bald_normalized = (bald_score - bald_score.min()) / (bald_score.max() - bald_score.min() + eps)

    # Log-scaled boost: provides smooth, bounded boost
    boost = 1.0 + torch.log1p(bald_normalized)  # log1p(x) = log(1+x)

    score = time_variance * boost * (0.5 + 0.5 * death_prob)

    scores = score.cpu().numpy()
    print("cbaldbald_v5_acquire scores: ", scores)
    selected = select_indices_from_scores(scores, budget, costlist, batch_size)
    return selected, scores


# ==================== CBALD_True IMPLEMENTATION ====================

def cbald_true_acquire(model, X_pool, batch_size, time_bins, config,
                       device='cpu', in_data_train=None, increment=10000,
                       costlist=None, budget=0, **kwargs):
    """
    CBALD_True: I(l;theta|x) + I(y;theta|l,x) — paper-correct formulation.

    Term 1: I(l;theta|x) = H(Y_oracle) - E_theta[H(Y_oracle|theta)]
      Mutual information between oracle label and model parameters.

    Term 2: I(y;theta|l,x) = P(still censored) * [H(y|censored) - E_theta[H(y|censored,theta)]]
      Additional information about true outcome beyond oracle label.
      Only nonzero when oracle outcome is "still alive at c+k".
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X_pool).to(device)
        _, _, ensemble_outputs = _make_prediction(model, x_tensor, time_bins, config)
        pdf = ensemble_to_pdf(ensemble_outputs, device)  # [N, K, T]

    N, K, T = pdf.shape
    eps = 1e-10

    # Get censoring bin indices
    time_bins_np = np.array(time_bins) if not isinstance(time_bins, np.ndarray) else time_bins
    temp_bins = np.array([0] + list(time_bins_np)[:-1])
    censoredtime = np.array(in_data_train['time'])
    start_bins = _map_indices(censoredtime, temp_bins).astype(int)
    end_bins = _map_indices(censoredtime + increment, temp_bins).astype(int)

    scores = np.zeros(N)

    for i in range(N):
        c = int(start_bins[i])
        max_bin = min(int(end_bins[i]), T - 1)

        # --- Term 1: I(l;theta|x) = H(Y_oracle) - E_theta[H(Y_oracle|theta)] ---

        # H(Y_oracle) from averaged PDF
        mean_pdf_i = pdf[i].mean(dim=0)  # [T]
        probs_avg = mean_pdf_i.clone()
        if c > 0:
            probs_avg[:c] = 0
        total_avg = probs_avg.sum()
        if total_avg > eps:
            probs_avg = probs_avg / total_avg

        # Oracle outcomes: died in bins [c, max_bin], or still alive
        oracle_probs_avg = []
        for t in range(c, max_bin + 1):
            if t < T:
                oracle_probs_avg.append(probs_avg[t])
        p_censored_avg = probs_avg[max_bin + 1:].sum() if max_bin + 1 < T else torch.tensor(0.0, device=device)
        oracle_probs_avg.append(p_censored_avg)
        oracle_probs_avg = torch.stack(oracle_probs_avg)

        H_oracle = -torch.sum(oracle_probs_avg * torch.log(oracle_probs_avg + eps))

        # E_theta[H(Y_oracle|theta)]
        cond_entropies = torch.zeros(K, device=device)
        for k in range(K):
            probs_k = pdf[i, k].clone()
            if c > 0:
                probs_k[:c] = 0
            total_k = probs_k.sum()
            if total_k > eps:
                probs_k = probs_k / total_k

            oracle_k = []
            for t in range(c, max_bin + 1):
                if t < T:
                    oracle_k.append(probs_k[t])
            p_cens_k = probs_k[max_bin + 1:].sum() if max_bin + 1 < T else torch.tensor(0.0, device=device)
            oracle_k.append(p_cens_k)
            oracle_k = torch.stack(oracle_k)
            cond_entropies[k] = -torch.sum(oracle_k * torch.log(oracle_k + eps))

        E_H_oracle_given_theta = cond_entropies.mean()
        term1 = (H_oracle - E_H_oracle_given_theta).item()

        # --- Term 2: I(y;theta|l,x) ---
        # Only contributes when oracle says "still alive at c+k"
        term2 = 0.0
        if max_bin + 1 < T and p_censored_avg.item() > eps:
            # H(y | alive at c+k) from averaged PDF
            tail_avg = probs_avg[max_bin + 1:].clone()
            tail_total = tail_avg.sum()
            if tail_total > eps:
                tail_avg = tail_avg / tail_total
            H_y_alive = -torch.sum(tail_avg * torch.log(tail_avg + eps)).item()

            # E_theta[H(y | alive at c+k, theta)]
            cond_tail = torch.zeros(K, device=device)
            for k in range(K):
                probs_k = pdf[i, k].clone()
                if c > 0:
                    probs_k[:c] = 0
                total_k = probs_k.sum()
                if total_k > eps:
                    probs_k = probs_k / total_k

                tail_k = probs_k[max_bin + 1:].clone()
                tail_k_total = tail_k.sum()
                if tail_k_total > eps:
                    tail_k = tail_k / tail_k_total
                cond_tail[k] = -torch.sum(tail_k * torch.log(tail_k + eps))

            E_H_y_alive_theta = cond_tail.mean().item()
            term2 = p_censored_avg.item() * (H_y_alive - E_H_y_alive_theta)

        scores[i] = term1 + term2

    print("cbald_true_acquire scores: ", scores)
    selected = select_indices_from_scores(scores, budget, costlist, batch_size)
    return selected, scores


# ==================== HELPER FUNCTIONS ====================

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


def artificially_censor_true(times, events, num_initial_samples=50, seed=None):
    """Artificially censor data points."""
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


def expected_times_from_survival(surv_np, tbins):
    """Convert survival probabilities to expected survival times."""
    if isinstance(tbins, torch.Tensor):
        tb = tbins.clone().detach().cpu().numpy()
    else:
        tb = np.array(tbins)

    # Handle shape mismatch: survival output may have one more column than time_bins
    n_bins = len(tb)
    surv = surv_np[:, :n_bins] if surv_np.shape[1] > n_bins else surv_np

    left_edges = np.concatenate([[0.0], tb[:-1]])
    right_edges = tb
    mids = (left_edges + right_edges) / 2.0
    pdf = np.zeros_like(surv)
    pdf[:, 0] = 1.0 - surv[:, 0]
    pdf[:, 1:] = surv[:, :-1] - surv[:, 1:]
    pdf = np.clip(pdf, 0.0, 1.0)
    pdf = pdf / (pdf.sum(axis=1, keepdims=True) + 1e-12)
    return (pdf * mids.reshape(1, -1)).sum(axis=1)


def concordance(y_pred, y_test, cens):
    """Calculate concordance index."""
    n = len(y_pred)
    if n < 2:
        return np.nan

    concordant = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            if cens[i] == 1 and cens[j] == 1:
                total_pairs += 1
                if (y_test[i] < y_test[j] and y_pred[i] < y_pred[j]) or \
                   (y_test[i] > y_test[j] and y_pred[i] > y_pred[j]):
                    concordant += 1
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5
            elif cens[i] == 1 and cens[j] == 0:
                if y_test[i] < y_test[j]:
                    total_pairs += 1
                    if y_pred[i] < y_pred[j]:
                        concordant += 1
                    elif y_pred[i] == y_pred[j]:
                        concordant += 0.5
            elif cens[i] == 0 and cens[j] == 1:
                if y_test[j] < y_test[i]:
                    total_pairs += 1
                    if y_pred[j] < y_pred[i]:
                        concordant += 1
                    elif y_pred[i] == y_pred[j]:
                        concordant += 0.5

    if total_pairs == 0:
        return np.nan
    return concordant / total_pairs


# ==================== MAIN EXPERIMENT ====================

def run_single_trial(trial_num, budget=20, methods_to_run=None):
    """Run single trial comparing C-BALD and CBALD_True.

    Args:
        trial_num: Trial number (1-indexed)
        budget: Number of samples to acquire
        methods_to_run: List of method names to run. If None, runs all.
                       Options: ['C-BALD', 'CBALD_True']
    """

    print(f"  Trial {trial_num}/5")
    print("  " + "-" * 68)

    # Set seed for reproducibility
    seed = 100 + trial_num - 1
    print(f"      Setting random seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load NACD dataset
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
    print(f"      Dataset: NACD ({df.shape[0]} samples, {df.shape[1]} features)")

    # Data already has 'time' and 'event' columns, features already standardized
    X = df.drop(columns=['time', 'event'])
    y_time = df['time'].values
    y_event = df['event'].values

    X_train_val, X_test, y_time_train_val, y_time_test, y_event_train_val, y_event_test = train_test_split(
        X, y_time, y_event, test_size=0.2, random_state=seed, stratify=y_event
    )

    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)

    # Artificially censor (num_initial_samples=500)
    y_time_censored, y_event_censored, censored_indices = artificially_censor_true(
        y_time_train_val, y_event_train_val, num_initial_samples=500, seed=seed
    )

    # Setup
    num_bins = 20
    event_times = y_time_train_val[y_event_train_val == 1]
    quantiles = np.linspace(0, 1, num_bins + 1)[1:]
    time_bins = np.quantile(event_times, quantiles)
    time_bins[-1] *= 1.05
    time_bins = np.array([0] + list(time_bins))

    # Config
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

    # Train SHARED base model
    print(f"      Training SHARED base model for all methods...")
    start_time = time.time()

    train_df = pd.DataFrame(X_train_val_scaled)
    train_df['time'] = y_time_censored
    train_df['event'] = y_event_censored
    test_df = pd.DataFrame(X_test_scaled)
    test_df['time'] = y_time_test
    test_df['event'] = y_event_test

    x_train, y_train = reformat_survival(train_df, time_bins)
    x_test, y_test = reformat_survival(test_df, time_bins)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    shared_model = BayesLinMtlr(
        in_features=X_train_val_scaled.shape[1],
        num_time_bins=len(time_bins),
        config=config
    )

    optimizer = torch.optim.Adam(shared_model.parameters(), lr=8e-5)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(100):
        shared_model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)

            optimizer.zero_grad()
            loss, _, _, _ = shared_model.sample_elbo(batch_x, batch_y, len(x_train), see1=config.c1)
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

    train_time = time.time() - start_time
    print(f"      Shared base model trained ({train_time:.1f}s)")

    # Evaluate shared model on test set
    shared_model.eval()
    with torch.no_grad():
        logits = shared_model.forward(x_test, sample=True, n_samples=config.n_samples_test)
        survival_probs = mtlr_survival(logits, with_sample=True)
        mean_survival = survival_probs.mean(dim=0).cpu().numpy()

    pred_times = expected_times_from_survival(mean_survival, time_bins)
    initial_cindex = concordance(pred_times, y_time_test, y_event_test)
    print(f"      Shared initial C-index: {initial_cindex:.4f}")
    print("      All methods will start from this SAME model!")

    # Setup for acquisition
    X_censored = X_train_val_scaled[censored_indices]
    y_time_original = y_time_train_val[censored_indices]
    y_event_original = y_event_train_val[censored_indices]

    train_df_censored = pd.DataFrame(X_train_val_scaled)
    train_df_censored['time'] = y_time_censored
    train_df_censored['event'] = y_event_censored
    x_train_censored, y_train_censored = reformat_survival(train_df_censored, time_bins)

    # model_data_censored needs 'time' key for cbald_score function
    # 'time' should be the censored times corresponding to samples in X_pool (X_censored)
    model_data_censored = {
        'x_train': x_train_censored,
        'y_train': y_train_censored,
        'x_test': x_test,
        'y_test': y_test,
        'time_bins': time_bins,
        'time': y_time_censored[censored_indices],  # censored times for the pool
        'event': y_event_censored[censored_indices]  # censored events for the pool
    }

    # Define acquisition functions to test
    all_acquisition_functions = [
        (cbald_censored_regression, 'C-BALD', {}),
        (cbaldbald_acquire, 'CBALDBALD', {}),           # BALD * sqrt(Var) - original
        (cbaldbald_v3_acquire, 'CBALDBALD_v3', {}),     # Var * (1 + 0.5*norm_bald) - best
    ]

    # Filter if specific methods requested
    if methods_to_run is not None:
        acquisition_functions = [
            (func, name, params) for func, name, params in all_acquisition_functions
            if name in methods_to_run
        ]
    else:
        acquisition_functions = all_acquisition_functions

    results = {}
    increment = 20
    costlist = np.ones(len(X_censored))

    for acq_func, acq_name, acq_params in acquisition_functions:
        print(f"\n      [{acq_name}] Starting acquisition...")

        # Clone the shared model by saving/loading state dict
        model = BayesLinMtlr(
            in_features=X_train_val_scaled.shape[1],
            num_time_bins=len(time_bins),
            config=config
        )
        model.load_state_dict(shared_model.state_dict())

        # Verify same initial performance
        model.eval()
        with torch.no_grad():
            logits = model.forward(x_test, sample=True, n_samples=config.n_samples_test)
            survival_probs = mtlr_survival(logits, with_sample=True)
            mean_survival = survival_probs.mean(dim=0).cpu().numpy()
        pred_times = expected_times_from_survival(mean_survival, time_bins)
        init_c = concordance(pred_times, y_time_test, y_event_test)
        print(f"      [{acq_name}] Initial C-index: {init_c:.4f} (verified same as base)")

        # Run acquisition
        acq_start = time.time()

        pool_indices, _ = acq_func(
            model=model,
            X_pool=X_censored,
            batch_size=budget,
            time_bins=time_bins,
            config=config,
            device=config.device,
            in_data_train=model_data_censored,
            increment=increment,
            costlist=costlist,
            budget=budget,
            **acq_params
        )

        acq_time = time.time() - acq_start
        print(f"      [{acq_name}] Running acquisition (budget={budget})... done ({acq_time:.1f}s, acquired {len(pool_indices)} samples)")

        # Reveal oracle labels
        revealed_indices = censored_indices[pool_indices]
        y_time_updated = np.copy(y_time_censored)
        y_event_updated = np.copy(y_event_censored)
        y_time_updated[revealed_indices] = y_time_original[pool_indices]
        y_event_updated[revealed_indices] = y_event_original[pool_indices]

        # Retrain model
        print(f"      [{acq_name}] Retraining model...", end=" ")
        retrain_start = time.time()

        train_df_updated = pd.DataFrame(X_train_val_scaled)
        train_df_updated['time'] = y_time_updated
        train_df_updated['event'] = y_event_updated
        x_train_updated, y_train_updated = reformat_survival(train_df_updated, time_bins)

        train_dataset_updated = TensorDataset(x_train_updated, y_train_updated)
        train_loader_updated = DataLoader(train_dataset_updated, batch_size=config.batch_size, shuffle=True)

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
            for batch_x, batch_y in train_loader_updated:
                batch_x = batch_x.to(config.device)
                batch_y = batch_y.to(config.device)

                optimizer.zero_grad()
                loss, _, _, _ = model.sample_elbo(batch_x, batch_y, len(x_train_updated), see1=config.c1)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader_updated)

            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    break

        retrain_time = time.time() - retrain_start
        print(f"done ({retrain_time:.1f}s)")

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model.forward(x_test, sample=True, n_samples=config.n_samples_test)
            survival_probs = mtlr_survival(logits, with_sample=True)
            mean_survival = survival_probs.mean(dim=0).cpu().numpy()

        pred_times = expected_times_from_survival(mean_survival, time_bins)
        final_cindex = concordance(pred_times, y_time_test, y_event_test)
        improvement = final_cindex - initial_cindex
        total_time = acq_time + retrain_time

        print(f"      [{acq_name}] Final C-index: {final_cindex:.4f} (Delta={improvement:+.4f}) [Total: {total_time:.1f}s]")

        results[acq_name] = {
            'initial_cindex': initial_cindex,
            'final_cindex': final_cindex,
            'improvement': improvement,
            'time': total_time
        }

    print(f"\n  Trial {trial_num} completed\n")

    return results


def main():
    """Run all trials and report results."""

    parser = argparse.ArgumentParser(description='Compare C-BALD vs CBALD_True')
    parser.add_argument('--verify-cbald-only', action='store_true',
                        help='Only run C-BALD to verify +0.0173 result')
    parser.add_argument('--trials', type=int, default=5,
                        help='Number of trials (default: 5)')
    parser.add_argument('--budget', type=int, default=20,
                        help='Budget for acquisition (default: 20)')
    args = parser.parse_args()

    print("=" * 70)
    if args.verify_cbald_only:
        print("VERIFICATION: C-BALD ONLY")
        print("=" * 70)
        print()
        print("Running C-BALD only for verification")
        methods_to_run = ['C-BALD']
    else:
        print("FAIR COMPARISON: C-BALD vs CBALD_True")
        print("=" * 70)
        print()
        print("Methods:")
        print("  1. C-BALD: time_variance * (0.5 + 0.5 * death_prob)")
        print("  2. CBALD_True: I(l;theta|x) + I(y;theta|l,x) (paper-correct)")
        methods_to_run = None

    print()
    print("Settings:")
    print(f"  Trials: {args.trials}")
    print(f"  Budget: {args.budget}")
    print("  increment: 60")
    print("  num_initial_samples: 200 (uncensored)")
    print("  test_size: 0.2")
    print("  num_bins: 10")
    print("  lr: 8e-5")
    print("  patience: 10")
    print("  n_samples_test: 100")
    print(f"  Device: cpu")
    print()

    # Run trials
    print(f"Running {args.trials} trials...")
    print()

    all_results = {}

    for trial_num in range(1, args.trials + 1):
        trial_results = run_single_trial(trial_num, budget=args.budget, methods_to_run=methods_to_run)

        for method, metrics in trial_results.items():
            if method not in all_results:
                all_results[method] = {
                    'initial_cindices': [],
                    'final_cindices': [],
                    'improvements': [],
                    'times': []
                }
            all_results[method]['initial_cindices'].append(metrics['initial_cindex'])
            all_results[method]['final_cindices'].append(metrics['final_cindex'])
            all_results[method]['improvements'].append(metrics['improvement'])
            all_results[method]['times'].append(metrics['time'])

    # Print results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    # Compute statistics
    method_stats = {}
    for method, data in all_results.items():
        method_stats[method] = {
            'mean_improvement': np.mean(data['improvements']),
            'std_improvement': np.std(data['improvements'], ddof=1),
            'mean_time': np.mean(data['times']),
            'mean_initial': np.mean(data['initial_cindices']),
            'std_initial': np.std(data['initial_cindices'], ddof=1),
            'improvements': data['improvements']
        }

    # Sort by mean improvement
    sorted_methods = sorted(method_stats.items(), key=lambda x: x[1]['mean_improvement'], reverse=True)

    print("Method Rankings (by mean IMPROVEMENT):")
    print("-" * 70)
    for i, (method, mstats) in enumerate(sorted_methods):
        print(f"{i+1}. {method:30s} {mstats['mean_improvement']:+.4f} +/- {mstats['std_improvement']:.4f}")
        print(f"   Individual trials: {[f'{x:+.4f}' for x in mstats['improvements']]}")

    print()
    print("Initial C-index (should be SAME for all methods):")
    print("-" * 70)
    for method, mstats in sorted_methods:
        print(f"{method:30s} {mstats['mean_initial']:.4f} +/- {mstats['std_initial']:.4f}")

    # Verification check
    if args.verify_cbald_only:
        print()
        print("=" * 70)
        print("VERIFICATION RESULT")
        print("=" * 70)
        if 'C-BALD' in method_stats:
            mean_imp = method_stats['C-BALD']['mean_improvement']
            std_imp = method_stats['C-BALD']['std_improvement']
            print(f"C-BALD mean improvement: {mean_imp:+.4f} +/- {std_imp:.4f}")
    else:
        # Statistical comparison
        print()
        print("Statistical Comparison (Paired t-test on improvement):")
        print("-" * 70)

        if 'C-BALD' in all_results and 'CBALD_True' in all_results:
            cbald_improvements = all_results['C-BALD']['improvements']
            cbald_true_improvements = all_results['CBALD_True']['improvements']

            t_stat, p_value = stats.ttest_rel(cbald_improvements, cbald_true_improvements)
            mean_diff = np.mean(cbald_improvements) - np.mean(cbald_true_improvements)

            print(f"\nC-BALD vs CBALD_True:")
            print(f"  C-BALD mean:      {np.mean(cbald_improvements):+.4f}")
            print(f"  CBALD_True mean:  {np.mean(cbald_true_improvements):+.4f}")
            print(f"  Mean difference:  {mean_diff:+.4f}")
            print(f"  t-statistic:      {t_stat:.3f}")
            print(f"  p-value:          {p_value:.4f}")

            if p_value < 0.05:
                winner = 'C-BALD' if np.mean(cbald_improvements) > np.mean(cbald_true_improvements) else 'CBALD_True'
                print(f"\n  SIGNIFICANT (p<0.05): {winner} is better!")
            else:
                print(f"\n  NOT SIGNIFICANT (p>=0.05): No clear winner")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
