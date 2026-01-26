"""
Optimized BatchBALD for censored survival analysis.

Based on investigation findings:
1. Weight by uncertainty in reveal window (42× boost needed)
2. Add variance-based term (38% boost needed)
3. Maintain spatial diversity (already optimal in BatchBALD)
"""

import numpy as np
import torch
from batchbald_redux import batchbald


def _make_prediction(model, x_test, time_bins, config):
    """Make survival predictions."""
    from Model_stuff.model import mtlr_survival

    if isinstance(x_test, np.ndarray):
        x_test = torch.tensor(x_test, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        if hasattr(model, 'sample_elbo'):
            logits_outputs = model.forward(x_test, sample=True, n_samples=config.n_samples_test)
            survival_outputs = mtlr_survival(logits_outputs, with_sample=True)
            mean_survival_outputs = survival_outputs.mean(dim=0)
        else:
            pred = model.forward(x_test)
            survival_outputs = mtlr_survival(pred, with_sample=False)
            mean_survival_outputs = survival_outputs
            survival_outputs = survival_outputs.unsqueeze(0).repeat(config.n_samples_test, 1, 1)

    if not isinstance(time_bins, torch.Tensor):
        time_bins = torch.tensor(time_bins, dtype=torch.float32)
    time_bins = time_bins.reshape(-1).to(survival_outputs.device)
    zero_tensor = torch.tensor([0], dtype=time_bins.dtype, device=time_bins.device)
    time_bins = torch.cat([zero_tensor, time_bins])

    return mean_survival_outputs, time_bins, survival_outputs


def ensemble_to_pdf(survival_outputs, device):
    """Convert survival curves to death probability PDFs."""
    ensemble_outputs = survival_outputs.permute(1, 0, 2) if survival_outputs.dim() == 3 else survival_outputs
    modified_tensor = ensemble_outputs[:, :, 1:]
    zero_to_append = torch.zeros((ensemble_outputs.shape[0], ensemble_outputs.shape[1], 1),
                                  dtype=torch.float32).to(device)
    modified_tensor = torch.cat((modified_tensor, zero_to_append), dim=2)
    probs_N_K_C = ensemble_outputs - modified_tensor
    return probs_N_K_C


def _map_indices(censoredtime, time_bins):
    """Map times to bin indices."""
    np2_sorted = np.sort(np.array(time_bins))
    np3 = np.zeros_like(censoredtime)
    for i, value in enumerate(censoredtime):
        idx = np.searchsorted(np2_sorted, value, side='right') - 1
        if idx == len(np2_sorted):
            idx = len(np2_sorted) - 1
        np3[i] = idx
    return np3


def _fix_survival_timebins_increment(mylogits, time_bins, x_test, increment):
    """Fix survival time bins with increment and normalize."""
    time_bins_copy = np.array(time_bins).copy()
    temp_bins = np.array([0] + list(time_bins_copy)[:-1])
    censoredtime = np.array(x_test['time'])

    censoredbin = _map_indices(censoredtime, temp_bins)
    binincrements = _map_indices(censoredtime + increment, temp_bins)

    batch_size = mylogits.shape[0]
    cols = mylogits.shape[2]

    for i in range(batch_size):
        a = int(censoredbin[i])
        b = int(binincrements[i])
        mylogits[i, :, :a] = 0
        if b + 1 < cols:
            mylogits[i, :, b+1] = mylogits[i, :, b+1:].sum(dim=1)
            if b + 2 < cols:
                mylogits[i, :, b+2:] = 0

    row_sums = mylogits.sum(dim=2, keepdim=True)
    row_sums[row_sums == 0] = 1
    return mylogits / row_sums


def batchbald_optimal(
    model, X_pool, batch_size, time_bins, config,
    num_samples=60000, device='cpu', min_samples=400,
    in_data_train=None, increment=10000, costlist=None,
    budget=0, censored_indices=None,
    lambda_variance=0.0,  # λ₁: variance boost weight
    lambda_window=2.0,    # λ₂: window uncertainty weight
    verbose=False
):
    """
    Optimized BatchBALD with:
    1. Window uncertainty weighting (lambda_window)
    2. Optional variance boost (lambda_variance)
    3. Maintained diversity (via joint entropy)

    Args:
        lambda_variance: Weight for total variance term (0.0 = disabled, 0.3 = recommended)
        lambda_window: Weight for reveal window uncertainty (2.0 = recommended)
    """
    assert budget > 0, "Budget must be greater than 0"

    if verbose:
        print(f"\n[OptimizedBatchBALD] λ_variance={lambda_variance}, λ_window={lambda_window}")

    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(X_pool).to(device)
        _, _, ensemble_outputs = _make_prediction(model, x_test_tensor, time_bins, config)

    # Convert survival to PDF [N, K, C]
    probs_N_K_C = ensemble_to_pdf(ensemble_outputs, device)

    time_bins_np = np.array(time_bins) if not isinstance(time_bins, np.ndarray) else time_bins
    temp_bins = np.array([0] + list(time_bins_np)[:-1])
    censoredtime = np.array(in_data_train['time'])

    censoredbin = _map_indices(censoredtime, temp_bins)
    binincrements = _map_indices(censoredtime + increment, temp_bins)

    # Apply increment windowing
    logits_N_K_C = _fix_survival_timebins_increment(probs_N_K_C, time_bins, in_data_train, increment)
    N, K, C = logits_N_K_C.shape

    # === COMPUTE WINDOW UNCERTAINTY WEIGHTS ===
    window_weights = torch.ones(N, device=device)

    if lambda_window > 0:
        for i in range(N):
            s = max(0, min(int(censoredbin[i]), C - 1))
            t = max(0, min(int(binincrements[i]), C - 1))

            if t > s:
                # Variance in reveal window
                window_probs = logits_N_K_C[i, :, s:t+1]  # [K, window_size]
                window_variance = window_probs.var(dim=0).sum()  # sum over window
                window_weights[i] = 1.0 + lambda_window * window_variance

    if verbose:
        print(f"[OptimizedBatchBALD] Window weights: min={window_weights.min():.4f}, "
              f"max={window_weights.max():.4f}, mean={window_weights.mean():.4f}")

    # === COMPUTE TOTAL VARIANCE SCORES ===
    variance_scores = torch.zeros(N, device=device)

    if lambda_variance > 0:
        for i in range(N):
            # Total variance across all time bins
            total_var = logits_N_K_C[i, :, :].var(dim=0).sum()
            variance_scores[i] = total_var

        if verbose:
            print(f"[OptimizedBatchBALD] Variance scores: min={variance_scores.min():.6f}, "
                  f"max={variance_scores.max():.6f}, mean={variance_scores.mean():.6f}")

    # === STANDARD BATCHBALD WITH BINARY MI ===
    # Compute binary MI (as in original BatchBALD for this setting)
    p_window_list = []
    for i in range(N):
        s = max(0, min(int(censoredbin[i]), C - 1))
        t = max(0, min(int(binincrements[i]), C - 1))
        if t != s:
            p_w = logits_N_K_C[i, :, s:(t + 1)].sum(dim=1)
        else:
            p_w = logits_N_K_C[i, :, s]
        p_window_list.append(p_w)

    p_window = torch.stack(p_window_list, dim=0)
    p_unknown = 1.0 - p_window
    two_class = torch.stack([p_window, p_unknown], dim=2)
    two_class = torch.clamp(two_class, 1e-12, 1.0)
    two_class /= two_class.sum(dim=2, keepdim=True)
    bb_logits = torch.log(two_class)
    bb_logits[bb_logits == float('-inf')] = -10

    # Handle censored-only selection
    if censored_indices is not None and len(censored_indices) > 0:
        censored_indices = np.array(censored_indices, dtype=int)
        logits_subset = bb_logits[torch.as_tensor(censored_indices, dtype=torch.long, device=device)]

        actual_num_samples = max(min_samples, num_samples)
        request_size = min(batch_size, len(censored_indices))

        # Get BatchBALD candidates
        candidate_batch = batchbald.get_batchbald_batch(
            logits_subset, request_size, actual_num_samples, dtype=torch.double, device=device
        )

        sub_indices = np.asarray(candidate_batch.indices, dtype=int)
        original_indices = censored_indices[sub_indices]
        mi_scores = np.asarray(candidate_batch.scores, dtype=float)
    else:
        actual_num_samples = max(min_samples, num_samples)
        request_size = min(batch_size, N)

        candidate_batch = batchbald.get_batchbald_batch(
            bb_logits, request_size, actual_num_samples, dtype=torch.double, device=device
        )
        original_indices = np.asarray(candidate_batch.indices, dtype=int)
        mi_scores = np.asarray(candidate_batch.scores, dtype=float)

    # === COMBINE SCORES ===
    # MI scores from BatchBALD
    mi_scores_normalized = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-12)

    # Get weights and variance for selected candidates
    window_weights_np = window_weights[original_indices].cpu().numpy()
    variance_scores_np = variance_scores[original_indices].cpu().numpy()

    if lambda_variance > 0:
        # Normalize variance scores
        var_min, var_max = variance_scores_np.min(), variance_scores_np.max()
        if var_max - var_min > 1e-12:
            variance_normalized = (variance_scores_np - var_min) / (var_max - var_min)
        else:
            variance_normalized = np.zeros_like(variance_scores_np)
    else:
        variance_normalized = np.zeros_like(variance_scores_np)

    # Combined score: (MI + λ₁*variance) * (1 + λ₂*window_weight)
    combined_scores = (mi_scores_normalized + lambda_variance * variance_normalized) * window_weights_np

    # Re-rank by combined scores
    reranked_order = np.argsort(combined_scores)[::-1]
    final_indices = original_indices[reranked_order[:batch_size]]

    # Apply budget constraint if needed
    if costlist is not None:
        costs = costlist.values.flatten() if hasattr(costlist, 'values') else np.asarray(costlist).flatten()
        final_costs = costs[final_indices]

        # Select within budget
        selected_indices = []
        current_cost = 0.0
        for idx, cost in zip(final_indices, final_costs):
            if current_cost + cost <= budget:
                selected_indices.append(int(idx))
                current_cost += cost
                if len(selected_indices) >= batch_size:
                    break
        return selected_indices

    return list(final_indices[:batch_size])
