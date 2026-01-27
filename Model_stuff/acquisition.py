"""
Acquisition functions for active learning in survival analysis.
"""

import numpy as np
import pandas as pd
import torch
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from batchbald_redux import batchbald

from utils import make_prediction, ensemble_to_pdf

def _make_prediction(model, x_test, time_bins, args):
    """Make survival predictions (local version to avoid circular import)."""
    from model import mtlr_survival
    
    if isinstance(x_test, np.ndarray):
        x_test = torch.tensor(x_test, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'sample_elbo'):
            logits_outputs = model.forward(x_test, sample=True, n_samples=args.n_samples_test)
            survival_outputs = mtlr_survival(logits_outputs, with_sample=True)
            mean_survival_outputs = survival_outputs.mean(dim=0)
        else:
            pred = model.forward(x_test)
            survival_outputs = mtlr_survival(pred, with_sample=False)
            mean_survival_outputs = survival_outputs
            survival_outputs = survival_outputs.unsqueeze(0).repeat(args.n_samples_test, 1, 1)

    if not isinstance(time_bins, torch.Tensor):
        time_bins = torch.tensor(time_bins, dtype=torch.float32)
    time_bins = time_bins.reshape(-1).to(survival_outputs.device)
    zero_tensor = torch.tensor([0], dtype=time_bins.dtype, device=time_bins.device)
    time_bins = torch.cat([zero_tensor, time_bins])
    
    return mean_survival_outputs, time_bins, survival_outputs


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


# =============================================================================
# BATCHBALD ACQUISITION
# =============================================================================

def bald_score(model,X_pool,time_bins,args,device,in_data_train=None,increment=0,censored_indices=None,preselect_multiplier=10,costlist=None,budget=None,seed=0,verbose=True):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X_pool).to(device)
        survival_outputs, _, ensemble_outputs = make_prediction(model, x_tensor, time_bins, args)
        pdf = ensemble_to_pdf(ensemble_outputs, device)
        p_mean = pdf.mean(dim=1)  # [N,T]
        H_mean = -(p_mean * torch.log(p_mean + 1e-12)).sum(dim=1)
        H_each = -(pdf * torch.log(pdf + 1e-12)).sum(dim=2)
        bald_score = (H_mean - H_each.mean(dim=1)).detach().cpu().numpy()
    return bald_score


def test_bald_score(
    model,
    X_pool,
    batch_size,
    time_bins,
    config,
    device="cpu",
    in_data_train=None,
    increment=0,
    censored_indices=None,
    preselect_multiplier=10,
    costlist=None,
    budget=None,
    seed=0,
    verbose=True,
):

    score = bald_score(model, X_pool, time_bins, config, device, in_data_train, increment, censored_indices, preselect_multiplier, costlist, budget, seed, verbose)
    print("after bald_score: ", score)
    return select_indices_from_scores(score, budget, costlist, batch_size), score



def new_bald_score(model, X, time_bins, args, device, eps=1e-12, normalize="logT"):
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float32, device=device)
        _, _, ensemble_outputs = make_prediction(model, x, time_bins, args)
        pdf = ensemble_to_pdf(ensemble_outputs, device)  # [N, K, T]

        # minimal stability, but less flattening than clamp+renorm
        pdf = pdf.clamp_min(eps)

        p_mean = pdf.mean(dim=1)  # [N, T]

        H_mean = -(p_mean * torch.log(p_mean)).sum(dim=-1)       # [N]
        H_each = -(pdf * torch.log(pdf)).sum(dim=-1).mean(dim=1) # [N]

        mi = H_mean - H_each                                     # [N]

        score = mi / (H_mean + eps)                               # relative BALD
        return score.detach().cpu().numpy()



def batchbald_acquire_budget_DEBUG(
    model,
    X_pool,
    batch_size,
    time_bins,
    config,
    device="cpu",
    in_data_train=None,
    increment=0,
    censored_indices=None,
    preselect_multiplier=10,
    costlist=None,
    budget=None,
    seed=0,
    verbose=True,
):
    if not verbose:
        _print = lambda *a, **k: None
    else:
        _print = print

    _print("\n==============================")
    _print(" START BatchBALD DEBUG RUN ")
    _print("==============================")

    assert in_data_train is not None
    assert "time" in in_data_train

    rng = np.random.default_rng(seed)
    N = len(X_pool)


    reveal_weight = 1    # 0 disables reveal-likelihood term
    reveal_beta = 0      # exponent if using product-style combination
    density_weight = 0
    density_knn_k=200
    combine_mode = "product"
    outwin_alpha = 1
    outwin_min_weight = 1
    inspect_k=5


    _print(f"[INFO] Pool size N = {N}")
    _print(f"[INFO] batch_size = {batch_size}")
    _print(f"[INFO] increment/probe depth = {increment}")
    _print(f"[INFO] density_weight = {density_weight}, combine_mode = {combine_mode}")
    _print(f"[INFO] reveal_weight = {reveal_weight}, reveal_beta = {reveal_beta}")
    _print(f"[INFO] outwin_alpha = {outwin_alpha}, outwin_min_weight = {outwin_min_weight}")

    # ---------------------------------------------------
    # 0) DATA SEMANTICS PRINTS (your special setting)
    # ---------------------------------------------------
    if True:
        cols = list(getattr(in_data_train, "columns", []))
        if cols:
            _print(f"\n[DATA] in_data_train columns (first 40): {cols[:40]}")
            if "event" in in_data_train:
                vals = np.unique(np.asarray(in_data_train["event"]))
                _print(f"[DATA] 'event' unique values: {vals[:20]}  (in your setting, expected all 0)")
            t = np.asarray(in_data_train["time"], dtype=float)
            _print("[DATA] Pool/probe-time summary (in_data_train['time']):")
            _print(f"  min={t.min():.4f}, p25={np.quantile(t,0.25):.4f}, median={np.median(t):.4f}, p75={np.quantile(t,0.75):.4f}, max={t.max():.4f}")
            _print(f"  frac <= 0: {(t<=0).mean():.4f}")
            _print(f"  frac > max_time_bin: {(t>float(np.max(time_bins)) ).mean():.4f}")
        else:
            _print("\n[DATA] in_data_train has no .columns attribute (not a DataFrame?)")

    # ---------------------------------------------------
    # 1) MODEL PREDICTION
    # ---------------------------------------------------
    if True:
        _print("\n[STEP 1] Running model prediction...")
        model.eval()
        with torch.no_grad():
            x_test = torch.as_tensor(np.asarray(X_pool), dtype=torch.float32, device=device)
            mean_surv, tb_with0, survival_outputs = make_prediction(model, x_test, time_bins, config)

        _print(f"[SHAPE] survival_outputs: {tuple(survival_outputs.shape)} (K, N, T)")
        _print(f"[SHAPE] mean_surv: {tuple(mean_surv.shape)}")
        _print(f"[INFO] time_bins (with 0 prepended): {tb_with0.detach().cpu().numpy()} ...")

        if survival_outputs.dim() != 3:
            raise ValueError(f"Expected survival_outputs to be 3D [K,N,T], got {tuple(survival_outputs.shape)}")

        K, N_check, T = survival_outputs.shape
        assert N_check == N

        _print("survival_outputs shape: ", survival_outputs.shape)
        _print(survival_outputs[0])
        _print(survival_outputs[0].shape)
        _print(survival_outputs[0, 0])
        _print(survival_outputs[0, 0].shape)
        _print(survival_outputs[0, 0, 0])
        _print(survival_outputs[0, 0, 0])

        _print("tb_with0 = ", tb_with0)

    # ---------------------------------------------------
    # 2) SURVIVAL → DEATH PDF
    # ---------------------------------------------------
    if True:
        _print("\n[STEP 2] Converting survival → death PDF")

        pdf = ensemble_to_pdf(survival_outputs, device)

        _print("pdf shape: ", pdf.shape)
        _print(pdf[0])
        _print(pdf[0].shape)
        _print(pdf[0, 0])
        _print(pdf[0, 0].shape)
        _print(pdf[0, 0, 0])
        _print(pdf[0, 0, 0])

        _print(f"[CHECK] PDF row sums (mean over N,K): {pdf.sum(dim=2).mean().item():.6f}")
        _print(f"[CHECK] PDF min={pdf.min().item():.3e}, max={pdf.max().item():.3e}")

    # ---------------------------------------------------
    # 3) APPLY CENSOR + INCREMENT COLLAPSE (your helper)
    # ---------------------------------------------------
    if True:
        _print("\n[STEP 3] Applying censor + increment window collapse (via _fix_survival_timebins_increment)")

        _print("time_bins: ", time_bins)
        _print(pdf[0,0])
        _print(pdf[1, 0])
        _print(pdf[2, 0])
        _print(pdf[3, 0])
        _print("in_data_train: ", in_data_train.loc[0,'time'])
        _print("in_data_train1: ", in_data_train.loc[1,'time'])
        _print("in_data_train2: ", in_data_train.loc[2,'time'])
        _print("in_data_train3: ", in_data_train.loc[3,'time'])

        pdf = _fix_survival_timebins_increment(pdf, time_bins, in_data_train, increment)
        _print(pdf[0,0])
        _print(pdf[1, 0])
        _print(pdf[2, 0])
        _print(pdf[3, 0])

        _print("pdf shape 2: ", pdf.shape)
        _print(pdf[0])
        _print(pdf[0].shape)
        _print(pdf[0, 0])
        _print(pdf[0, 0].shape)
        _print(pdf[0, 0, 0])
        _print(pdf[0, 0, 0])

        row_sums = pdf.sum(dim=2)
        _print(f"[CHECK] After fix: row sum mean={row_sums.mean().item():.6f}, min={row_sums.min().item():.6f}, max={row_sums.max().item():.6f}")

    # ---------------------------------------------------
    # 4) IDENTIFY BINS + UNKNOWN BIN (OUT-OF-WINDOW)
    # ---------------------------------------------------
    if True:
        _print("\n[STEP 4] Identifying censor bins and out-of-window bins")

        censoredtime = np.asarray(in_data_train["time"], dtype=float)
        time_bins_np = np.asarray(time_bins, dtype=float)

        temp_bins = np.array([0] + list(time_bins_np)[:-1], dtype=float)
        censoredbin = _map_indices(censoredtime, temp_bins).astype(int)
        binincrements = _map_indices(censoredtime + float(increment), temp_bins).astype(int)

        unknown_bins = np.minimum(binincrements + 1, T - 1).astype(int)
        unknown_bins = np.clip(unknown_bins, 0, T - 1)

        _print(f"[INFO] censor_bin stats: min={censoredbin.min()}, max={censoredbin.max()}")
        _print(f"[INFO] end_bin stats: min={binincrements.min()}, max={binincrements.max()}")
        _print(f"[INFO] unknown_bin stats: min={unknown_bins.min()}, max={unknown_bins.max()}")

        inspect_n = min(inspect_k, N)
        inspect_ids = rng.choice(N, size=inspect_n, replace=False) if inspect_n > 0 else np.array([], dtype=int)

        for i in inspect_ids:
            i = int(i)
            ub = int(unknown_bins[i])
            mean_pdf = pdf[i].mean(dim=0).detach().cpu().numpy()
            p_unknown = float(mean_pdf[ub])
            p_reveal = float(1.0 - p_unknown)

            _print(f"\n  [EXAMPLE i={i}]")
            _print(f"    probe/censor_time={censoredtime[i]:.4f}")
            _print(f"    censor_bin={int(censoredbin[i])}, end_bin={int(binincrements[i])}, unknown_bin={ub}")
            _print(f"    p_unknown={p_unknown:.4f}, p_reveal≈{p_reveal:.4f}")
            _print(f"    pdf mean over K: {mean_pdf}")

            a = int(censoredbin[i])
            if a > 0:
                mass_before = float(pdf[i, :, :a].mean(dim=0).sum().detach().cpu().item())
                _print(f"    [SANITY] mean mass in bins < censor_bin: {mass_before:.6e}")

    # ---------------------------------------------------
    # 4.5) OPTIONAL RESTRICTION MASK
    # ---------------------------------------------------
    if True:
        active_mask = np.ones(N, dtype=bool)
        if censored_indices is not None and len(censored_indices) > 0:
            active_mask[:] = False
            censored_indices = np.asarray(censored_indices, dtype=int)
            censored_indices = censored_indices[(censored_indices >= 0) & (censored_indices < N)]
            active_mask[censored_indices] = True
            _print(f"\n[INFO] Restricting candidates to censored_indices: {active_mask.sum()}/{N} active")

    # ---------------------------------------------------
    # 5) DOWNWEIGHT OUT-OF-WINDOW BIN
    # ---------------------------------------------------
    if True:

        _print("\n[STEP 5] Downweighting out-of-window bin (unknown bin)")

        _print(pdf[0, 0])
        _print(pdf[1, 0])
        _print(pdf[2, 0])
        _print(pdf[3, 0])

        max_time = float(time_bins_np[-1]) if float(time_bins_np[-1]) > 0 else 1.0
        ratio = np.clip(censoredtime / max_time, 0.0, 1.0)
        weights = ratio #np.clip(ratio ** float(outwin_alpha), float(outwin_min_weight), 1.0)   #one big change...

        _print(f"[INFO] out-window weights: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")

        #for i in inspect_ids:  #one big change...
        for i in range(N):
            i = int(i)
            ub = int(unknown_bins[i])

            # before = float(pdf[i, :, ub].mean().detach().cpu().item())
            pdf[i, :, ub] *= float(weights[i])
            pdf[i] /= torch.clamp(pdf[i].sum(dim=1, keepdim=True), min=1e-12)
            # after = float(pdf[i, :, ub].mean().detach().cpu().item())

            # _print(f"  i={i}: unknown-bin mean before={before:.4f}, after={after:.4f} (w={weights[i]:.3f}, ub={ub})")

        _print(pdf[0, 0])
        _print(pdf[1, 0])
        _print(pdf[2, 0])
        _print(pdf[3, 0])

        _print(f"[CHECK] After downweight: PDF row sums (mean over N,K): {pdf.sum(dim=2).mean().item():.6f}")

    # ---------------------------------------------------
    # 6) BALD SCORE
    # ---------------------------------------------------
    if True:
        _print("\n[STEP 6] Computing BALD scores")

        p_mean = pdf.mean(dim=1)  # [N,T]
        H_mean = -(p_mean * torch.log(p_mean + 1e-12)).sum(dim=1)
        H_each = -(pdf * torch.log(pdf + 1e-12)).sum(dim=2)
        bald = (H_mean - H_each.mean(dim=1)).detach().cpu().numpy()

        bald[~active_mask] = -np.inf
        finite_bald = bald[np.isfinite(bald)]
        _print(f"[INFO] BALD stats (finite only): min={finite_bald.min():.4e}, max={finite_bald.max():.4e}, mean={finite_bald.mean():.4e}, n={finite_bald.size}")

    # ---------------------------------------------------
    # 6.5) REVEAL-LIKELIHOOD SCORE (new term aligned with your oracle game)
    # ---------------------------------------------------
    if True:
        _print("\n[STEP 6.5] Computing reveal-likelihood proxy (1 - p_unknown)")

        pmean_np = p_mean.detach().cpu().numpy()  # [N,T]
        p_unknown = np.zeros(N, dtype=float)
        for i in range(N):
            ub = int(unknown_bins[i])
            p_unknown[i] = float(pmean_np[i, ub])
        p_reveal = 1.0 - p_unknown

        p_reveal[~active_mask] = 0.0
        _print(f"[INFO] p_unknown stats (active): min={p_unknown[active_mask].min():.4f}, max={p_unknown[active_mask].max():.4f}, mean={p_unknown[active_mask].mean():.4f}")
        _print(f"[INFO] p_reveal  stats (active): min={p_reveal[active_mask].min():.4f}, max={p_reveal[active_mask].max():.4f}, mean={p_reveal[active_mask].mean():.4f}")

    # ---------------------------------------------------
    # 7) DENSITY
    # ---------------------------------------------------
    if True:
        if density_weight > 0:
            
            _print("\n[STEP 7] Computing density scores (kNN inverse distance)")

            from sklearn.neighbors import NearestNeighbors
            X_np = np.asarray(X_pool)
            k = int(min(max(2, density_knn_k), len(X_np)))
            nbrs = NearestNeighbors(n_neighbors=k).fit(X_np)
            dists, _ = nbrs.kneighbors(X_np)

            avg_dist = dists[:, 1:].mean(axis=1) if dists.shape[1] > 1 else np.ones(len(X_np))
            density = 1.0 / (avg_dist + 1e-8)

            dmin, dmax = float(density.min()), float(density.max())
            density = np.ones_like(density) if (dmax - dmin < 1e-12) else (density - dmin) / (dmax - dmin)
            density[~active_mask] = 0.0

            _print(f"[INFO] Density stats (active): min={density[active_mask].min():.4f}, max={density[active_mask].max():.4f}, mean={density[active_mask].mean():.4f}")
        else:
            density = np.ones(N, dtype=float)
            density[~active_mask] = 0.0

    # ---------------------------------------------------
    # 8) COMBINE SCORES
    # ---------------------------------------------------
    if True:
        _print("\n[STEP 8] Combining BALD + density (+ reveal proxy)")

        # Normalize BALD to [0,1] over active
        active_bald = bald[active_mask]
        if np.isfinite(active_bald).any():
            bmin = float(np.min(active_bald[np.isfinite(active_bald)]))
            bmax = float(np.max(active_bald[np.isfinite(active_bald)]))
        else:
            bmin, bmax = 0.0, 1.0
        bald_norm = (bald - bmin) / (bmax - bmin + 1e-12)
        bald_norm[~active_mask] = 0.0

        # Normalize reveal to [0,1]
        pr = p_reveal.copy()
        pr_min, pr_max = float(pr[active_mask].min()), float(pr[active_mask].max())
        pr_norm = np.ones_like(pr) if (pr_max - pr_min < 1e-12) else (pr - pr_min) / (pr_max - pr_min)
        pr_norm[~active_mask] = 0.0

        combine_mode = "product"

        # Base MI+density
        if density_weight <= 0:
            base = bald_norm
        else:
            if combine_mode == "product":
                base = bald_norm * (density + 1e-6) #** float(density_weight)
            elif combine_mode == "sum":
                w = float(density_weight)
                base = (1.0 - w) * bald_norm + w * density
        
        score = base

        # Removed to see if it works
        # Inject reveal proxy (aligned with “oracle reveals within increment”)
        if reveal_weight and reveal_weight > 0:
            if combine_mode == "product":
                score = base * (pr_norm + 1e-6) ** float(reveal_beta)
            else:
                score = (1.0 - float(reveal_weight)) * base + float(reveal_weight) * pr_norm
        else:
            score = base

        score[~active_mask] = -np.inf
        finite_score = score[np.isfinite(score)]
        _print(f"[INFO] Combined score stats (finite only): min={finite_score.min():.4f}, max={finite_score.max():.4f}, mean={finite_score.mean():.4f}, n={finite_score.size}")

        # Print top candidates breakdown
        order = np.argsort(score)[::-1]
        top = order[:10]
        _print("\n[TOP-10] idx | score | bald_norm | density | p_reveal_norm | p_unknown")
        for idx in top:
            idx = int(idx)
            _print(f"  {idx:4d} | {score[idx]:.4f} | {bald_norm[idx]:.4f} | {density[idx]:.4f} | {pr_norm[idx]:.4f} | {p_unknown[idx]:.4f}")

    # ---------------------------------------------------
    # 9) SELECTION (top-k for debug)
    # ---------------------------------------------------
    if True:
        _print("\n[STEP 9] Selection")

        M = int(min(N, max(batch_size, preselect_multiplier * batch_size)))
        cand = order[:M]
        _print(f"[INFO] Candidate pool size M = {M}")
        _print(f"[INFO] Top-10 candidate scores: {score[cand[:10]]}")

        selected = cand[:batch_size].astype(int).tolist()
        _print(f"[RESULT] Selected indices: {selected}")

    # ---------------------------------------------------
    # 10) Optional budget
    # ---------------------------------------------------
    if True:
        if budget is not None and costlist is not None:
            _print("\n[STEP 10] Applying budget constraint")

            costs = costlist.values.flatten() if hasattr(costlist, "values") else np.asarray(costlist).flatten()
            costs = costs.astype(float)

            ratios = []
            for idx in cand:
                idx = int(idx)
                c = max(float(costs[idx]), 1e-12)
                ratios.append((idx, float(score[idx]) / c, float(score[idx]), float(costs[idx])))
            ratios.sort(key=lambda x: x[1], reverse=True)

            picked = []
            total = 0.0
            for idx, r, sc, c in ratios:
                if total + c <= float(budget):
                    picked.append(idx)
                    total += c
                if len(picked) >= batch_size:
                    break

            _print(f"[RESULT] Budget-picked indices: {picked}")
            _print(f"[INFO] Total cost used: {total:.4f} / budget {float(budget):.4f}")

            _print("\n==============================")
            _print(" END BatchBALD DEBUG RUN ")
            _print("==============================\n")
            return picked

    _print("\n==============================")
    _print(" END BatchBALD DEBUG RUN ")
    _print("==============================\n")

    return selected,None


# =============================================================================
#teleport

def batchbald_acquire_budget(model, X_pool, batch_size, time_bins, config, 
                             num_samples=60000, device='cpu', min_samples=400, 
                             in_data_train=None, increment=10000, costlist=None, 
                             budget=0, censored_indices=None, out_of_window_downweight=True, 
                             temperature=1.25, inwindow_gamma=0.3, use_binary_mi=None, 
                             use_diversity_filter=True, diversity_weight=0.15, density_weight=0.0):
    """
    Enhanced BatchBALD acquisition for survival analysis with censoring-aware weighting.
    """
    assert budget > 0, "Budget must be greater than 0"

    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(X_pool).to(device)
        _, _, ensemble_outputs = _make_prediction(model, x_test_tensor, time_bins, config)

    # Convert survival to PDF
    ensemble_outputs = ensemble_outputs.permute(1, 0, 2)
    modified_tensor = ensemble_outputs[:, :, 1:]
    zero_to_append = torch.zeros((ensemble_outputs.shape[0], ensemble_outputs.shape[1], 1), 
                                  dtype=torch.float32).to(device)
    modified_tensor = torch.cat((modified_tensor, zero_to_append), dim=2)
    probs_N_K_C = ensemble_outputs - modified_tensor

    time_bins_np = np.array(time_bins) if not isinstance(time_bins, np.ndarray) else time_bins
    temp_bins = np.array([0] + list(time_bins_np)[:-1])
    censoredtime = np.array(in_data_train['time'])
    
    censoredbin = _map_indices(censoredtime, temp_bins)
    binincrements = _map_indices(censoredtime + increment, temp_bins)
    
    start_bins = censoredbin
    end_bins = binincrements
    window_bins = np.maximum(end_bins - start_bins + 1, 1)
    avg_window_bins = float(np.mean(window_bins))

    logits_N_K_C = _fix_survival_timebins_increment(probs_N_K_C, time_bins, in_data_train, increment)
    N, K, C = logits_N_K_C.shape
    max_time = time_bins_np[-1]

    # Censoring-time-aware weighting
    if out_of_window_downweight:
        for i in range(N):
            unknown_bin_idx = int(min(binincrements[i] + 1, C - 1))
            censor_ratio = float(censoredtime[i]) / float(max_time)
            censor_weight = np.clip(np.sqrt(censor_ratio), 0.2, 1.0)
            logits_N_K_C[i, :, unknown_bin_idx] *= censor_weight
            row_sums = torch.clamp(logits_N_K_C[i, :, :].sum(dim=1, keepdim=True), min=1e-12)
            logits_N_K_C[i, :, :] /= row_sums

    # In-window emphasis
    if inwindow_gamma is not None and inwindow_gamma > 0:
        for i in range(N):
            s = max(0, min(int(start_bins[i]), C - 1))
            t = max(0, min(int(end_bins[i]), C - 1))
            if t < s: s, t = t, s
            if t == s: continue
            
            length = float(max(t - s, 1))
            w = torch.ones(C, dtype=logits_N_K_C.dtype, device=logits_N_K_C.device)
            for b in range(s, t + 1):
                rel_pos = float(b - s) / length
                w[b] = 1.0 + inwindow_gamma * (1.0 - rel_pos)
            
            logits_N_K_C[i, :, :] *= w.unsqueeze(0)
            logits_N_K_C[i, :, :] /= logits_N_K_C[i, :, :].sum(dim=1, keepdim=True)

    # Temperature smoothing
    if temperature is not None and temperature > 1.0:
        logits_N_K_C = torch.clamp(logits_N_K_C, min=1e-12)
        logits_N_K_C = logits_N_K_C ** (1.0 / temperature)
        logits_N_K_C /= logits_N_K_C.sum(dim=2, keepdim=True)

    # Auto binary MI
    if use_binary_mi is None:
        use_binary_mi = avg_window_bins <= 3
    
    if use_binary_mi:
        p_window_list = []
        for i in range(N):
            s = max(0, min(int(start_bins[i]), C - 1))
            t = max(0, min(int(end_bins[i]), C - 1))
            if t < s: s, t = t, s
            p_w = logits_N_K_C[i, :, s:(t + 1)].sum(dim=1) if t != s else logits_N_K_C[i, :, s]
            p_window_list.append(p_w)
        
        p_window = torch.stack(p_window_list, dim=0)
        p_unknown = 1.0 - p_window
        two_class = torch.stack([p_window, p_unknown], dim=2)
        two_class = torch.clamp(two_class, 1e-12, 1.0)
        two_class /= two_class.sum(dim=2, keepdim=True)
        bb_logits = torch.log(two_class)
    else:
        bb_logits = torch.log(torch.clamp(logits_N_K_C, min=1e-12))
    
    bb_logits[bb_logits == float('-inf')] = -10

    # === DENSITY SCORES ===
    density_scores = None
    if density_weight > 0:
        try:
            from sklearn.neighbors import NearestNeighbors
            # Compute density as inverse of average distance to k neighbors
            nbrs = NearestNeighbors(n_neighbors=min(20, len(X_pool)), algorithm='auto').fit(X_pool)
            distances, _ = nbrs.kneighbors(X_pool)
            
            # Exclude self (first column is 0 distance)
            if distances.shape[1] > 1:
                avg_dist = distances[:, 1:].mean(axis=1)
            else:
                avg_dist = np.ones(len(X_pool))
                
            density = 1.0 / (avg_dist + 1e-8)
            
            # Normalize to [0, 1]
            d_min, d_max = density.min(), density.max()
            if d_max - d_min > 1e-8:
                density_scores = (density - d_min) / (d_max - d_min)
            else:
                density_scores = np.ones_like(density)
                
        except Exception as e:
            print(f"Density computation failed: {e}")
            density_weight = 0.0

    # Handle censored-only selection
    if censored_indices is not None and len(censored_indices) > 0:
        censored_indices = np.array(censored_indices, dtype=int)
        logits_subset = bb_logits[torch.as_tensor(censored_indices, dtype=torch.long, device=device)]
        
        actual_num_samples = max(min_samples, num_samples)
        request_size = min(batch_size, len(censored_indices))
        
        candidate_batch = batchbald.get_batchbald_batch(
            logits_subset, request_size, actual_num_samples, dtype=torch.double, device=device
        )
        
        sub_indices = np.asarray(candidate_batch.indices, dtype=int)
        original_indices = censored_indices[sub_indices]
        scores = np.asarray(candidate_batch.scores, dtype=float)
        probs_for_div = logits_N_K_C[torch.as_tensor(original_indices, dtype=torch.long, device=device)].mean(dim=1)
    else:
        actual_num_samples = max(min_samples, num_samples)
        request_size = min(batch_size, N)
        
        candidate_batch = batchbald.get_batchbald_batch(
            bb_logits, request_size, actual_num_samples, dtype=torch.double, device=device
        )
        original_indices = np.asarray(candidate_batch.indices, dtype=int)
        scores = np.asarray(candidate_batch.scores, dtype=float)
        probs_for_div = logits_N_K_C[torch.as_tensor(original_indices, dtype=torch.long, device=device)].mean(dim=1)

    # === Select final batch by combining MI scores, density scores, and diversity ===
    if (use_diversity_filter or density_weight > 0) and len(original_indices) > batch_size:
        # Normalize MI scores to [0, 1]
        mi_min, mi_max = scores.min(), scores.max()
        if mi_max - mi_min > 1e-8:
            mi_normalized = (scores - mi_min) / (mi_max - mi_min)
        else:
            mi_normalized = np.ones_like(scores)
        
        # Greedy selection with combined scoring
        probs_np = probs_for_div.cpu().numpy()  # [M, C]
        selected = []
        selected_probs = []
        remaining = list(range(len(original_indices)))
        
        # Pre-fetch density scores for candidates
        cand_density = np.zeros(len(original_indices))
        if density_weight > 0 and density_scores is not None:
            cand_density = density_scores[original_indices]
        
        while len(selected) < batch_size and len(remaining) > 0:
            best_idx = None
            best_score = -np.inf
            
            for i in remaining:
                # MI component
                mi_score = mi_normalized[i]
                
                # Density component
                dens_score = cand_density[i]
                
                # Diversity component: average JS divergence to already selected
                if len(selected_probs) > 0 and use_diversity_filter:
                    p_i = probs_np[i] + 1e-12
                    p_i = p_i / p_i.sum()
                    
                    js_divs = []
                    for sp in selected_probs:
                        m = 0.5 * (p_i + sp)
                        kl1 = np.sum(p_i * np.log(p_i / (m + 1e-12) + 1e-12))
                        kl2 = np.sum(sp * np.log(sp / (m + 1e-12) + 1e-12))
                        js = 0.5 * (kl1 + kl2)
                        js_divs.append(js)
                    div_score = np.mean(js_divs)
                else:
                    div_score = 1.0  # Max diversity if no selected yet
                
                # Combined score
                w_mi = max(0.0, 1.0 - diversity_weight - density_weight)
                
                combined = w_mi * mi_score
                if use_diversity_filter:
                    combined += diversity_weight * div_score
                if density_weight > 0:
                    combined += density_weight * dens_score
                
                if combined > best_score:
                    best_score = combined
                    best_idx = i
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
                p_sel = probs_np[best_idx] + 1e-12
                p_sel = p_sel / p_sel.sum()
                selected_probs.append(p_sel)
        
        final_indices = [int(original_indices[i]) for i in selected]
    else:
        final_indices = list(original_indices[:batch_size])

    # Apply budget constraint
    if costlist is not None:
        costs = costlist.values.flatten() if hasattr(costlist, 'values') else np.asarray(costlist).flatten()
        # Re-rank by score/cost ratio
        final_scores = scores[[list(original_indices).index(i) for i in final_indices if i in list(original_indices)]]
        safe_costs = np.maximum(costs[final_indices], 1e-8)
        score_cost_ratio = final_scores / safe_costs
        order = np.argsort(score_cost_ratio)[::-1]

        selected_indices = []
        current_cost = 0.0
        for j in order:
            idx = int(final_indices[j])
            c = float(costs[idx])
            if current_cost + c <= budget:
                selected_indices.append(idx)
                current_cost += c
                if len(selected_indices) >= batch_size:
                    break
        return selected_indices
    
    return final_indices[:batch_size]


# =============================================================================
# BATCHBALD-C: BatchBALD with C-BALD Death Probability Weighting
# =============================================================================

def batchbald_c_acquire_budget(model, X_pool, batch_size, time_bins, config,
                                num_samples=60000, device='cpu', min_samples=400,
                                in_data_train=None, increment=10000, costlist=None,
                                budget=0, censored_indices=None, out_of_window_downweight=True,
                                temperature=1.25, inwindow_gamma=0.3, use_binary_mi=None,
                                use_diversity_filter=True, diversity_weight=0.15, density_weight=0.0,
                                death_prob_weight=0.7):
    """
    BatchBALD-C: Enhanced BatchBALD with C-BALD's death probability weighting.

    Combines:
    - BatchBALD's diversity mechanism (joint entropy)
    - C-BALD's information value weighting (death probability in reveal window)

    This should beat both:
    - C-BALD: by adding diversity (reducing redundancy)
    - BatchBALD: by adding information value (prioritizing high-value samples)

    Args:
        death_prob_weight: Weight for death probability component (0-1)
                          0 = pure BatchBALD, 1 = pure death probability weighting
                          Default 0.7 balances both strategies
    """
    assert budget > 0, "Budget must be greater than 0"

    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(X_pool).to(device)
        _, _, ensemble_outputs = _make_prediction(model, x_test_tensor, time_bins, config)

    # Convert survival to PDF
    ensemble_outputs = ensemble_outputs.permute(1, 0, 2)
    modified_tensor = ensemble_outputs[:, :, 1:]
    zero_to_append = torch.zeros((ensemble_outputs.shape[0], ensemble_outputs.shape[1], 1),
                                  dtype=torch.float32).to(device)
    modified_tensor = torch.cat((modified_tensor, zero_to_append), dim=2)
    probs_N_K_C = ensemble_outputs - modified_tensor

    time_bins_np = np.array(time_bins) if not isinstance(time_bins, np.ndarray) else time_bins
    temp_bins = np.array([0] + list(time_bins_np)[:-1])
    censoredtime = np.array(in_data_train['time'])

    censoredbin = _map_indices(censoredtime, temp_bins)
    binincrements = _map_indices(censoredtime + increment, temp_bins)

    start_bins = censoredbin
    end_bins = binincrements
    window_bins = np.maximum(end_bins - start_bins + 1, 1)
    avg_window_bins = float(np.mean(window_bins))

    logits_N_K_C = _fix_survival_timebins_increment(probs_N_K_C, time_bins, in_data_train, increment)
    N, K, C = logits_N_K_C.shape
    max_time = time_bins_np[-1]

    # Censoring-time-aware weighting
    if out_of_window_downweight:
        for i in range(N):
            unknown_bin_idx = int(min(binincrements[i] + 1, C - 1))
            censor_ratio = float(censoredtime[i]) / float(max_time)
            censor_weight = np.clip(np.sqrt(censor_ratio), 0.2, 1.0)
            logits_N_K_C[i, :, unknown_bin_idx] *= censor_weight
            row_sums = torch.clamp(logits_N_K_C[i, :, :].sum(dim=1, keepdim=True), min=1e-12)
            logits_N_K_C[i, :, :] /= row_sums

    # In-window emphasis
    if inwindow_gamma is not None and inwindow_gamma > 0:
        for i in range(N):
            s = max(0, min(int(start_bins[i]), C - 1))
            t = max(0, min(int(end_bins[i]), C - 1))
            if t < s: s, t = t, s
            if t == s: continue

            length = float(max(t - s, 1))
            w = torch.ones(C, dtype=logits_N_K_C.dtype, device=logits_N_K_C.device)
            for b in range(s, t + 1):
                rel_pos = float(b - s) / length
                w[b] = 1.0 + inwindow_gamma * (1.0 - rel_pos)

            logits_N_K_C[i, :, :] *= w.unsqueeze(0)
            logits_N_K_C[i, :, :] /= logits_N_K_C[i, :, :].sum(dim=1, keepdim=True)

    # Temperature smoothing
    if temperature is not None and temperature > 1.0:
        logits_N_K_C = torch.clamp(logits_N_K_C, min=1e-12)
        logits_N_K_C = logits_N_K_C ** (1.0 / temperature)
        logits_N_K_C /= logits_N_K_C.sum(dim=2, keepdim=True)

    # === C-BALD STYLE DEATH PROBABILITY WEIGHTING (NEW APPROACH!) ===
    # Weight probabilities by death probability BEFORE BatchBALD sees them
    # This makes BatchBALD naturally prioritize high-information-value samples

    # Compute death probability for each sample
    avg_probs = logits_N_K_C.mean(dim=1)  # [N, C]
    death_prob_samples = torch.zeros(N, device=device)

    for i in range(N):
        s = int(start_bins[i])
        e = min(int(end_bins[i]) + 1, C)
        death_prob_samples[i] = avg_probs[i, s:e].sum()

    # Boost probabilities in the reveal window by death probability
    # Higher death prob → more weight on that window → higher MI → more likely to be selected
    for i in range(N):
        s = int(start_bins[i])
        e = min(int(end_bins[i]) + 1, C)

        # C-BALD style weighting: (0.5 + 0.5 * death_prob)
        weight = (0.5 + 0.5 * death_prob_samples[i].item()) ** death_prob_weight

        # Apply weight to the reveal window
        logits_N_K_C[i, :, s:e] *= weight

        # Renormalize
        row_sums = logits_N_K_C[i, :, :].sum(dim=1, keepdim=True)
        logits_N_K_C[i, :, :] /= torch.clamp(row_sums, min=1e-12)

    # Auto binary MI
    if use_binary_mi is None:
        use_binary_mi = avg_window_bins <= 3

    if use_binary_mi:
        p_window_list = []
        for i in range(N):
            s = max(0, min(int(start_bins[i]), C - 1))
            t = max(0, min(int(end_bins[i]), C - 1))
            if t < s: s, t = t, s
            p_w = logits_N_K_C[i, :, s:(t + 1)].sum(dim=1) if t != s else logits_N_K_C[i, :, s]
            p_window_list.append(p_w)

        p_window = torch.stack(p_window_list, dim=0)
        p_unknown = 1.0 - p_window
        two_class = torch.stack([p_window, p_unknown], dim=2)
        two_class = torch.clamp(two_class, 1e-12, 1.0)
        two_class /= two_class.sum(dim=2, keepdim=True)
        bb_logits = torch.log(two_class)
    else:
        bb_logits = torch.log(torch.clamp(logits_N_K_C, min=1e-12))

    bb_logits[bb_logits == float('-inf')] = -10

    # === BATCHBALD SELECTION (probabilities already weighted by death probability!) ===
    # Since we weighted the logits above, BatchBALD will naturally select high-info-value samples

    # Handle censored-only selection
    if censored_indices is not None and len(censored_indices) > 0:
        censored_indices = np.array(censored_indices, dtype=int)
        logits_subset = bb_logits[torch.as_tensor(censored_indices, dtype=torch.long, device=device)]

        actual_num_samples = max(min_samples, num_samples)
        request_size = min(batch_size, len(censored_indices))

        candidate_batch = batchbald.get_batchbald_batch(
            logits_subset, request_size, actual_num_samples, dtype=torch.double, device=device
        )

        sub_indices = np.asarray(candidate_batch.indices, dtype=int)
        final_indices = censored_indices[sub_indices].tolist()
    else:
        actual_num_samples = max(min_samples, num_samples)
        request_size = min(batch_size, N)

        candidate_batch = batchbald.get_batchbald_batch(
            bb_logits, request_size, actual_num_samples, dtype=torch.double, device=device
        )
        final_indices = list(candidate_batch.indices)

    # Apply budget constraint
    if costlist is not None:
        costs = costlist.values.flatten() if hasattr(costlist, 'values') else np.asarray(costlist).flatten()
        batchbald_scores = np.asarray(candidate_batch.scores, dtype=float)
        safe_costs = np.maximum(costs[final_indices[:batch_size]], 1e-8)
        score_cost_ratio = batchbald_scores[:batch_size] / safe_costs
        order = np.argsort(score_cost_ratio)[::-1]

        selected_indices = []
        current_cost = 0.0
        for j in order:
            idx = int(final_indices[j])
            c = float(costs[idx])
            if current_cost + c <= budget:
                selected_indices.append(idx)
                current_cost += c
                if len(selected_indices) >= batch_size:
                    break
        return selected_indices

    return final_indices[:batch_size]


# =============================================================================
# RANDOM ACQUISITION
# =============================================================================

def random_knapsack(costlist, budget, type=0):
    """Random selection within budget."""
    if costlist is None:
        raise ValueError("random_knapsack requires costlist")
    
    costs = np.array(costlist).reshape(-1)
    indices = list(range(len(costs)))
    chosen_indices = []
    current_budget = 0

    if type == 0:
        random.shuffle(indices)
        for index in indices:
            if current_budget + costs[index] <= budget:
                chosen_indices.append(index)
                current_budget += costs[index]
    elif type == 1:
        inverse_costs = [1 / cost for cost in costs]
        total_inverse_cost = sum(inverse_costs)
        probabilities = [inv_cost / total_inverse_cost for inv_cost in inverse_costs]

        while indices and current_budget < budget:
            index = random.choices(indices, weights=probabilities, k=1)[0]
            if current_budget + costs[index] <= budget:
                chosen_indices.append(index)
                current_budget += costs[index]
            idx_pos = indices.index(index)
            indices.pop(idx_pos)
            probabilities.pop(idx_pos)
            if indices:
                total_inverse_cost = sum(probabilities)
                probabilities = [prob / total_inverse_cost for prob in probabilities]
    
    return chosen_indices


# =============================================================================
# CBALD-DIVERSE: C-BALD with Diversity Filtering (Hybrid Approach)
# =============================================================================

def cbald_diverse_acquire(model, X_pool, batch_size, time_bins, config,
                          device='cpu', in_data_train=None, increment=10000,
                          costlist=None, budget=0, diversity_ratio=0.3, **kwargs):
    """
    Hybrid: C-BALD scoring with diversity filtering.

    Strategy:
    1. Compute C-BALD scores (information value - proven winner)
    2. Select top candidates by C-BALD score
    3. Greedily pick diverse subset using JS divergence

    This should beat C-BALD by adding diversity while keeping information value.

    Args:
        diversity_ratio: Emphasis on diversity vs C-BALD score (0-1)
                        0 = pure C-BALD, 1 = pure diversity
                        Default 0.3 balances both
    """

    # Step 1: Compute C-BALD scores
    cbald_scores = cbald_score(model, X_pool, batch_size, time_bins, config,
                               device, in_data_train, increment, budget, costlist, **kwargs)

    # Step 2: Get top candidates (3x budget for diversity selection)
    top_k = min(batch_size * 3, len(X_pool))
    top_indices = np.argsort(cbald_scores)[-top_k:][::-1]

    # If no diversity needed or pool too small, just return top samples
    if diversity_ratio == 0 or len(top_indices) <= batch_size:
        final_indices = top_indices[:batch_size].tolist()
        return final_indices, cbald_scores

    # Step 3: Get probability distributions for diversity computation
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(X_pool[top_indices]).to(device)
        survival_outputs, _, ensemble_outputs = _make_prediction(model, x_test_tensor, time_bins, config)

    K, N, C = ensemble_outputs.shape
    probs = ensemble_to_pdf(ensemble_outputs, device)
    avg_probs = probs.mean(dim=1).cpu().numpy()  # [N, C]

    # Step 4: Greedy diversity selection from top C-BALD candidates
    selected_local = []  # Indices in top_indices array
    selected_probs = []
    remaining = list(range(len(top_indices)))

    # Select first sample (highest C-BALD score)
    selected_local.append(0)
    remaining.remove(0)
    p_sel = avg_probs[0]
    p_sel = p_sel / (p_sel.sum() + 1e-12)
    selected_probs.append(p_sel)

    # Greedily select remaining samples balancing C-BALD score + diversity
    while len(selected_local) < batch_size and len(remaining) > 0:
        best_idx = None
        best_combined_score = -np.inf

        for i in remaining:
            # C-BALD score component
            local_cbald_score = cbald_scores[top_indices[i]]

            # Diversity component: average JS divergence to already selected
            p_i = avg_probs[i]
            p_i = p_i / (p_i.sum() + 1e-12)

            js_divs = []
            for sp in selected_probs:
                m = 0.5 * (p_i + sp)
                kl1 = np.sum(p_i * np.log((p_i + 1e-12) / (m + 1e-12)))
                kl2 = np.sum(sp * np.log((sp + 1e-12) / (m + 1e-12)))
                js = 0.5 * (kl1 + kl2)
                js_divs.append(js)
            diversity_score = np.mean(js_divs)

            # Normalize and combine
            max_cbald = cbald_scores[top_indices].max()
            normalized_cbald = local_cbald_score / (max_cbald + 1e-12)
            normalized_diversity = diversity_score  # Already 0-1 range

            combined = (1 - diversity_ratio) * normalized_cbald + diversity_ratio * normalized_diversity

            if combined > best_combined_score:
                best_combined_score = combined
                best_idx = i

        if best_idx is not None:
            selected_local.append(best_idx)
            remaining.remove(best_idx)
            p_sel = avg_probs[best_idx]
            p_sel = p_sel / (p_sel.sum() + 1e-12)
            selected_probs.append(p_sel)

    # Map back to original indices
    final_indices = [top_indices[i] for i in selected_local]

    return final_indices, cbald_scores


def cbald_diverse_adaptive_acquire(model, X_pool, batch_size, time_bins, config,
                                    device='cpu', in_data_train=None, increment=10000,
                                    costlist=None, budget=0, start_ratio=0.1, end_ratio=0.5, **kwargs):
    """
    CBALD-Diverse with Adaptive Ratio: Exploit early, explore later.

    Strategy:
    - Early selections: High C-BALD emphasis (90% C-BALD + 10% diversity)
    - Later selections: Balanced (50% C-BALD + 50% diversity)

    This should improve on CBALD-Diverse by adapting strategy as batch fills.

    Args:
        start_ratio: Initial diversity ratio (default 0.1 = 10% diversity)
        end_ratio: Final diversity ratio (default 0.5 = 50% diversity)
    """

    # Step 1: Compute C-BALD scores
    cbald_scores = cbald_score(model, X_pool, batch_size, time_bins, config,
                               device, in_data_train, increment, budget, costlist, **kwargs)

    # Step 2: Get top candidates (3x budget for diversity selection)
    top_k = min(batch_size * 3, len(X_pool))
    top_indices = np.argsort(cbald_scores)[-top_k:][::-1]

    # If pool too small, just return top samples
    if len(top_indices) <= batch_size:
        final_indices = top_indices[:batch_size].tolist()
        return final_indices, cbald_scores

    # Step 3: Get probability distributions for diversity computation
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(X_pool[top_indices]).to(device)
        survival_outputs, _, ensemble_outputs = _make_prediction(model, x_test_tensor, time_bins, config)

    K, N, C = ensemble_outputs.shape
    probs = ensemble_to_pdf(ensemble_outputs, device)
    avg_probs = probs.mean(dim=1).cpu().numpy()  # [N, C]

    # Step 4: Adaptive greedy diversity selection
    selected_local = []
    selected_probs = []
    remaining = list(range(len(top_indices)))

    # Select first sample (highest C-BALD score)
    selected_local.append(0)
    remaining.remove(0)
    p_sel = avg_probs[0]
    p_sel = p_sel / (p_sel.sum() + 1e-12)
    selected_probs.append(p_sel)

    # Greedily select remaining samples with ADAPTIVE ratio
    while len(selected_local) < batch_size and len(remaining) > 0:
        # Compute current diversity ratio based on progress
        progress = len(selected_local) / batch_size  # 0 to 1
        current_ratio = start_ratio + (end_ratio - start_ratio) * progress

        best_idx = None
        best_combined_score = -np.inf

        for i in remaining:
            # C-BALD score component
            local_cbald_score = cbald_scores[top_indices[i]]

            # Diversity component
            p_i = avg_probs[i]
            p_i = p_i / (p_i.sum() + 1e-12)

            js_divs = []
            for sp in selected_probs:
                m = 0.5 * (p_i + sp)
                kl1 = np.sum(p_i * np.log((p_i + 1e-12) / (m + 1e-12)))
                kl2 = np.sum(sp * np.log((sp + 1e-12) / (m + 1e-12)))
                js = 0.5 * (kl1 + kl2)
                js_divs.append(js)
            diversity_score = np.mean(js_divs)

            # Normalize and combine with ADAPTIVE ratio
            max_cbald = cbald_scores[top_indices].max()
            normalized_cbald = local_cbald_score / (max_cbald + 1e-12)
            normalized_diversity = diversity_score

            combined = (1 - current_ratio) * normalized_cbald + current_ratio * normalized_diversity

            if combined > best_combined_score:
                best_combined_score = combined
                best_idx = i

        if best_idx is not None:
            selected_local.append(best_idx)
            remaining.remove(best_idx)
            p_sel = avg_probs[best_idx]
            p_sel = p_sel / (p_sel.sum() + 1e-12)
            selected_probs.append(p_sel)

    # Map back to original indices
    final_indices = [top_indices[i] for i in selected_local]

    return final_indices, cbald_scores


def cbald_diverse_nofilter_acquire(model, X_pool, batch_size, time_bins, config,
                                    device='cpu', in_data_train=None, increment=10000,
                                    costlist=None, budget=0, diversity_ratio=0.3, **kwargs):
    """
    CBALD-Diverse without pre-filtering: Score ALL samples.

    Strategy:
    - No variance-based pre-filtering (removes bias)
    - Score all ~1961 samples with C-BALD (slower but more thorough)
    - Select diverse subset from top candidates

    This should find high-value samples missed by variance pre-filtering.
    """

    # Step 1: Compute C-BALD scores for ALL samples (no pre-filtering)
    cbald_scores = cbald_score(model, X_pool, batch_size, time_bins, config,
                               device, in_data_train, increment, budget, costlist, **kwargs)

    # Step 2: Get top candidates (3x budget for diversity selection)
    top_k = min(batch_size * 3, len(X_pool))
    top_indices = np.argsort(cbald_scores)[-top_k:][::-1]

    # If no diversity needed or pool too small, just return top samples
    if diversity_ratio == 0 or len(top_indices) <= batch_size:
        final_indices = top_indices[:batch_size].tolist()
        return final_indices, cbald_scores

    # Step 3: Get probability distributions for diversity computation
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(X_pool[top_indices]).to(device)
        survival_outputs, _, ensemble_outputs = _make_prediction(model, x_test_tensor, time_bins, config)

    K, N, C = ensemble_outputs.shape
    probs = ensemble_to_pdf(ensemble_outputs, device)
    avg_probs = probs.mean(dim=1).cpu().numpy()  # [N, C]

    # Step 4: Greedy diversity selection (same as original CBALD-Diverse)
    selected_local = []
    selected_probs = []
    remaining = list(range(len(top_indices)))

    # Select first sample (highest C-BALD score)
    selected_local.append(0)
    remaining.remove(0)
    p_sel = avg_probs[0]
    p_sel = p_sel / (p_sel.sum() + 1e-12)
    selected_probs.append(p_sel)

    # Greedily select remaining samples
    while len(selected_local) < batch_size and len(remaining) > 0:
        best_idx = None
        best_combined_score = -np.inf

        for i in remaining:
            # C-BALD score component
            local_cbald_score = cbald_scores[top_indices[i]]

            # Diversity component
            p_i = avg_probs[i]
            p_i = p_i / (p_i.sum() + 1e-12)

            js_divs = []
            for sp in selected_probs:
                m = 0.5 * (p_i + sp)
                kl1 = np.sum(p_i * np.log((p_i + 1e-12) / (m + 1e-12)))
                kl2 = np.sum(sp * np.log((sp + 1e-12) / (m + 1e-12)))
                js = 0.5 * (kl1 + kl2)
                js_divs.append(js)
            diversity_score = np.mean(js_divs)

            # Normalize and combine
            max_cbald = cbald_scores[top_indices].max()
            normalized_cbald = local_cbald_score / (max_cbald + 1e-12)
            normalized_diversity = diversity_score

            combined = (1 - diversity_ratio) * normalized_cbald + diversity_ratio * normalized_diversity

            if combined > best_combined_score:
                best_combined_score = combined
                best_idx = i

        if best_idx is not None:
            selected_local.append(best_idx)
            remaining.remove(best_idx)
            p_sel = avg_probs[best_idx]
            p_sel = p_sel / (p_sel.sum() + 1e-12)
            selected_probs.append(p_sel)

    # Map back to original indices
    final_indices = [top_indices[i] for i in selected_local]

    return final_indices, cbald_scores


def cbald_twostage_acquire(model, X_pool, batch_size, time_bins, config,
                            device='cpu', in_data_train=None, increment=10000,
                            costlist=None, budget=0, exploit_ratio=0.5, diversity_ratio=0.3, **kwargs):
    """
    CBALD Two-Stage: Pure exploitation then diverse exploration.

    Strategy:
    - Stage 1: Select top 50% samples by pure C-BALD score (best value)
    - Stage 2: Select remaining 50% with diversity filtering

    This guarantees we get the absolute highest-value samples first.

    Args:
        exploit_ratio: Fraction of budget for pure C-BALD (default 0.5 = 50%)
        diversity_ratio: Diversity emphasis in stage 2 (default 0.3)
    """

    # Step 1: Compute C-BALD scores
    cbald_scores = cbald_score(model, X_pool, batch_size, time_bins, config,
                               device, in_data_train, increment, budget, costlist, **kwargs)

    # Step 2: Determine stage sizes
    stage1_size = int(batch_size * exploit_ratio)
    stage2_size = batch_size - stage1_size

    # Get top candidates for stage 2 selection (3x stage2 budget)
    top_k = min(batch_size + stage2_size * 3, len(X_pool))
    top_indices = np.argsort(cbald_scores)[-top_k:][::-1]

    # STAGE 1: Pure C-BALD exploitation
    stage1_indices = top_indices[:stage1_size].tolist()

    if stage2_size == 0:
        return stage1_indices, cbald_scores

    # STAGE 2: Diverse exploration from remaining top candidates
    stage2_candidates = top_indices[stage1_size:]

    if len(stage2_candidates) <= stage2_size:
        # Not enough candidates, just take all
        final_indices = stage1_indices + stage2_candidates.tolist()
        return final_indices, cbald_scores

    # Get probability distributions for diversity computation
    model.eval()
    with torch.no_grad():
        # Include stage1 samples for diversity computation
        x_test_tensor = torch.FloatTensor(X_pool[top_indices]).to(device)
        survival_outputs, _, ensemble_outputs = _make_prediction(model, x_test_tensor, time_bins, config)

    K, N, C = ensemble_outputs.shape
    probs = ensemble_to_pdf(ensemble_outputs, device)
    avg_probs = probs.mean(dim=1).cpu().numpy()  # [N, C]

    # Initialize with stage 1 selections
    selected_local = list(range(stage1_size))
    selected_probs = []
    for i in range(stage1_size):
        p_sel = avg_probs[i]
        p_sel = p_sel / (p_sel.sum() + 1e-12)
        selected_probs.append(p_sel)

    # Stage 2 candidates (offset by stage1_size)
    remaining = list(range(stage1_size, len(top_indices)))

    # Greedy diverse selection for stage 2
    while len(selected_local) < batch_size and len(remaining) > 0:
        best_idx = None
        best_combined_score = -np.inf

        for i in remaining:
            # C-BALD score component
            local_cbald_score = cbald_scores[top_indices[i]]

            # Diversity component
            p_i = avg_probs[i]
            p_i = p_i / (p_i.sum() + 1e-12)

            js_divs = []
            for sp in selected_probs:
                m = 0.5 * (p_i + sp)
                kl1 = np.sum(p_i * np.log((p_i + 1e-12) / (m + 1e-12)))
                kl2 = np.sum(sp * np.log((sp + 1e-12) / (m + 1e-12)))
                js = 0.5 * (kl1 + kl2)
                js_divs.append(js)
            diversity_score = np.mean(js_divs)

            # Normalize and combine
            max_cbald = cbald_scores[top_indices].max()
            normalized_cbald = local_cbald_score / (max_cbald + 1e-12)
            normalized_diversity = diversity_score

            combined = (1 - diversity_ratio) * normalized_cbald + diversity_ratio * normalized_diversity

            if combined > best_combined_score:
                best_combined_score = combined
                best_idx = i

        if best_idx is not None:
            selected_local.append(best_idx)
            remaining.remove(best_idx)
            p_sel = avg_probs[best_idx]
            p_sel = p_sel / (p_sel.sum() + 1e-12)
            selected_probs.append(p_sel)

    # Map back to original indices
    final_indices = [top_indices[i] for i in selected_local]

    return final_indices, cbald_scores


# =============================================================================
# ENTROPY AND VARIANCE ACQUISITION
# =============================================================================


#==
#entropy_of_probs_score
#==

def entropy_of_probs_score(model, X_pool, batch_size, time_bins, config, device='cuda', 
                     in_data_train=None, increment=10, budget=5, costlist=None, return_scores=False):
    """Entropy-based acquisition."""
    assert budget > 0, "Budget must be greater than 0"
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(X_pool).to(device)
        _, _, ensemble_outputs = _make_prediction(model, x_test_tensor, time_bins, config)

    logits_N_K_C = ensemble_to_pdf(ensemble_outputs, device)
    logits_N_K_C = _fix_survival_timebins_increment(logits_N_K_C, time_bins, in_data_train, increment)
    
    # Compute entropy per sample
    avg_probs = logits_N_K_C.mean(dim=1)  # [N, C]
    avg_probs = torch.clamp(avg_probs, min=1e-10)
    entropy = -torch.sum(avg_probs * torch.log2(avg_probs), dim=1)  # [N]
    
    if return_scores:
        return entropy.cpu().numpy()
    
    entropy_score = entropy.cpu().numpy()
    return entropy_score

def entropy_of_probs(model, X_pool, batch_size, time_bins, config, device='cuda', 
                     in_data_train=None, increment=10, budget=5, costlist=None, return_scores=False):
    """Entropy-based acquisition."""
    assert budget > 0, "Budget must be greater than 0"

    entropy_score = entropy_of_probs_score(model, X_pool, batch_size, time_bins, config, device, in_data_train, increment, budget, costlist, return_scores)
    print("entropy_of_probs score: ", entropy_score)
    return select_indices_from_scores(entropy_score, budget, costlist, batch_size), entropy_score


#==
#variance_of_probs_score
#==

def variance_of_probs_score(model, X_pool, batch_size, time_bins, config, num_samples=10000, 
                      device='cpu', min_samples=10, in_data_train=None, increment=10000, 
                      costlist=None, budget=0):
    """Variance-based acquisition."""
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(X_pool).to(device)
        _, _, ensemble_outputs = _make_prediction(model, x_test_tensor, time_bins, config)

    logits_N_K_C = ensemble_to_pdf(ensemble_outputs, device)

    logits_N_K_C = _fix_survival_timebins_increment(logits_N_K_C, time_bins, in_data_train, increment)
    
    variance = logits_N_K_C.var(dim=1).sum(dim=1)  # [N]
    variance_score = variance.cpu().numpy()
    return variance_score

def variance_of_probs(model, X_pool, batch_size, time_bins, config, num_samples=10000, 
                      device='cpu', min_samples=10, in_data_train=None, increment=10000, 
                      costlist=None, budget=0):
    """Variance-based acquisition."""
    assert budget > 0, "Budget must be greater than 0"   
    variance_score = variance_of_probs_score(model, X_pool, batch_size, time_bins, config, num_samples, device, min_samples, in_data_train, increment, costlist, budget)
    print("variance_of_probs score: ", variance_score)
    return select_indices_from_scores(variance_score, budget, costlist, batch_size), variance_score


# =============================================================================
# OTHER ACQUISITION FUNCTIONS
# =============================================================================

#==
#clostest_to_half_in_increment_score
#==
def clostest_to_half_in_increment_score(model, X_pool, batch_size, time_bins, config, 
                                  num_samples=10000, device='cpu', min_samples=10, 
                                  in_data_train=None, increment=10, costlist=None, budget=0):
    """Select samples with predicted probability closest to 0.5 in window."""
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(X_pool).to(device)
        _, _, ensemble_outputs = _make_prediction(model, x_test_tensor, time_bins, config)

    logits_N_K_C = ensemble_to_pdf(ensemble_outputs, device)
    logits_N_K_C = _fix_survival_timebins_increment(logits_N_K_C, time_bins, in_data_train, increment)
    
    avg_probs = logits_N_K_C.mean(dim=1)  # [N, C]
    max_probs = avg_probs.max(dim=1)[0]  # [N]
    distance_to_half_score = torch.abs(max_probs - 0.5)
    return distance_to_half_score.cpu().numpy()

def clostest_to_half_in_increment(model, X_pool, batch_size, time_bins, config, 
                                  num_samples=10000, device='cpu', min_samples=10, 
                                  in_data_train=None, increment=10, costlist=None, budget=0):
    """Select samples with predicted probability closest to 0.5 in window."""
    assert budget > 0, "Budget must be greater than 0"
    distance_to_half_score = clostest_to_half_in_increment_score(model, X_pool, batch_size, time_bins, config, num_samples, device, min_samples, in_data_train, increment, costlist, budget)
    print("clostest_to_half_in_increment score: ", distance_to_half_score)
    return select_indices_from_scores(distance_to_half_score, budget, costlist, batch_size), distance_to_half_score

#==
#mean_closest_to_middle
#==

def mean_closest_to_middle_score(model, X_pool, batch_size, time_bins, config, 
                           num_samples=10000, device='cpu', min_samples=10, 
                           in_data_train=None, increment=10000, costlist=None, budget=0):
    """Select samples with mean prediction closest to middle time bin."""
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(X_pool).to(device)
        _, _, ensemble_outputs = _make_prediction(model, x_test_tensor, time_bins, config)

    survival_diff = ensemble_to_pdf(ensemble_outputs, device)

    print("middle of mean_closest_to_middle")

    
    time_bins_np = np.array(time_bins)

    print("survival_diff of mean_closest_to_middle, time_bins_np", survival_diff.shape, time_bins_np.shape)

    bin_centers = torch.tensor((time_bins_np[:-1] + time_bins_np[1:]) / 2, 
                                dtype=torch.float32, device=device)

    print("middle of mean_closest_to_middle1")

    
    expected_time = (survival_diff * bin_centers[:len(survival_diff[0])]).sum(dim=1)
    
    print("middle of mean_closest_to_middle2")


    middle_time = (time_bins_np.max() + time_bins_np.min()) / 2

    print("middle of mean_closest_to_middle3")

    distance_to_middle_score = torch.abs(expected_time - middle_time)

    print("middle of mean_closest_to_middle4")

    return distance_to_middle_score.cpu().numpy()

def mean_closest_to_middle(model, X_pool, batch_size, time_bins, config, 
                           num_samples=10000, device='cpu', min_samples=10, 
                           in_data_train=None, increment=10000, costlist=None, budget=0):
    """Select samples with mean prediction closest to middle time bin."""
    assert budget > 0, "Budget must be greater than 0"

    print("starting mean_closest_to_middle")
    distance_to_middle_score = mean_closest_to_middle_score(model, X_pool, batch_size, time_bins, config, 
                           num_samples=num_samples, device=device, min_samples=min_samples, 
                           in_data_train=in_data_train, increment=increment, costlist=costlist, budget=budget)
    print("mean_closest_to_middle score: ", distance_to_middle_score)
    return select_indices_from_scores(distance_to_middle_score, budget, costlist, batch_size), distance_to_middle_score


#==
#Using_Clusters_for_batch
#==


def Using_Clusters_for_batch(model, X_pool, batch_size, time_bins, config, 
                             num_samples=10000, device='cpu', min_samples=10, 
                             in_data_train=None, increment=10000, costlist=None, budget=0):
    """Cluster-based diverse sampling."""
    assert budget > 0, "Budget must be greater than 0"
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pool)
    
    n_clusters = min(batch_size, len(X_pool))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    selected = []
    for center in kmeans.cluster_centers_:
        distances = np.sqrt(np.sum((X_scaled - center) ** 2, axis=1))
        closest = np.argmin(distances)
        if closest not in selected:
            selected.append(closest)
    
    if costlist is not None:
        costs = costlist.values.flatten() if hasattr(costlist, 'values') else np.asarray(costlist).flatten()
        final_selected = []
        current_cost = 0
        for idx in selected:
            if current_cost + costs[idx] <= budget:
                final_selected.append(int(idx))
                current_cost += costs[idx]
                if len(final_selected) >= batch_size:
                    break
        return final_selected
    
    return selected[:batch_size],None

#==
#highest_death_probability_in_window_score
#==

def highest_death_probability_in_window_score(model, X_pool, batch_size, time_bins, config, 
                                        num_samples=10000, device='cpu', min_samples=10, 
                                        in_data_train=None, increment=10000, costlist=None, budget=0):
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(X_pool).to(device)
        _, _, ensemble_outputs = _make_prediction(model, x_test_tensor, time_bins, config)


    probs_N_K_C = ensemble_to_pdf(ensemble_outputs, device)

    temp_bins = np.array([0] + list(time_bins)[:-1])
    censoredtime = np.array(in_data_train['time'])
    
    start_bins = _map_indices(censoredtime, temp_bins).astype(int)
    end_bins = _map_indices(censoredtime + increment, temp_bins).astype(int)
    
    avg_probs = probs_N_K_C.mean(dim=1)  # [N, C]
    death_prob = torch.zeros(len(X_pool), device=device)
    
    for i in range(len(X_pool)):
        s = int(start_bins[i])
        e = min(int(end_bins[i]) + 1, avg_probs.shape[1])
        death_prob[i] = avg_probs[i, s:e].sum()
    death_prob_score = death_prob.cpu().numpy()
    return death_prob_score

def highest_death_probability_in_window(model, X_pool, batch_size, time_bins, config, 
                                        num_samples=10000, device='cpu', min_samples=10, 
                                        in_data_train=None, increment=10000, costlist=None, budget=0):
    """Select samples with highest death probability in window."""
    assert budget > 0, "Budget must be greater than 0"

    score = highest_death_probability_in_window_score(model=model, X_pool=X_pool, batch_size=batch_size, time_bins=time_bins, config=config, 
                                        num_samples=num_samples, device=device, min_samples=min_samples, 
                                        in_data_train=in_data_train, increment=increment, costlist=costlist, budget=budget)
    print("highest_death_probability_in_window score: ", score)
    return select_indices_from_scores(score, budget, costlist, batch_size), score



#==
#CBALD
#==

def cbald_score(model, X_pool, batch_size, time_bins, config, 
                              device='cpu', in_data_train=None, increment=10000, 
                              budget=0, costlist=None, **kwargs):
    """C-BALD for censored regression."""

    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(X_pool).to(device)
        survival_outputs, _, ensemble_outputs = _make_prediction(model, x_test_tensor, time_bins, config)

    K, N, C = ensemble_outputs.shape
    probs = ensemble_to_pdf(ensemble_outputs, device)
    
    bin_mids = (np.concatenate([[0], time_bins[:-1]]) + time_bins) / 2

    bin_mids_tensor = torch.tensor(bin_mids, dtype=torch.float32, device=device)
    expected_times = (probs[:, :, :len(bin_mids)] * bin_mids_tensor).sum(dim=2)  # [N, K]
    
    # BALD: variance of expected time (epistemic uncertainty)
    time_variance = expected_times.var(dim=1)  # [N]
    
    # Weight by death probability in window
    temp_bins = np.array([0] + list(time_bins)[:-1])
    censoredtime = np.array(in_data_train['time'])
    start_bins = _map_indices(censoredtime, temp_bins).astype(int)
    end_bins = _map_indices(censoredtime + increment, temp_bins).astype(int)
    
    avg_probs = probs.mean(dim=1)
    death_prob = torch.zeros(N, device=device)
    for i in range(N):
        s = int(start_bins[i])
        e = min(int(end_bins[i]) + 1, avg_probs.shape[1])
        death_prob[i] = avg_probs[i, s:e].sum()
    
    # Combined score
    cbald_score = time_variance * (0.5 + 0.5 * death_prob)
    return cbald_score.cpu().numpy()

def cbald_censored_regression(model, X_pool, batch_size, time_bins, config, 
                              device='cpu', in_data_train=None, increment=10000, 
                              costlist=None, budget=0, **kwargs):
    # Combined score
    score = cbald_score(model, X_pool, batch_size, time_bins, config, device, in_data_train, increment, budget,costlist,**kwargs)
    print("cbald_censored_regression score: ", score)
    return select_indices_from_scores(score, budget, costlist, batch_size), score


#===============================================
# Acquisition Function Utilities
#===============================================

def select_indices_from_scores(scores, budget, costlist, batch_size):
    scores_t = torch.as_tensor(scores)
    sorted_indices = torch.argsort(scores_t, descending=True).cpu().numpy()
    print("sorted_indices: ", sorted_indices)

    if costlist is not None:
        costs = costlist.values.flatten() if hasattr(costlist, 'values') else np.asarray(costlist).flatten()
        selected = []
        current_cost = 0
        for idx in sorted_indices:
            if current_cost + costs[idx] <= budget:
                selected.append(int(idx))
                current_cost += costs[idx]
                if len(selected) >= batch_size:
                    break
        return selected

    return list(sorted_indices[:batch_size])

# # Aliases
# Batchbald_new1 = batchbald_acquire_budget
# DecensorBALD = batchbald_acquire_budget
# MAEPO_BatchBALD = batchbald_acquire_budget
# BatchBALD_Density = batchbald_density




##batchbald attempt implementation

def test_batchbald_score(
    model,
    X_pool,
    batch_size,
    time_bins,
    config,
    device="cpu",
    in_data_train=None,
    increment=0,
    censored_indices=None,
    preselect_multiplier=10,
    costlist=None,
    budget=None,
    seed=0,
    verbose=True,
):
    """
    BatchBALD acquisition function for survival analysis with probe depth constraints.
    
    Args:
        model: Bayesian survival model
        X_pool: Pool of candidate points (features)
        batch_size: Number of points to acquire
        time_bins: Array of time bin boundaries
        config: Configuration object with model parameters
        device: torch device
        in_data_train: Training data containing censoring information
            Expected to have 'artificial_time' and 'artificial_event' fields
            or 'time' and 'event' fields for censoring status
        increment: Probe depth k - how many bins forward the oracle can reveal
        censored_indices: Indices of censored points in the pool (if None, derived from in_data_train)
        preselect_multiplier: For efficiency, preselect top candidates before BatchBALD
        costlist: Optional costs per point
        budget: Optional budget constraint
        seed: Random seed
        verbose: Print progress
        
    Returns:
        List of indices of selected points
    """
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(X_pool, dtype=torch.float32, device=device)
        _, _, ensemble_outputs = make_prediction(model, x, time_bins, config)
        pdf = ensemble_to_pdf(ensemble_outputs, device)  # [N, K, T]
    
    N, K, T = pdf.shape
    
    # Extract censoring information from training data
    # artificial_time is the bin at which the point is censored
    # artificial_event: 0 = censored, 1 = uncensored (event observed)
    if in_data_train is not None:
        if hasattr(in_data_train, 'artificial_time'):
            censor_times = in_data_train.artificial_time  # bin index where censored
            censor_events = in_data_train.artificial_event  # 0 = censored
        else:
            censor_times = in_data_train.time
            censor_events = in_data_train.event
    else:
        # If no data provided, assume all are censored at bin 0
        censor_times = np.zeros(N, dtype=int)
        censor_events = np.zeros(N, dtype=int)
    
    # Convert to numpy if needed
    if torch.is_tensor(censor_times):
        censor_times = censor_times.cpu().numpy()
    if torch.is_tensor(censor_events):
        censor_events = censor_events.cpu().numpy()
    
    # Identify censored points (only these can be queried)
    if censored_indices is None:
        censored_indices = np.where(censor_events == 0)[0]
    
    if len(censored_indices) == 0:
        if verbose:
            print("No censored points to query!")
        return []
    
    # Compute oracle entropy for each censored point
    # H(Y_oracle) = -sum_{t=c}^{c+k} p_t log p_t - P(T > c+k) log P(T > c+k)
    oracle_entropies = compute_oracle_entropies(
        pdf, censor_times, increment, censored_indices, device
    )
    
    # Preselect top candidates for efficiency
    n_preselect = min(len(censored_indices), batch_size * preselect_multiplier)
    top_indices_in_censored = torch.argsort(oracle_entropies, descending=True)[:n_preselect]
    candidate_indices = censored_indices[top_indices_in_censored.cpu().numpy()]
    
    if verbose:
        print(f"Preselected {len(candidate_indices)} candidates from {len(censored_indices)} censored points")
    
    # Run BatchBALD on preselected candidates
    selected_indices = batchbald_selection(
        pdf=pdf,
        censor_times=censor_times,
        increment=increment,
        candidate_indices=candidate_indices,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )
    
    return selected_indices,_


def compute_oracle_entropies(
    pdf: torch.Tensor,
    censor_times: np.ndarray,
    increment: int,
    indices: np.ndarray,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute H(Y_oracle) for each point.
    
    For a point censored at time c with probe depth k:
    Y_oracle can be: {died at c, died at c+1, ..., died at c+k, still alive at c+k}
    
    H(Y_oracle) = -sum_{t=c}^{c+k} p_t log p_t - P(T > c+k) log P(T > c+k)
    
    Args:
        pdf: [N, K, T] probability distribution over time bins
        censor_times: [N] censoring time bin for each point
        increment: probe depth k
        indices: which points to compute entropy for
        device: torch device
        
    Returns:
        [len(indices)] tensor of oracle entropies
    """
    N, K, T = pdf.shape
    eps = 1e-10
    
    # Average over ensemble members to get expected probabilities
    # [N, T]
    mean_pdf = pdf.mean(dim=1)
    
    entropies = torch.zeros(len(indices), device=device)
    
    for i, idx in enumerate(indices):
        c = int(censor_times[idx])  # censoring bin
        
        # Renormalize: zero out bins before c (person survived past those)
        # P(T=t | T >= c) = P(T=t) / P(T >= c)
        probs = mean_pdf[idx].clone()  # [T]
        
        # Zero out bins before censoring time
        if c > 0:
            probs[:c] = 0
        
        # Renormalize
        total_prob = probs.sum()
        if total_prob > eps:
            probs = probs / total_prob
        
        # Define the oracle window: bins c to c+k (inclusive)
        # Everything beyond c+k collapses to "censored at c+k"
        max_reveal_bin = min(c + increment, T - 1)
        
        # Probability of each outcome the oracle can reveal
        # Outcomes: died in bin c, c+1, ..., c+k, OR still alive at c+k
        oracle_probs = []
        
        # Probability of dying in each bin within the window
        for t in range(c, max_reveal_bin + 1):
            if t < T:
                oracle_probs.append(probs[t])
        
        # Probability of still being alive at c+k (censored)
        # This is P(T > c+k) = sum of probs for bins > c+k
        if max_reveal_bin + 1 < T:
            p_censored = probs[max_reveal_bin + 1:].sum()
        else:
            p_censored = torch.tensor(0.0, device=device)
        oracle_probs.append(p_censored)
        
        oracle_probs = torch.stack(oracle_probs)
        
        # Compute entropy: -sum p log p
        # Handle zeros carefully
        log_probs = torch.log(oracle_probs + eps)
        entropy = -torch.sum(oracle_probs * log_probs)
        
        entropies[i] = entropy
    
    return entropies


def compute_conditional_oracle_entropy(
    pdf: torch.Tensor,
    censor_times: np.ndarray,
    increment: int,
    indices: np.ndarray,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute E_theta[H(Y_oracle | theta)] for each point.
    
    This is the expected entropy under each model sample, then averaged.
    
    Args:
        pdf: [N, K, T] probability distribution over time bins per ensemble member
        censor_times: [N] censoring time bin for each point
        increment: probe depth k
        indices: which points to compute entropy for
        device: torch device
        
    Returns:
        [len(indices)] tensor of conditional entropies
    """
    N, K, T = pdf.shape
    eps = 1e-10
    
    conditional_entropies = torch.zeros(len(indices), device=device)
    
    for i, idx in enumerate(indices):
        c = int(censor_times[idx])
        max_reveal_bin = min(c + increment, T - 1)
        
        # Compute entropy for each ensemble member, then average
        member_entropies = torch.zeros(K, device=device)
        
        for k in range(K):
            probs = pdf[idx, k].clone()  # [T]
            
            # Zero out and renormalize
            if c > 0:
                probs[:c] = 0
            total_prob = probs.sum()
            if total_prob > eps:
                probs = probs / total_prob
            
            # Oracle probabilities for this ensemble member
            oracle_probs = []
            for t in range(c, max_reveal_bin + 1):
                if t < T:
                    oracle_probs.append(probs[t])
            
            if max_reveal_bin + 1 < T:
                p_censored = probs[max_reveal_bin + 1:].sum()
            else:
                p_censored = torch.tensor(0.0, device=device)
            oracle_probs.append(p_censored)
            
            oracle_probs = torch.stack(oracle_probs)
            log_probs = torch.log(oracle_probs + eps)
            member_entropies[k] = -torch.sum(oracle_probs * log_probs)
        
        conditional_entropies[i] = member_entropies.mean()
    
    return conditional_entropies


def batchbald_selection(
    pdf: torch.Tensor,
    censor_times: np.ndarray,
    increment: int,
    candidate_indices: np.ndarray,
    batch_size: int,
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Greedy BatchBALD selection.
    
    BatchBALD score = I(batch; theta) = H(Y_oracle^batch) - E_theta[H(Y_oracle^batch | theta)]
    
    We greedily select points that maximize the mutual information.
    
    Args:
        pdf: [N, K, T] probability distribution
        censor_times: [N] censoring times
        increment: probe depth
        candidate_indices: indices to select from
        batch_size: number of points to select
        device: torch device
        verbose: print progress
        
    Returns:
        List of selected indices
    """
    N, K, T = pdf.shape
    n_candidates = len(candidate_indices)
    batch_size = min(batch_size, n_candidates)
    
    selected = []
    remaining = list(candidate_indices)
    
    # Precompute individual BALD scores (mutual information for single points)
    # I(x; theta) = H(Y_oracle) - E_theta[H(Y_oracle | theta)]
    marginal_entropies = compute_oracle_entropies(
        pdf, censor_times, increment, candidate_indices, device
    )
    conditional_entropies = compute_conditional_oracle_entropy(
        pdf, censor_times, increment, candidate_indices, device
    )
    bald_scores = marginal_entropies - conditional_entropies
    
    # Create mapping from original index to position in candidate list
    idx_to_pos = {idx: pos for pos, idx in enumerate(candidate_indices)}
    
    for b in range(batch_size):
        if verbose and b % 10 == 0:
            print(f"Selecting point {b+1}/{batch_size}")
        
        best_score = -float('inf')
        best_idx = None
        
        if len(selected) == 0:
            # First point: just use BALD score
            for idx in remaining:
                pos = idx_to_pos[idx]
                score = bald_scores[pos].item()
                if score > best_score:
                    best_score = score
                    best_idx = idx
        else:
            # Subsequent points: compute joint mutual information gain
            # For efficiency, we approximate using pairwise redundancy
            for idx in remaining:
                pos = idx_to_pos[idx]
                
                # Start with individual BALD score
                score = bald_scores[pos].item()
                
                # Subtract redundancy with already selected points
                # This is an approximation to the full joint entropy computation
                redundancy = compute_pairwise_redundancy(
                    pdf, censor_times, increment, idx, selected, device
                )
                score -= redundancy
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
        
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
            if verbose:
                print(f"  Selected index {best_idx} with score {best_score:.4f}")
    
    return selected


def compute_pairwise_redundancy(
    pdf: torch.Tensor,
    censor_times: np.ndarray,
    increment: int,
    new_idx,
    selected_indices,
    device: str = "cpu",
) -> float:
    """
    Compute redundancy between a new point and already selected points.
    
    Approximation: sum of pairwise mutual informations
    I(x_new; x_selected) ≈ sum_i I(x_new; x_i)
    
    where I(x_new; x_i) = H(Y_oracle^new) + H(Y_oracle^i) - H(Y_oracle^new, Y_oracle^i)
    
    Args:
        pdf: [N, K, T] probability distribution
        censor_times: [N] censoring times
        increment: probe depth
        new_idx: index of new point
        selected_indices: already selected indices
        device: torch device
        
    Returns:
        Total redundancy score
    """
    if len(selected_indices) == 0:
        return 0.0
    
    N, K, T = pdf.shape
    eps = 1e-10
    total_redundancy = 0.0
    
    for sel_idx in selected_indices:
        # Compute joint entropy H(Y_oracle^new, Y_oracle^sel)
        # and compare with sum of marginals
        
        c_new = int(censor_times[new_idx])
        c_sel = int(censor_times[sel_idx])
        
        max_bin_new = min(c_new + increment, T - 1)
        max_bin_sel = min(c_sel + increment, T - 1)
        
        # Number of outcomes for each point
        n_outcomes_new = (max_bin_new - c_new + 1) + 1  # bins + censored
        n_outcomes_sel = (max_bin_sel - c_sel + 1) + 1
        
        # Compute joint distribution over oracle outcomes
        # Average over ensemble members
        joint_entropy_per_member = torch.zeros(K, device=device)
        marginal_new_per_member = torch.zeros(K, device=device)
        marginal_sel_per_member = torch.zeros(K, device=device)
        
        for k in range(K):
            # Get oracle probabilities for new point
            probs_new = pdf[new_idx, k].clone()
            if c_new > 0:
                probs_new[:c_new] = 0
            if probs_new.sum() > eps:
                probs_new = probs_new / probs_new.sum()
            
            oracle_new = []
            for t in range(c_new, max_bin_new + 1):
                if t < T:
                    oracle_new.append(probs_new[t])
            if max_bin_new + 1 < T:
                oracle_new.append(probs_new[max_bin_new + 1:].sum())
            else:
                oracle_new.append(torch.tensor(0.0, device=device))
            oracle_new = torch.stack(oracle_new)
            
            # Get oracle probabilities for selected point
            probs_sel = pdf[sel_idx, k].clone()
            if c_sel > 0:
                probs_sel[:c_sel] = 0
            if probs_sel.sum() > eps:
                probs_sel = probs_sel / probs_sel.sum()
            
            oracle_sel = []
            for t in range(c_sel, max_bin_sel + 1):
                if t < T:
                    oracle_sel.append(probs_sel[t])
            if max_bin_sel + 1 < T:
                oracle_sel.append(probs_sel[max_bin_sel + 1:].sum())
            else:
                oracle_sel.append(torch.tensor(0.0, device=device))
            oracle_sel = torch.stack(oracle_sel)
            
            # Joint distribution (assuming conditional independence given theta)
            # P(Y_new, Y_sel | theta) = P(Y_new | theta) * P(Y_sel | theta)
            joint = torch.outer(oracle_new, oracle_sel)  # [n_new, n_sel]
            
            # Compute entropies
            joint_flat = joint.flatten()
            joint_entropy_per_member[k] = -torch.sum(joint_flat * torch.log(joint_flat + eps))
            marginal_new_per_member[k] = -torch.sum(oracle_new * torch.log(oracle_new + eps))
            marginal_sel_per_member[k] = -torch.sum(oracle_sel * torch.log(oracle_sel + eps))
        
        # E[H(Y_new, Y_sel | theta)] - this is what we computed above averaged
        expected_joint_entropy = joint_entropy_per_member.mean()
        
        # For mutual information under the model, we need:
        # I(Y_new; Y_sel | theta) = H(Y_new | theta) + H(Y_sel | theta) - H(Y_new, Y_sel | theta)
        # But given theta, they're independent, so I(Y_new; Y_sel | theta) = 0
        
        # The redundancy we care about is in the marginal (averaged) distribution
        # We approximate this by computing MI in the averaged distribution
        
        # Average oracle distributions
        avg_oracle_new = torch.zeros(n_outcomes_new, device=device)
        avg_oracle_sel = torch.zeros(n_outcomes_sel, device=device)
        avg_joint = torch.zeros(n_outcomes_new, n_outcomes_sel, device=device)
        
        for k in range(K):
            probs_new = pdf[new_idx, k].clone()
            if c_new > 0:
                probs_new[:c_new] = 0
            if probs_new.sum() > eps:
                probs_new = probs_new / probs_new.sum()
            
            oracle_new = []
            for t in range(c_new, max_bin_new + 1):
                if t < T:
                    oracle_new.append(probs_new[t])
            if max_bin_new + 1 < T:
                oracle_new.append(probs_new[max_bin_new + 1:].sum())
            else:
                oracle_new.append(torch.tensor(0.0, device=device))
            oracle_new = torch.stack(oracle_new)
            
            probs_sel = pdf[sel_idx, k].clone()
            if c_sel > 0:
                probs_sel[:c_sel] = 0
            if probs_sel.sum() > eps:
                probs_sel = probs_sel / probs_sel.sum()
            
            oracle_sel = []
            for t in range(c_sel, max_bin_sel + 1):
                if t < T:
                    oracle_sel.append(probs_sel[t])
            if max_bin_sel + 1 < T:
                oracle_sel.append(probs_sel[max_bin_sel + 1:].sum())
            else:
                oracle_sel.append(torch.tensor(0.0, device=device))
            oracle_sel = torch.stack(oracle_sel)
            
            avg_oracle_new += oracle_new / K
            avg_oracle_sel += oracle_sel / K
            avg_joint += torch.outer(oracle_new, oracle_sel) / K
        
        # Mutual information in averaged distribution
        # I(Y_new; Y_sel) = H(Y_new) + H(Y_sel) - H(Y_new, Y_sel)
        h_new = -torch.sum(avg_oracle_new * torch.log(avg_oracle_new + eps))
        h_sel = -torch.sum(avg_oracle_sel * torch.log(avg_oracle_sel + eps))
        h_joint = -torch.sum(avg_joint * torch.log(avg_joint + eps))
        
        mi = h_new + h_sel - h_joint
        total_redundancy += max(0, mi.item())  # Ensure non-negative
    
    return total_redundancy


