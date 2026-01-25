"""
Consolidated utilities module.
Combines: utils, Useful_functions, prediction_utils, plots, hyper_params
"""

import numpy as np
import pandas as pd
import random
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional, Union
import statistics
import argparse
import json
import pickle
from sklearn.utils import shuffle
from skmultilearn.model_selection import iterative_train_test_split
from datetime import datetime
from scipy.stats import qmc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# Type aliases
Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]


# =============================================================================
# HYPERPARAMETERS (from hyper_params.py)
# =============================================================================

def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="SUPPORT",
                        choices=["Synthetic-I", "Synthetic-II", "Synthetic-III", "SUPPORT", "NACD", "MIMIC"])
    parser.add_argument('--model', type=str, default="BayesianMTLR",
                        choices=["MTLR", "BayesianHorseshoeLinearMTLR", "BayesianElementwiseMTLR", 
                                 "BayesianHorseshoeMTLR", "BayesianLinearMTLR", "BayesianMTLR",
                                 "CoxPH", "BayesianHorseshoeLinearCox", "BayesianElementwiseCox", 
                                 "BayesianHorseshoeCox", "BayesianLinearCox", "BayesianCox"])
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--early_stop', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=39)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.00008)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--mu_scale', type=float, default=None)
    parser.add_argument('--rho_scale', type=float, default=-5.)
    parser.add_argument('--n_samples_train', type=int, default=10)
    parser.add_argument('--n_samples_test', type=int, default=100)
    parser.add_argument('--pi', type=float, default=0.5)
    parser.add_argument('--sigma1', type=float, default=1)
    parser.add_argument('--sigma2', type=float, default=math.exp(-6))
    parser.add_argument('--weight_cauchy_scale', type=float, default=1)
    parser.add_argument('--global_cauchy_scale', type=float, default=1)
    parser.add_argument('--c1', type=float, default=0.01)
    parser.add_argument('--batch_method', type=str, default='random')
    args = parser.parse_args()
    return args


def load_parser(file: str):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(file, 'r') as f:
        args.__dict__ = json.load(f)
    return args


# =============================================================================
# CORE UTILITIES (from utils.py)
# =============================================================================

def save_params(config: argparse.Namespace) -> str:
    """Save the parameters to a json file."""
    path = os.path.join("results", config.dataset, config.model, config.timestamp)
    os.makedirs(path, exist_ok=True)
    config_dict = config.__dict__.copy()
    if 'device' in config_dict:
        config_dict['device'] = str(config_dict['device'])
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    return path


def save_predictions(path: str, exp_num: int, model, data_test: pd.DataFrame) -> None:
    """Saves model and test set."""
    subpath = os.path.join(path, f'split_{exp_num}')
    os.makedirs(subpath, exist_ok=True)
    torch.save(model.state_dict(), f"{subpath}/model.pt")
    data_test.to_pickle(f"{subpath}/testset.pkl")


def print_performance(con=None, ibs=None, l1_unc=None, l1_hinge=None, 
                     l1_margin=None, pvalues=None, cvge=None, thk=None, path=None) -> None:
    """Print performance using mean and std."""
    prf = ""
    if con: prf += f"Concordance: {statistics.mean(con):.4f} ± {statistics.stdev(con):.4f}\n"
    if ibs: prf += f"IBS: {statistics.mean(ibs):.4f} ± {statistics.stdev(ibs):.4f}\n"
    if l1_unc: prf += f"L1-uncensored: {statistics.mean(l1_unc):.4f} ± {statistics.stdev(l1_unc):.4f}\n"
    if l1_hinge: prf += f"L1-hinge: {statistics.mean(l1_hinge):.4f} ± {statistics.stdev(l1_hinge):.4f}\n"
    if l1_margin: prf += f"L1-margin: {statistics.mean(l1_margin):.4f} ± {statistics.stdev(l1_margin):.4f}\n"
    if pvalues: prf += f"D-Calibration: calibrated {sum(i >= 0.05 for i in pvalues)}/{len(pvalues)}\n"
    if thk: prf += f"Thickness: {statistics.mean(thk):.4f} ± {statistics.stdev(thk):.4f}\n"
    print(prf)
    if path:
        with open(f"{path}/performance.txt", 'w') as f:
            f.write(prf)


def is_monotonic(array):
    return (all(array[i] <= array[i + 1] for i in range(len(array) - 1)) or
            all(array[i] >= array[i + 1] for i in range(len(array) - 1)))


def make_monotonic(array):
    for i in range(len(array) - 1):
        if not array[i] >= array[i + 1]:
            array[i + 1] = array[i]
    return array


def compute_unique_counts(event, time, order=None):
    """Count right censored and uncensored samples at each unique time point."""
    n_samples = event.shape[0]
    if order is None:
        order = torch.argsort(time)

    uniq_times = torch.empty(n_samples, dtype=time.dtype, device=time.device)
    uniq_events = torch.empty(n_samples, dtype=torch.int, device=time.device)
    uniq_counts = torch.empty(n_samples, dtype=torch.int, device=time.device)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1
            count += 1
            i += 1
        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1
        if i == n_samples:
            break
        prev_val = time[order[i]]

    uniq_times = uniq_times[:j]
    uniq_events = uniq_events[:j]
    uniq_counts = uniq_counts[:j]
    n_censored = uniq_counts - uniq_events
    total_count = torch.cat([torch.tensor([0], device=uniq_counts.device), uniq_counts], dim=0)
    n_at_risk = n_samples - torch.cumsum(total_count, dim=0)
    return uniq_times, uniq_events, n_at_risk[:-1], n_censored

#commented out to see if it breaks anything..
def reformat_survival(dataset: pd.DataFrame, time_bins: NumericArrayLike):
    """Reformat survival data for training."""
    x = torch.tensor(dataset.drop(["time", "event"], axis=1).values, dtype=torch.float)
    y = encode_survival(dataset["time"].values, dataset["event"].values, time_bins)
    return x, y


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


#commented to see if it breaks anything..
# def make_time_bins(times, num_bins=None, use_quantiles=True, event=None):
#     """Creates bins for survival time discretisation."""
#     if event is not None:
#         times = times[event == 1]
#     if num_bins is None:
#         num_bins = math.ceil(math.sqrt(len(times)))
#     if use_quantiles:
#         bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
#     else:
#         bins = np.linspace(times.min(), times.max(), num_bins)
#     return torch.tensor(bins, dtype=torch.float)


def train_val_test_stratified_split(df, stratify_colname='event', frac_train=0.5, 
                                     frac_val=0.0, frac_test=0.5, random_state=None):
    """Stratified train/val/test split."""
    frac_sum = frac_train + frac_val + frac_test
    frac_train, frac_val, frac_test = frac_train/frac_sum, frac_val/frac_sum, frac_test/frac_sum
    
    X = df.values
    columns = df.columns
    
    if stratify_colname == 'event':
        stra_lab = df[stratify_colname]
    elif stratify_colname == 'time':
        stra_lab = df[stratify_colname]
        bins = np.linspace(start=stra_lab.min(), stop=stra_lab.max(), num=20)
        stra_lab = np.digitize(stra_lab, bins, right=True)
    elif stratify_colname == "both":
        t = df["time"]
        bins = np.linspace(start=t.min(), stop=t.max(), num=20)
        t = np.digitize(t, bins, right=True)
        e = df["event"]
        stra_lab = np.stack([t, e], axis=1)
    else:
        raise ValueError("unrecognized stratify policy")

    X, stra_lab = shuffle(X, stra_lab, random_state=random_state)
    x_train, _, x_temp, y_temp = iterative_train_test_split(X, stra_lab, test_size=(1.0 - frac_train))
    
    if frac_val == 0:
        x_val, x_test = [], x_temp
    else:
        x_val, _, x_test, _ = iterative_train_test_split(x_temp, y_temp, test_size=frac_test/(frac_val + frac_test))
    
    return pd.DataFrame(x_train, columns=columns), pd.DataFrame(x_val, columns=columns), pd.DataFrame(x_test, columns=columns)


def two_sided_olshen(cloud, coverage, B=10):
    """Two-sided Olshen credible interval."""
    cloud, fix = degenerate_fix_factory(cloud)
    bootstraps = np.random.choice(np.arange(cloud.shape[0]), size=(B, cloud.shape[0]))
    bootstraps = torch.tensor(bootstraps)
    clouds = cloud[bootstraps]
    maxes = torch.empty((B, cloud.shape[0]))
    for i, cloud_b in enumerate(clouds):
        zscores_ = torch.empty_like(cloud)
        for j, col in enumerate(cloud_b.T):
            median = col.median()
            above_mask = col >= median
            below_mask = col <= median
            above = col[above_mask]
            below = col[below_mask]
            zscores_[above_mask, j] = (above - median) / ((median - above).square().sum() / (above.shape[0] - 1)).sqrt()
            zscores_[below_mask, j] = (below - median) / ((median - below).square().sum() / (below.shape[0] - 1)).sqrt()
        maxes[i] = zscores_.nan_to_num().abs().max(dim=-1)[0]
    median_cloud = torch.median(cloud, dim=0)[0]
    sigma_minus = torch.empty_like(median_cloud)
    sigma_plus = torch.empty_like(median_cloud)
    for i, col in enumerate(cloud.T):
        median = median_cloud[i]
        above = col[col > median]
        below = col[col < median]
        sigma_plus[i] = ((above - median).square().sum() / (above.shape[0] - 1)).sqrt()
        sigma_minus[i] = ((below - median).square().sum() / (below.shape[0] - 1)).sqrt()

    def helper(p):
        k = torch.quantile(maxes, q=p, interpolation='higher')
        upper_bounds = median_cloud + k * sigma_plus
        lower_bounds = median_cloud - k * sigma_minus
        orthotope = fix(torch.stack([lower_bounds, upper_bounds]))
        orthotope = surv_orthotope(orthotope)
        return orthotope

    if isinstance(coverage, float):
        return helper(coverage)
    elif isinstance(coverage, (list, np.ndarray, torch.Tensor)):
        return [helper(p) for p in coverage]
    elif isinstance(coverage, dict):
        return {p: helper(p) for p in coverage}


def degenerate_fix_factory(cloud, epsilon=1e-6):
    degenerate = torch.isclose(cloud.max(dim=0)[0], cloud.min(dim=0)[0])
    deg_cloud = cloud[:, degenerate]
    nondeg_cloud = cloud[:, ~degenerate]
    mean_deg = deg_cloud.mean(dim=0)
    lower_bounds = mean_deg - epsilon
    upper_bounds = mean_deg + epsilon

    def fix(nondeg_orthotope):
        fixed_orthotope = torch.empty((2, cloud.shape[1]))
        fixed_orthotope[:, degenerate] = torch.stack([lower_bounds, upper_bounds])
        fixed_orthotope[:, ~degenerate] = nondeg_orthotope
        return fixed_orthotope
    return nondeg_cloud, fix


def surv_orthotope(orthotope):
    orthotope = torch.clamp(orthotope, 0, 1)
    inv_idx = torch.arange(orthotope.shape[1] - 1, -1, -1).long()
    orthotope[0] = torch.cummax(orthotope[0, inv_idx], dim=0)[0][inv_idx]
    orthotope[1] = torch.cummin(orthotope[1], dim=0)[0]
    return orthotope


# =============================================================================
# USEFUL FUNCTIONS (from Useful_functions.py)
# =============================================================================

def max_indices(list_of_lists):
    """Return index of max value in each inner list."""
    return [row.index(max(row)) for row in list_of_lists]


def assign_to_bins(y_test, time_bins):
    """Assign time values to bin indices."""
    if time_bins[0] > 0:
        time_bins = np.concatenate(([0], time_bins))
    bin_indices = np.digitize(y_test, time_bins) - 1
    bin_indices[bin_indices == len(time_bins) - 1] = len(time_bins) - 2
    return bin_indices


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


def remove_nan_rows(X, y, e):
    """Remove rows with NaN values."""
    valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & ~np.isnan(e)
    return X[valid_indices], y[valid_indices], e[valid_indices]


def random_acquire(X_pool, batch_size):
    """Random sampling from pool."""
    return np.random.choice(len(X_pool), batch_size, replace=False)


def map_indices(censoredtime, time_bins):
    """Map times to bin indices."""
    np1 = censoredtime.copy()
    np2_sorted = np.sort(time_bins.copy())
    np3 = np.zeros_like(np1)
    for i, value in enumerate(np1):
        idx = np.searchsorted(np2_sorted, value, side='right') - 1
        if idx == len(np2_sorted):
            idx = len(np2_sorted) - 1
        np3[i] = idx
    return np3


def batch_modify_and_normalize_logits(my_tensor, list_a, list_b):
    """Modify and normalize logits for time window."""
    batch_size = my_tensor.shape[0]
    cols = my_tensor.shape[2]

    for i in range(batch_size):
        a = int(list_a[i])
        b = int(list_b[i])
        my_tensor[i, :, :a] = 0
        if b + 1 < cols:
            my_tensor[i, :, b+1] = my_tensor[i, :, b+1:].sum(dim=1)
            if b + 2 < cols:
                my_tensor[i, :, b+2:] = 0

    row_sums = my_tensor.sum(dim=2, keepdim=True)
    row_sums[row_sums == 0] = 1
    return my_tensor / row_sums


#commented out to see if it will break anything or if this function is just useless..
# def fix_survival_timebins_increment(mylogits, time_bins, x_test, inc):
#     """Fix survival time bins with increment."""
#     time_bins_copy = np.array(time_bins).copy()
#     temp_bins = np.array([0] + list(time_bins_copy)[:-1])
#     censoredtime = np.array(x_test['time'])
#     censoredbin = map_indices(censoredtime, temp_bins)
#     binincrements = map_indices(censoredtime + inc, temp_bins)
#     return batch_modify_and_normalize_logits(mylogits, censoredbin, binincrements)


def generate_costs(length, min_cost, max_cost):
    """Generate random costs."""
    costlist = np.random.randint(min_cost, max_cost+1, size=length)
    return pd.DataFrame(costlist, columns=['Cost'])


def take_most_within_budget(values, costlist, budget):
    """Select highest values within budget."""
    values = values.squeeze(1) if values.dim() > 1 else values
    sorted_indices = torch.argsort(values, descending=True)
    total_cost = 0
    selected_indices = []

    if isinstance(costlist, torch.Tensor):
        costlist = costlist.squeeze()

    for idx in sorted_indices:
        idx = idx.item()
        cost = costlist[idx].item() if isinstance(costlist[idx], torch.Tensor) else costlist[idx]
        if total_cost + cost <= budget:
            selected_indices.append(idx)
            total_cost += cost
    return selected_indices


def initialize_with_lhs(x_min, x_max, n_samples, random_state=42):
    """Initialize with Latin Hypercube Sampling."""
    sampler = qmc.LatinHypercube(d=len(x_min), seed=random_state)
    samples = sampler.random(n=n_samples)
    x_min, x_max = np.array(x_min), np.array(x_max)
    if x_min.ndim == 0: x_min = np.array([x_min])
    if x_max.ndim == 0: x_max = np.array([x_max])
    return qmc.scale(samples, x_min, x_max)


def initialize_with_kmeans(X_pool, n_clusters, random_state=42):
    """Initialize with K-means clustering."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pool)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_scaled)
    closest_indices = []
    for center in kmeans.cluster_centers_:
        distances = np.sqrt(np.sum((X_scaled - center) ** 2, axis=1))
        closest_indices.append(np.argmin(distances))
    return closest_indices


def get_acquired_indices(acquisition_function, model, masked_X_labeled, batch_size, 
                         time_bins, config, device='CUDA', in_data_train=None, 
                         increment=None, budget=None, costlist=None, num_samples=20000, typu=0):
    """Get acquired indices using acquisition function."""
    if costlist is None:
        newcostlist = pd.DataFrame([1] * len(masked_X_labeled), columns=['Cost'])
    else:
        newcostlist = costlist

    if acquisition_function.__name__ == 'random_knapsack':
        return acquisition_function(newcostlist, budget, type=typu)
    elif 'batchbald' in acquisition_function.__name__:
        return acquisition_function(model, masked_X_labeled, batch_size, time_bins, config, 
                                   device=device, in_data_train=in_data_train, 
                                   increment=increment, budget=budget, 
                                   costlist=newcostlist, num_samples=num_samples)
    else:
        return acquisition_function(model, masked_X_labeled, batch_size, time_bins, config,
                                   device=device, in_data_train=in_data_train,
                                   increment=increment, budget=budget, costlist=newcostlist)


def fix_y_pred(y_pred):
    """Fix prediction format."""
    y_pred_np = np.array(y_pred)
    differences = y_pred_np[:, :-1] - y_pred_np[:, 1:]
    last_column = y_pred_np[:, -1].reshape(-1, 1)
    y_pred_transformed_np = np.concatenate((differences, last_column), axis=1)
    return max_indices(y_pred_transformed_np.tolist())


# =============================================================================
# PREDICTION UTILITIES (from prediction_utils.py)
# =============================================================================

def make_prediction(model, x_test, time_bins, args):
    """Make survival predictions."""
    if isinstance(x_test, np.ndarray):
        x_test = torch.tensor(x_test, dtype=torch.float32)
    
    # Import locally to avoid circular imports
    from model import mtlr_survival, BayesEleMtlr, BayesLinMtlr, CoxPH, BayesEleCox, BayesLinCox
    
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'sample_elbo'):  # Bayesian models
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
    
    # Only prepend 0 if time_bins doesn't already start with 0
    if time_bins[0] != 0:
        zero_tensor = torch.tensor([0], dtype=time_bins.dtype, device=time_bins.device)
        time_bins = torch.cat([zero_tensor, time_bins])
    
    return mean_survival_outputs, time_bins, survival_outputs


def ensemble_to_pdf(ensemble_outputs, device='cpu'):
        K, N, T = ensemble_outputs.shape
        surv = ensemble_outputs.permute(1, 0, 2).contiguous()  # [N,K,T]
        surv = torch.clamp(surv, 1e-12, 1.0)

        S_prev = torch.ones((N, K, 1), device=device, dtype=surv.dtype)
        S_shift = torch.cat([S_prev, surv[:, :, :-1]], dim=2)
        pdf = torch.clamp(S_shift - surv, min=0.0)
        pdf = pdf / torch.clamp(pdf.sum(dim=2, keepdim=True), min=1e-12)
        return pdf
# =============================================================================
# PLOTTING UTILITIES (from plots.py)
# =============================================================================

def plot_curve_with_bar(time_bins, mean_outputs, upper_outputs, lower_outputs, index=0, save_path=None):
    """Plot survival curve with credible interval."""
    plt.rcParams["figure.figsize"] = [4, 3]
    plt.plot(time_bins.cpu().numpy(), mean_outputs.cpu().numpy()[index, :], '-')
    plt.fill_between(time_bins.cpu().numpy(), upper_outputs.cpu().numpy()[index, :],
                     lower_outputs.cpu().numpy()[index, :], color='gray', alpha=0.2)
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def plot_weights_dist(means, variances, feature_names, path=None):
    """Plot weight distribution."""
    y_pos = np.arange(len(feature_names))
    fig, ax = plt.subplots()
    ax.barh(y_pos, np.abs(means), xerr=variances)
    ax.set_xlabel('|Weight|')
    plt.yticks(y_pos, feature_names)
    plt.tight_layout()
    if path:
        plt.savefig(f"{path}.png", dpi=300)
    else:
        plt.show()


def plot_weights_hist(means):
    """Plot weights histogram."""
    log_means = np.log10(np.abs(means))
    fig, ax = plt.subplots()
    ax.hist(log_means, bins=20)
    ax.set_xlabel('log(|Weight|)')
    ax.set_ylabel('Counts')
    plt.title("Weights Histogram")
    plt.tight_layout()
    plt.show()


# =============================================================================
# Visualization Utilities
# =============================================================================

import umap  # pip install umap-learn
# import plotly.express as px


def umap_compare_2d_two_sets(
    X_train,
    X_test,
    censored_indices,
    n_fit_train=30000,      # how many train points to fit scaler+UMAP on
    n_test=30000,           # how many test points to plot
    n_interesting=None,     # optionally downsample interesting points
    random_state=42,
    n_neighbors=30,
    min_dist=0.05,
    metric="euclidean",
):
    rng = np.random.default_rng(random_state)

    # -------------------------
    # 1) choose train subset to FIT UMAP on
    # -------------------------
    if True:
        fit_idx = np.arange(len(X_train))
        if len(fit_idx) > n_fit_train:
            fit_idx = rng.choice(fit_idx, size=n_fit_train, replace=False)
        X_fit = X_train[fit_idx]

    # -------------------------
    # 2) choose test points to PLOT
    # -------------------------
    if True:
        te_idx = np.arange(len(X_test))
        if len(te_idx) > n_test:
            te_idx = rng.choice(te_idx, size=n_test, replace=False)
        X_te = X_test[te_idx]

    # -------------------------
    # 3) choose interesting points to PLOT
    # -------------------------
    if True:
        X_ip = X_train[censored_indices]
        if (n_interesting is not None) and (len(X_ip) > n_interesting):
            ip_idx = rng.choice(np.arange(len(X_ip)), size=n_interesting, replace=False)
            X_ip = X_ip[ip_idx]

    # -------------------------
    # 4) scale using TRAIN-FIT subset only
    # -------------------------
    if True:
        scaler = StandardScaler()
        X_fit_s = scaler.fit_transform(X_fit)
        X_te_s  = scaler.transform(X_te)
        X_ip_s  = scaler.transform(X_ip)
        print("len(X_fit_s), len(X_te_s), len(X_ip_s): ", len(X_fit_s), len(X_te_s), len(X_ip_s))

    # -------------------------
    # 5) fit UMAP on TRAIN-FIT subset only
    # -------------------------
    if True:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        reducer.fit(X_fit_s)

        Z_te = reducer.transform(X_te_s)
        Z_ip = reducer.transform(X_ip_s)

    # -------------------------
    # 6) plot ONLY the two sets
    # -------------------------
    if True:
        df_te = pd.DataFrame({"u1": Z_te[:, 0], "u2": Z_te[:, 1], "set": "test"})
        df_ip = pd.DataFrame({"u1": Z_ip[:, 0], "u2": Z_ip[:, 1], "set": "interesting"})
        df = pd.concat([df_te, df_ip], ignore_index=True)

        fig = px.scatter(
            df, x="u1", y="u2",
            color="set",
            title="UMAP: Test (red) vs Interesting (green)",
            opacity=0.55
        )

        # Force exact colors + make interesting slightly bigger
        fig.update_traces(marker=dict(size=4))
        fig.for_each_trace(lambda t: t.update(
            marker=dict(color=("red" if t.name == "test" else "green"),
                        size=(4 if t.name == "test" else 6),
                        opacity=(0.45 if t.name == "test" else 0.9))
        ))

        fig.show()



def umap_train_then_color_interesting(
    X_train,
    interesting_points,
    scores,
    n_fit_train=30000,          # how many train points to fit scaler+UMAP on
    show_train_background=True, # plot faint train background for context
    n_train_plot=20000,         # how many train points to plot (if background True)
    random_state=42,
    n_neighbors=30,
    min_dist=0.05,
    metric="euclidean",
    title="UMAP: interesting points colored by score",
):
    scores = np.asarray(scores).reshape(-1)
    if len(scores) != len(interesting_points):
        raise ValueError(f"`scores` must have same length as `interesting_points`. Got {len(scores)} vs {len(interesting_points)}")

    rng = np.random.default_rng(random_state)

    # -------------------------
    # 1) choose train subset to FIT on (scaler+UMAP)
    # -------------------------
    fit_idx = np.arange(len(X_train))
    if len(fit_idx) > n_fit_train:
        fit_idx = rng.choice(fit_idx, size=n_fit_train, replace=False)
    X_fit = X_train[fit_idx]

    # -------------------------
    # 2) scale using train-fit only
    # -------------------------
    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit)
    X_ip_s  = scaler.transform(interesting_points)

    # -------------------------
    # 3) fit UMAP on train-fit only, then transform interesting
    # -------------------------
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    reducer.fit(X_fit_s)

    Z_ip = reducer.transform(X_ip_s)

    # Optional: background train points (project & downsample)
    df_bg = None
    if show_train_background:
        # transform a subset of train for plotting background
        plot_idx = np.arange(len(X_train))
        if len(plot_idx) > n_train_plot:
            plot_idx = rng.choice(plot_idx, size=n_train_plot, replace=False)
        X_tr_plot_s = scaler.transform(X_train[plot_idx])
        Z_tr = reducer.transform(X_tr_plot_s)
        df_bg = pd.DataFrame({"u1": Z_tr[:, 0], "u2": Z_tr[:, 1]})

    # -------------------------
    # 4) build plotly figure
    # -------------------------
    df_ip = pd.DataFrame({"u1": Z_ip[:, 0], "u2": Z_ip[:, 1], "score": scores})

    fig = px.scatter(
        df_ip,
        x="u1", y="u2",
        color="score",
        color_continuous_scale="Viridis",
        title=title,
        opacity=0.95,
    )
    fig.update_traces(marker=dict(size=6))

    if df_bg is not None:
        # add background as faint gray points behind
        fig.add_scatter(
            x=df_bg["u1"], y=df_bg["u2"],
            mode="markers",
            name="train (background)",
            marker=dict(size=3, opacity=0.08, color="gray"),
            hoverinfo="skip",
        )

    fig.show()

    # optionally return reducer+scaler so you can reuse them
    return reducer, scaler


