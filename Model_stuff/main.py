"""
Main script for Survival Analysis with Active Learning.
Consolidated from the original minimal_folder.
"""

import os
import sys
import copy
import time
import random
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

# Local imports
from data import Get_Dataset
from model import BayesLinMtlr, mtlr_survival, initialize_bayesian_model
from utils import (save_params,artificially_censor_true, remove_nan_rows, make_prediction)
from evaluation import Mae_Po, concordance, brier_score, integrated_brier_score
from acquisition import (batchbald_acquire_budget,
                         random_knapsack, entropy_of_probs, variance_of_probs,
                         clostest_to_half_in_increment, mean_closest_to_middle,
                         Using_Clusters_for_batch, highest_death_probability_in_window,
                         cbald_censored_regression,batchbald_acquire_budget_DEBUG, test_bald_score, test_batchbald_score)


def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#didnt understand this code
def stratified_train_test_split(X, y, e, test_size=0.25, random_state=42):
    """Stratified split ensuring similar event/time distributions."""
    try:
        time_quartiles = pd.qcut(y, q=4, labels=[0, 1, 2, 3], duplicates='drop')
    except ValueError:
        time_quartiles = pd.qcut(y, q=2, labels=[0, 1], duplicates='drop')
    
    stratify_labels = e * 10 + np.array(time_quartiles).astype(int)
    unique, counts = np.unique(stratify_labels, return_counts=True)
    
    if counts.min() < 2:
        stratify_labels = e
    
    return train_test_split(X, y, e, test_size=test_size, 
                           random_state=random_state, stratify=stratify_labels)


#double check outputs..
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


def train_model(model, data_train, time_bins, config, path, device, 
                num_epochs=100, patience=20, verbose=True, reset_model=True):
    """Train the Bayesian model with early stopping."""
    from utils import reformat_survival
    from model import mtlr_nll
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import StratifiedKFold
    
    torch.manual_seed(42)
    np.random.seed(42)
    if reset_model:
        model.reset_parameters()
    
    X = data_train.drop(["time", "event"], axis=1).values
    y = data_train["time"].values
    e = data_train["event"].values
    
    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).to(device)
    e = torch.FloatTensor(e).to(device)
    
    train_size = int(0.9 * len(X))
    
    try:
        train_indices, val_indices = next(StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(X.cpu(), e.cpu()))
    except:
        indices = torch.randperm(len(X))
        train_indices = indices[:train_size].numpy()
        val_indices = indices[train_size:].numpy()
    
    x_train, y_train, e_train = X[train_indices], y[train_indices], e[train_indices]
    x_val, y_val, e_val = X[val_indices], y[val_indices], e[val_indices]
    
    train_df = pd.DataFrame(x_train.cpu().numpy(), columns=[f'f{i}' for i in range(x_train.shape[1])])
    train_df['time'] = y_train.cpu().numpy()
    train_df['event'] = e_train.cpu().numpy()
    # encode_survival creates bins.shape[0] + 1 output columns.
    # Since time_bins has 11 elements (including leading 0), we pass time_bins[1:] (10 elements)
    # to get 11 output columns, matching the model's expected output shape.
    x_formatted, y_formatted = reformat_survival(train_df, time_bins[1:])
    
    val_df = pd.DataFrame(x_val.cpu().numpy(), columns=[f'f{i}' for i in range(x_val.shape[1])])
    val_df['time'] = y_val.cpu().numpy()
    val_df['event'] = e_val.cpu().numpy()
    # Same reason: pass time_bins[1:] to get 11 output columns
    x_val_formatted, y_val_formatted = reformat_survival(val_df, time_bins[1:])
    
    train_dataset = TensorDataset(x_formatted, y_formatted)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    best_val_loss = float('inf')
    best_ep = 0
    best_state = None
    total_loss = 0
    
    pbar = tqdm(range(num_epochs), desc="Training", disable=not verbose)
    for epoch in pbar:
        model.train()
        total_loss = 0
        
        for xi, yi in train_loader:
            xi, yi = xi.to(device), yi.to(device)
            optimizer.zero_grad()
            loss, _, _, _ = model.sample_elbo(xi, yi, train_size, see1=config.c1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() / train_size
        
        model.eval()
        with torch.no_grad():
            val_loss, _, _, _ = model.sample_elbo(x_val_formatted, y_val_formatted, 
                                                   dataset_size=len(val_indices), see1=config.c1)
            val_loss /= len(val_indices)
        
        pbar.set_postfix({"Train": f"{total_loss:.4f}", "Val": f"{val_loss.item():.4f}"})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ep = epoch
            best_state = model.state_dict().copy()
            torch.save(best_state, path)
        elif epoch - best_ep > patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, total_loss


def General_Function(dataset="NACD", initial_data_config=[100, 1000], increments=[60],
                     budget=30, acquisition_functions=[entropy_of_probs],
                     this_costlist=None, model_retrain=False, num_epochs=100,
                     verbose=False, num_bins=10, use_saved_base_model=True,
                     early_stopping_patience=20):
    """
    Main function for survival analysis with active learning.
    """

    if True: #device and dataset preparation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        datapreserver, args = Get_Dataset(dataset)
        
        # Update config ... what do these mean?
        args.num_epochs = num_epochs
        args.device = device
        args.learning_rate = 8e-5
        args.batch_size = 32
        args.dropout_rate = 0.2
        args.l2_penalty = 1e-4
        args.hidden_size = 50
        args.num_time_bins = num_bins
        args.pi = 0.5
        args.sigma1 = 1.0
        args.sigma2 = 0.0025
        args.rho_scale = -3.0
        args.mu_scale = 0.1
        
        base_model_path = f"saved_models/{dataset}_base_model.pth"
        os.makedirs("saved_models", exist_ok=True)
        
        increment_list = []
    
    for increment in increments:
        if True: #Data preparacion
            data = datapreserver.copy()
            columns = data.columns
            X = data.drop(["time", "event"], axis=1).values
            y = data["time"].values
            e = data["event"].values
            
            X, y, e = remove_nan_rows(X, y, e)
            
            total_n = len(X)
            train_size = initial_data_config[1]
            test_size = max(0.1, min(0.5, 1.0 - (train_size / total_n)))
            
            X_train, X_test, y_train, y_test, e_train, e_test = stratified_train_test_split(
                X, y, e, test_size=test_size, random_state=42
            )
        
            if verbose:
                print(f"Train: {len(X_train)}, Test: {len(X_test)}, Event rate: {e_train.mean():.3f}")
            
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            event_times = y_train[e_train == 1] #maybe lets actually check the data out to see if this makes sense....
            quantiles = np.linspace(0, 1, num_bins + 1)[1:]
            time_bins = np.quantile(event_times, quantiles)
            time_bins[-1] *= 1.05
            time_bins = np.array([0] + list(time_bins))
        
        if True: #artificially_censor_true
            X_labeled = X_train.copy()
            y_labeled = y_train.copy()
            e_labeled = e_train.copy()
            
            y_labeled, e_labeled, censored_indices = artificially_censor_true( #double check this...
                y_labeled, e_labeled,
                num_initial_samples=initial_data_config[0])
        
        if True: #train base model and initial evaluation
            # time_bins includes leading 0, so pass len-1 since BayesLinMtlr adds +1 internally
            num_time_bins_for_model = len(time_bins) - 1
            base_model = BayesLinMtlr(X_train.shape[1], num_time_bins_for_model, args).to(device)
            
            model_data_train = pd.DataFrame(X_labeled, columns=columns.drop(["time", "event"]))
            model_data_train["time"] = y_labeled
            model_data_train["event"] = e_labeled
            
            path = save_params(args)
            
            base_model_path = "saved_models/base_model_[200, 1400].pth"

            if use_saved_base_model and os.path.exists(base_model_path):
                import traceback
                
                try:
                    state_dict = torch.load(base_model_path, map_location=device)
                    missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
                    base_loss = 0
                    print("Loaded!", base_model_path)
                    print("Missing keys:", missing[:20], "..." if len(missing) > 20 else "")
                    print("Unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")
                except Exception as e:
                    print("FAILED loading:", base_model_path)
                    print(type(e).__name__, ":", e)
                    traceback.print_exc()
                    print(f"Error loading saved model: {e}")
                    print("Training new base model instead")
                    base_model, base_loss = train_model(
                        base_model, model_data_train, time_bins, args,
                        path + "/exp.pth", device, num_epochs=num_epochs,
                        patience=early_stopping_patience, verbose=verbose
                    )
                    torch.save(base_model.state_dict(), base_model_path)
            else:
                print("Training new base model...")
                base_model, base_loss = train_model(
                    base_model, model_data_train, time_bins, args,
                    path + "/exp.pth", device, num_epochs=num_epochs,
                    patience=early_stopping_patience, verbose=verbose
                )
                torch.save(base_model.state_dict(), base_model_path)
                print(f"Base model loss: {base_loss}")
            
            # Initial evaluation
            survival_outputs, time_bins_returned, _ = make_prediction(base_model, X_test, time_bins, args)
            
            # Verify time_bins consistency
            time_bins_np = time_bins if isinstance(time_bins, np.ndarray) else np.array(time_bins)
            time_bins_returned_np = time_bins_returned.cpu().numpy() if isinstance(time_bins_returned, torch.Tensor) else np.array(time_bins_returned)
            assert np.allclose(time_bins_np, time_bins_returned_np), f"Time bins mismatch: input {time_bins_np.shape} vs returned {time_bins_returned_np.shape}"
            
            if isinstance(survival_outputs, torch.Tensor):
                survival_outputs = survival_outputs.cpu().numpy()
            
            initial_pred_times = expected_times_from_survival(survival_outputs, time_bins)
            initial_concordance = concordance(-initial_pred_times, y_test, e_test)
            initial_mae = Mae_Po(initial_pred_times, y_test, e_test, y_train, e_train)
            initial_brier = brier_score(-initial_pred_times, y_test)
            
            print(f"\nInitial C-INDEX: {initial_concordance:.4f}, MAE: {initial_mae:.2f}")
        
        acquisition_results = {}
        
        for acquisition_function in acquisition_functions:
            results = {
                'mae': [initial_mae],
                'concordance': [initial_concordance],
                'brier_score': [initial_brier],
                'training_loss': [base_loss]
            }
            
            # Deep copy model
            temp_path = path + f"/temp_{acquisition_function.__name__}.pth"
            torch.save(base_model.state_dict(), temp_path)
            
            model = BayesLinMtlr(X_train.shape[1], num_time_bins_for_model, args).to(device)
            model.load_state_dict(torch.load(temp_path))
            
            X_labeled_copy = X_labeled.copy()
            y_labeled_copy = y_labeled.copy()
            e_labeled_copy = e_labeled.copy()
            censored_indices_copy = censored_indices.copy()
            
            model_data = pd.DataFrame(X_labeled_copy, columns=columns.drop(["time", "event"]))
            model_data["time"] = y_labeled_copy
            model_data["event"] = e_labeled_copy
            
            if verbose:
                print(f"\nRunning {acquisition_function.__name__}...")

            # Get censored pool - filter out points where artificial_time == true_time
            # (nothing to learn from these points as oracle can't reveal more info)
            learnable_censored = [idx for idx in censored_indices_copy
                                  if y_labeled_copy[idx] < y_train[idx]]
            censored_list = list(learnable_censored)

            if len(censored_list) == 0:
                if verbose:
                    print(f"  No learnable censored points remaining, skipping {acquisition_function.__name__}")
                acquisition_results[acquisition_function.__name__] = results
                continue

            X_censored = X_labeled_copy[censored_list]
            
            model_data_censored = pd.DataFrame(X_censored, columns=columns.drop(["time", "event"]))
            model_data_censored["time"] = y_labeled_copy[censored_list]
            model_data_censored["event"] = e_labeled_copy[censored_list]
            
            censored_costlist = None
            if this_costlist is not None:
                if hasattr(this_costlist, 'iloc'):
                    censored_costlist = this_costlist.iloc[censored_list].reset_index(drop=True)
                else:
                    censored_costlist = pd.DataFrame(
                        np.array(this_costlist).flatten()[censored_list], columns=['Cost']
                    )
            
            try:
                if acquisition_function.__name__ == 'random_knapsack':
                    if censored_costlist is None:
                        censored_costlist = pd.DataFrame([1] * len(X_censored), columns=['Cost'])
                    pool_indices = acquisition_function(censored_costlist, budget)
                else:
                    pool_indices,_ = acquisition_function(
                        model=model, X_pool=X_censored, batch_size=budget,
                        time_bins=time_bins, config=args, device=device,
                        in_data_train=model_data_censored, increment=increment,
                        costlist=censored_costlist, budget=budget
                    )
                
                acquired_indices = [censored_list[i] for i in pool_indices]  # Map pool indices back to original indices
                print(f"  {acquisition_function.__name__}: selected {len(acquired_indices)} samples")
                
            except Exception as ex:
                print(f"  Error with {acquisition_function.__name__}: {ex}")
                acquisition_results[acquisition_function.__name__] = results
                continue
            
            if not acquired_indices:
                acquisition_results[acquisition_function.__name__] = results
                continue
            
            # Update labels
            for idx in acquired_indices:
                y_labeled_copy[idx] = min(y_train[idx], y_labeled_copy[idx] + increment)
                if y_labeled_copy[idx] >= y_train[idx]:
                    e_labeled_copy[idx] = e_train[idx]
            
            model_data["time"] = y_labeled_copy
            model_data["event"] = e_labeled_copy
            
            # Retrain
            model, total_loss = train_model(
                model, model_data, time_bins, args,
                path + f"/{acquisition_function.__name__}.pth", device,
                num_epochs=num_epochs, patience=early_stopping_patience, verbose=verbose, reset_model=False
            )
            
            # Evaluate
            survival_outputs, time_bins_returned, _ = make_prediction(model, X_test, time_bins, args)

            # Verify time_bins consistency
            time_bins_np = time_bins if isinstance(time_bins, np.ndarray) else np.array(time_bins)
            time_bins_returned_np = time_bins_returned.cpu().numpy() if isinstance(time_bins_returned, torch.Tensor) else np.array(time_bins_returned)
            assert np.allclose(time_bins_np, time_bins_returned_np), f"Time bins mismatch: input {time_bins_np.shape} vs returned {time_bins_returned_np.shape}"

            if isinstance(survival_outputs, torch.Tensor):
                survival_outputs = survival_outputs.cpu().numpy()
            
            pred_times = expected_times_from_survival(survival_outputs, time_bins)
            new_concordance = concordance(-pred_times, y_test, e_test)
            new_mae = Mae_Po(pred_times, y_test, e_test, y_train, e_train)
            new_brier = brier_score(-pred_times, y_test)
            
            results['mae'].append(new_mae)
            results['concordance'].append(new_concordance)
            results['brier_score'].append(new_brier)
            results['training_loss'].append(total_loss)
            
            c_improvement = new_concordance - initial_concordance
            print(f"  C-index: {new_concordance:.4f} (Δ={c_improvement:+.4f})")
            
            acquisition_results[acquisition_function.__name__] = results
        
        increment_list.append(acquisition_results)
    
    return increment_list


if __name__ == "__main__":
    start_time = time.time()
    seed_everything(42)
    
    acquisition_functions = [
        # batchbald_acquire_budget,
        # batchbald_density,  # BatchBALD + Density
        batchbald_acquire_budget_DEBUG,
        test_bald_score,
        test_batchbald_score,
        # batchbald_acquire_survival_optimized,
        highest_death_probability_in_window,
        entropy_of_probs,
        variance_of_probs,
        clostest_to_half_in_increment,
        mean_closest_to_middle,
        Using_Clusters_for_batch,
        cbald_censored_regression,
        random_knapsack
    ]
    
    print("\n" + "="*70)
    print("SURVIVAL ANALYSIS WITH ACTIVE LEARNING")
    print("="*70)
    
    results = General_Function(
        dataset="NACD",
        initial_data_config=[200, 1400],
        increments=[6],
        budget=10,
        acquisition_functions=acquisition_functions,
        num_epochs=1000,
        verbose=True,
        num_bins=10,
        use_saved_base_model=True,
        early_stopping_patience=15
    )
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS - RANKED BY C-INDEX")
    print("="*70)
    print(f"{'Rank':<6} {'Method':<35} {'C-Index':>12} {'MAE':>10}")
    print("-"*65)
    
    scores = []
    for increment_results in results:
        for func_name, metrics in increment_results.items():
            c_idx = metrics['concordance'][-1] if 'concordance' in metrics else 0
            mae = metrics['mae'][-1] if 'mae' in metrics else float('inf')
            scores.append((func_name, c_idx, mae))
    
    scores.sort(key=lambda x: -x[1])
    for rank, (name, c_idx, mae) in enumerate(scores, 1):
        marker = " **" if "batchbald" in name.lower() else ""
        print(f"#{rank:<5} {name:<35} {c_idx:>12.4f} {mae:>10.2f}{marker}")
    
    end_time = time.time()
    print(f"\nTotal time: {(end_time - start_time)/60:.2f} minutes")
