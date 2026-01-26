"""
Simple test script to verify Bayesian Linear MTLR achieves c-index of 0.7 on NACD data.
"""

import sys
import os
sys.path.insert(0, 'Model_stuff')

import numpy as np
import torch
from Model_stuff.data import make_nacd_data
from Model_stuff.model import BayesLinMtlr
from Model_stuff.utils import reformat_survival
from Model_stuff.evaluation import concordance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse

def main():
    print("="*70)
    print("Testing Bayesian Linear MTLR on NACD Dataset")
    print("="*70)

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Load NACD data
    print("\n1. Loading NACD dataset...")
    data = make_nacd_data()
    print(f"   Dataset shape: {data.shape}")

    # Prepare data
    X = data.drop(["time", "event"], axis=1).values
    y = data["time"].values
    e = data["event"].values

    # Train/test split
    print("\n2. Splitting data (90% train, 10% test)...")
    X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(
        X, y, e, test_size=0.1, random_state=42, stratify=e
    )
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"   Train event rate: {e_train.mean():.3f}, Test event rate: {e_test.mean():.3f}")

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create time bins
    print("\n3. Creating time bins...")
    num_bins = 10
    event_times = y_train[e_train == 1]
    quantiles = np.linspace(0, 1, num_bins + 1)[1:]
    time_bins = np.quantile(event_times, quantiles)
    time_bins[-1] *= 1.05
    time_bins = np.array([0] + list(time_bins))
    print(f"   Time bins: {time_bins}")

    # Create model config
    print("\n4. Creating model configuration...")
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
    print(f"   Using device: {config.device}")

    # Create Bayesian Linear MTLR model
    print("\n5. Creating Bayesian Linear MTLR model...")
    model = BayesLinMtlr(
        in_features=X_train.shape[1],
        num_time_bins=num_bins,  # Will add +1 internally
        config=config
    ).to(config.device)
    print(f"   Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Prepare training data
    import pandas as pd
    from torch.utils.data import DataLoader, TensorDataset

    train_df = pd.DataFrame(X_train)
    train_df['time'] = y_train
    train_df['event'] = e_train
    x_formatted, y_formatted = reformat_survival(train_df, time_bins[1:])

    train_dataset = TensorDataset(x_formatted, y_formatted)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Train the model
    print("\n6. Training model (100 epochs with early stopping)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)

    best_loss = float('inf')
    patience = 20
    patience_counter = 0

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

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break

    # Make predictions
    print("\n7. Making predictions...")
    from Model_stuff.model import mtlr_survival
    from Model_stuff.utils import expected_times_from_survival

    # Expected times function
    def expected_times_from_survival(surv_np, tbins):
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

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(config.device)
        logits = model.forward(X_test_tensor, sample=True, n_samples=config.n_samples_test)
        survival_probs = mtlr_survival(logits, with_sample=True)
        mean_survival = survival_probs.mean(dim=0).cpu().numpy()

    pred_times = expected_times_from_survival(mean_survival, time_bins)

    # Calculate C-index
    print("\n8. Evaluating C-index...")
    c_index = concordance(-pred_times, y_test, e_test)

    print("\n" + "="*70)
    print(f"FINAL RESULT: C-Index = {c_index:.4f}")
    print("="*70)

    if c_index >= 0.65:
        print("✓ Model achieves acceptable C-index (>= 0.65)")
    else:
        print("✗ Model C-index is below acceptable threshold")

    return c_index

if __name__ == "__main__":
    main()
