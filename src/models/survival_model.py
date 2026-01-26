"""
Bayesian Survival Model using Bayesian Linear MTLR.
This follows the correct implementation from Model_stuff/model.py.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional

# Add Model_stuff to path to import the correct Bayesian MTLR implementation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Model_stuff'))

try:
    from model import BayesLinMtlr, mtlr_survival
    from utils import reformat_survival, encode_survival
except ImportError:
    # Fallback: try importing from root
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from Model_stuff.model import BayesLinMtlr, mtlr_survival
    from Model_stuff.utils import reformat_survival, encode_survival


class BayesianSurvivalModel:
    """
    Bayesian survival model using Bayesian Linear MTLR.

    This implementation uses the correct Bayesian Linear MTLR approach with:
    - BayesianLinear layers with scale mixture priors
    - Proper ELBO loss with KL divergence
    - Variational inference for uncertainty quantification

    Outputs predictions of shape (K, N, T) where:
    - K: number of samples from the posterior
    - N: number of instances
    - T: number of time bins
    """

    def __init__(
        self,
        n_features: int,
        n_time_bins: int,
        n_ensemble: int = 10,  # This becomes n_samples in Bayesian setting
        hidden_size: int = 50,
        device: str = 'cpu',
        config: Optional[object] = None
    ):
        self.n_features = n_features
        self.n_time_bins = n_time_bins
        self.n_samples = n_ensemble  # Number of posterior samples
        self.hidden_size = hidden_size
        self.device = device

        # Create config if not provided
        if config is None:
            import argparse
            config = argparse.Namespace()
            config.pi = 0.5
            config.sigma1 = 1.0
            config.sigma2 = 0.0025
            config.rho_scale = -3.0
            config.mu_scale = 0.1
            config.batch_size = 32
            config.c1 = 0.01
            config.n_samples_train = 10
            config.n_samples_test = n_ensemble
            config.device = device
            config.hidden_size = hidden_size

        self.config = config

        # Create the Bayesian Linear MTLR model
        # Note: BayesLinMtlr adds +1 internally, so we pass n_time_bins - 1
        self.model = BayesLinMtlr(
            in_features=n_features,
            num_time_bins=n_time_bins - 1,
            config=config
        ).to(device)

    def fit(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        time_bins: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 8e-5,
        patience: int = 20,
        verbose: bool = False
    ):
        """
        Train the Bayesian Linear MTLR model.

        Uses proper Bayesian training with ELBO loss (Evidence Lower Bound).
        """
        import pandas as pd
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.model_selection import train_test_split

        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(time).to(self.device)
        e_tensor = torch.FloatTensor(event).to(self.device)

        # Create time bins if not provided
        if time_bins is None:
            event_times = time[event == 1]
            if len(event_times) > 0:
                quantiles = np.linspace(0, 1, self.n_time_bins + 1)[1:]
                time_bins = np.quantile(event_times, quantiles)
                time_bins[-1] *= 1.05
                time_bins = np.array([0] + list(time_bins))
            else:
                time_bins = np.linspace(0, time.max(), self.n_time_bins + 1)

        # Convert to proper format
        train_indices, val_indices = train_test_split(
            np.arange(len(X)), test_size=0.1, random_state=42, stratify=event
        )

        x_train, y_train, e_train = X_tensor[train_indices], y_tensor[train_indices], e_tensor[train_indices]
        x_val, y_val, e_val = X_tensor[val_indices], y_tensor[val_indices], e_tensor[val_indices]

        # Format for MTLR training
        train_df = pd.DataFrame(x_train.cpu().numpy())
        train_df['time'] = y_train.cpu().numpy()
        train_df['event'] = e_train.cpu().numpy()
        x_formatted, y_formatted = reformat_survival(train_df, time_bins[1:])

        val_df = pd.DataFrame(x_val.cpu().numpy())
        val_df['time'] = y_val.cpu().numpy()
        val_df['event'] = e_val.cpu().numpy()
        x_val_formatted, y_val_formatted = reformat_survival(val_df, time_bins[1:])

        # Create data loader
        train_dataset = TensorDataset(x_formatted, y_formatted)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Training loop with early stopping
        best_val_loss = float('inf')
        best_epoch = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for xi, yi in train_loader:
                xi, yi = xi.to(self.device), yi.to(self.device)
                optimizer.zero_grad()

                # Use ELBO loss
                loss, _, _, _ = self.model.sample_elbo(
                    xi, yi, len(train_indices), see1=self.config.c1
                )

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss, _, _, _ = self.model.sample_elbo(
                    x_val_formatted, y_val_formatted,
                    dataset_size=len(val_indices), see1=self.config.c1
                )
                val_loss = val_loss.item() / len(val_indices)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = self.model.state_dict().copy()
            elif epoch - best_epoch > patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Store time bins for later use
        self.time_bins = time_bins

    def predict_proba(
        self,
        X: np.ndarray,
        return_logits: bool = False,
        mc_dropout: bool = False
    ) -> np.ndarray:
        """
        Predict survival probabilities using Bayesian inference.

        Args:
            X: Input features
            return_logits: Whether to return logits instead of probabilities
            mc_dropout: Ignored (kept for interface compatibility)

        Returns:
            predictions: (K, N, T) array of survival probabilities
        """
        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            # Get logits from Bayesian model (samples from posterior)
            logits = self.model.forward(
                X_tensor, sample=True, n_samples=self.n_samples
            )  # (K, N, T)

            if return_logits:
                return logits.cpu().numpy()
            else:
                # Convert to survival probabilities
                survival = mtlr_survival(logits, with_sample=True)  # (K, N, T)
                return survival.cpu().numpy()

    def predict_survival_function(self, X: np.ndarray) -> np.ndarray:
        """
        Predict survival function S(t) = P(T > t).

        Returns:
            survival: (K, N, T) array of survival probabilities
        """
        return self.predict_proba(X, return_logits=False)
