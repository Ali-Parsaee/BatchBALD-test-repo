import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional


class SurvivalNet(nn.Module):
    """Single neural network for survival prediction."""

    def __init__(self, n_features: int, n_time_bins: int, hidden_size: int = 64, dropout_rate: float = 0.3):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.network = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, n_time_bins)
        )

        # Initialize weights with more variance for diversity
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with higher variance for ensemble diversity."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use larger std for initialization
                nn.init.xavier_normal_(module.weight, gain=2.0)
                if module.bias is not None:
                    nn.init.normal_(module.bias, std=0.1)

    def forward(self, x):
        return self.network(x)


class BayesianSurvivalModel:
    """
    Bayesian survival model using ensemble.

    Outputs predictions of shape (K, N, T) where:
    - K: number of ensemble members
    - N: number of instances
    - T: number of time bins
    """

    def __init__(
        self,
        n_features: int,
        n_time_bins: int,
        n_ensemble: int = 10,
        hidden_size: int = 64,
        device: str = 'cpu'
    ):
        self.n_features = n_features
        self.n_time_bins = n_time_bins
        self.n_ensemble = n_ensemble
        self.hidden_size = hidden_size
        self.device = device

        # Create ensemble of models with balanced dropout for diversity
        self.models = [
            SurvivalNet(n_features, n_time_bins, hidden_size, dropout_rate=0.2).to(device)
            for _ in range(n_ensemble)
        ]

    def fit(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
        verbose: bool = False
    ):
        """
        Train the ensemble on survival data.

        Uses discrete survival loss (negative log-likelihood for discrete hazards).
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        time_tensor = torch.LongTensor(time).to(self.device)
        event_tensor = torch.FloatTensor(event).to(self.device)

        n_samples = len(X)

        for k, model in enumerate(self.models):
            # Use light L2 regularization for diversity
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

            # Bootstrap sample for this ensemble member (with replacement)
            # This increases diversity by training on different data subsets
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X_tensor[bootstrap_indices]
            time_bootstrap = time_tensor[bootstrap_indices]
            event_bootstrap = event_tensor[bootstrap_indices]

            for epoch in range(epochs):
                model.train()
                total_loss = 0.0

                # Mini-batch training on bootstrapped data
                indices = np.random.permutation(n_samples)
                for i in range(0, n_samples, batch_size):
                    batch_idx = indices[i:i + batch_size]

                    X_batch = X_bootstrap[batch_idx]
                    time_batch = time_bootstrap[batch_idx]
                    event_batch = event_bootstrap[batch_idx]

                    optimizer.zero_grad()
                    logits = model(X_batch)  # (batch_size, n_time_bins)

                    # Survival loss
                    loss = self._survival_loss(logits, time_batch, event_batch)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Model {k+1}/{self.n_ensemble}, Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    def _survival_loss(
        self,
        logits: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
    ) -> torch.Tensor:
        """
        Discrete survival loss (negative log-likelihood).

        For each sample:
        - If event=1 (death at time t): log P(T=t) = log h_t + log S_{t-1}
        - If event=0 (censored at time t): log P(T>t) = log S_t

        where h_t is hazard at time t, S_t is survival function at time t.
        """
        # Get probabilities for each time bin
        probs = torch.softmax(logits, dim=1)  # (batch_size, n_time_bins)

        batch_size = logits.size(0)
        loss = 0.0

        for i in range(batch_size):
            t = time[i].item()
            e = event[i].item()

            # Survival function: S(t) = prod(1 - h_j) for j <= t
            # Approximate with cumulative sum for numerical stability
            cum_hazard = torch.cumsum(probs[i], dim=0)
            survival = 1.0 - cum_hazard

            if e == 1:  # Death event
                # P(T=t) = h_t * S_{t-1}
                hazard_t = probs[i, t]
                if t > 0:
                    survival_t_minus_1 = survival[t - 1]
                else:
                    survival_t_minus_1 = torch.tensor(1.0).to(self.device)

                prob = hazard_t * survival_t_minus_1
                prob = torch.clamp(prob, min=1e-8, max=1.0)
                loss -= torch.log(prob)
            else:  # Censored
                # P(T > t)
                if t < self.n_time_bins:
                    survival_t = survival[t]
                else:
                    survival_t = survival[-1]

                survival_t = torch.clamp(survival_t, min=1e-8, max=1.0)
                loss -= torch.log(survival_t)

        return loss / batch_size

    def predict_proba(
        self,
        X: np.ndarray,
        return_logits: bool = False,
        mc_dropout: bool = False
    ) -> np.ndarray:
        """
        Predict survival probabilities.

        Args:
            X: Input features
            return_logits: Whether to return logits instead of probabilities
            mc_dropout: If True, keep dropout enabled during inference (MC Dropout)

        Returns:
            predictions: (K, N, T) array of probabilities (or logits if return_logits=True)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        n_samples = len(X)

        predictions = np.zeros((self.n_ensemble, n_samples, self.n_time_bins))

        for k, model in enumerate(self.models):
            if mc_dropout:
                # Keep dropout enabled for Monte Carlo Dropout
                model.train()
            else:
                model.eval()

            with torch.no_grad():
                logits = model(X_tensor)  # (N, T)

                if return_logits:
                    predictions[k] = logits.cpu().numpy()
                else:
                    probs = torch.softmax(logits, dim=1)
                    predictions[k] = probs.cpu().numpy()

        return predictions

    def predict_survival_function(self, X: np.ndarray) -> np.ndarray:
        """
        Predict survival function S(t) = P(T > t).

        Returns:
            survival: (K, N, T) array of survival probabilities
        """
        probs = self.predict_proba(X)  # (K, N, T)

        # S(t) = product of (1 - h_j) for j <= t
        # where h_j is the hazard at time j
        survival = np.zeros_like(probs)

        for k in range(self.n_ensemble):
            for i in range(len(X)):
                for t in range(self.n_time_bins):
                    # Cumulative product
                    cum_hazard = np.sum(probs[k, i, :t+1])
                    survival[k, i, t] = 1.0 - cum_hazard

        return np.clip(survival, 0.0, 1.0)
