# CLAUDE.md - AI Assistant Guide for BatchBALD Repository

## Project Overview

This repository implements and evaluates **BatchBALD (Batch Active Learning with Deep Bayesian Learning)** for survival analysis with a novel **probe depth** oracle constraint. The project compares BatchBALD against simpler baselines (entropy and variance) in a single-shot active learning setting where the oracle can only reveal information up to k bins beyond the current censoring time.

### Key Research Finding
BatchBALD shows marginal advantage (+0.0105 Δ C-index) over baselines but is **NOT statistically significant** (p=0.094). Ensemble diversity is critical for BatchBALD to work effectively.

---

## Repository Structure

```
BatchBALD-test-repo/
├── src/                          # Main source code (modular architecture)
│   ├── data/
│   │   └── synthetic.py          # Data generation: generate_survival_data(), artificially_censor()
│   ├── models/
│   │   └── survival_model.py     # BayesianSurvivalModel, SurvivalNet (neural network)
│   ├── oracle/
│   │   └── oracle.py             # Oracle class with probe depth constraint
│   ├── acquisition/
│   │   ├── batchbald.py          # SurvivalBatchBALD (core implementation)
│   │   ├── entropy.py            # EntropyAcquisition baseline
│   │   ├── variance.py           # VarianceAcquisition baseline
│   │   ├── improved_batchbald.py # Enhanced variants (observable_mass, MI×P(death))
│   │   └── weighted_batchbald.py # Weighted acquisition function
│   └── evaluation/
│       └── metrics.py            # concordance_index(), brier_score()
│
├── tests/                        # Unit and integration tests
│   ├── test_data.py              # Tests for data generation
│   ├── test_oracle.py            # Tests for oracle logic
│   ├── test_acquisition.py       # Tests for acquisition functions
│   └── run_all_tests.py          # Test runner
│
├── experiments/                  # Experimental scripts
│   ├── run_comparison.py         # Main comparison script
│   ├── final_comparison.py       # Final evaluation (10 runs)
│   ├── fast_diagnostic.py        # Quick 3-run diagnostic
│   ├── check_ensemble_diversity.py
│   ├── test_nacd_continuous.py   # Real NACD dataset tests
│   └── [additional experimental variants]
│
├── Model_stuff/                  # Legacy Bayesian survival models
│   ├── model.py                  # Distribution definitions
│   ├── data.py                   # Dataset loading utilities
│   ├── main.py                   # Legacy training pipeline
│   └── acquisition.py            # Legacy acquisition implementations
│
├── data/                         # Data storage
│   └── MIMIC/                    # Real MIMIC dataset
│
├── model.py                      # Root-level consolidated model module
├── Making_and_getting_datasets.py  # Dataset creation utilities
├── requirements.txt              # Dependencies
├── README.md                     # Quick start guide
└── RESULTS.md                    # Detailed experimental results
```

---

## Key Concepts

### Data Representations

**Predictions Shape:** `(K, N, T)`
- **K**: Number of ensemble members (typically 5-10)
- **N**: Number of samples
- **T**: Number of time bins (typically 8-10)
- **Values**: Probabilities (summing to 1 per sample)

**Oracle Probabilities Shape:** `(K, N, k+1)`
- **K**: Ensemble members
- **N**: Number of samples
- **k+1**: Oracle outcomes (k death bins + 1 censored outcome)

### Probe Depth Constraint

The oracle can only reveal information up to `probe_depth` (k) bins beyond the current censoring time:
- Reveals: death within k bins OR censored at k
- This constraint simulates real-world scenarios where follow-up has time limitations

### Ensemble Diversity

**Critical for BatchBALD success:**
- Dropout rate: 0.2
- Bootstrap sampling per ensemble member
- Xavier initialization with gain=2.0
- Light L2 regularization (weight_decay=0.001)

---

## Development Workflow

### Installation

```bash
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all unit tests
python tests/run_all_tests.py
```

### Running Experiments

```bash
# Quick diagnostic (3 runs)
python experiments/fast_diagnostic.py

# Full comparison (10 runs)
python experiments/final_comparison.py

# Check ensemble diversity
python experiments/check_ensemble_diversity.py
```

---

## Code Conventions

### Python Style

- Type hints are used for function signatures (via `typing` module)
- NumPy arrays preferred for numerical data
- PyTorch tensors for model training and inference
- Docstrings follow Google style with Args/Returns sections

### Class Structure Example

```python
class SurvivalBatchBALD:
    """
    BatchBALD acquisition function for survival analysis with probe depth.

    Computes I(Y_oracle; theta | X, D) where Y_oracle is what the oracle reveals.
    """

    def __init__(self, probe_depth: int):
        """
        Args:
            probe_depth: Oracle probe depth (k bins)
        """
        self.probe_depth = probe_depth

    def compute_mutual_information(self, oracle_probs: np.ndarray) -> np.ndarray:
        """
        Compute I(Y_oracle; theta) = H(E[Y_oracle]) - E[H(Y_oracle | theta)].

        Args:
            oracle_probs: (K, N, k+1) probabilities

        Returns:
            mutual_info: (N,) mutual information for each instance
        """
```

### Testing Pattern

Tests are standalone functions prefixed with `test_`:
- No pytest markers or fixtures - direct function calls
- Use assertions for validation
- Use real data generation (no mocking)
- Verbose output with print statements for debugging

```python
def test_batchbald():
    """Test BatchBALD acquisition function."""
    acq = SurvivalBatchBALD(probe_depth=3)
    oracle_probs = np.random.rand(5, 10, 4)  # (K=5, N=10, k+1=4)
    oracle_probs = oracle_probs / oracle_probs.sum(axis=2, keepdims=True)

    # Test shape
    mi = acq.compute_mutual_information(oracle_probs)
    assert mi.shape == (10,)
    assert np.all(mi >= 0)  # MI is non-negative
    print("  ✓ BatchBALD mutual information computed correctly")
```

---

## Default Hyperparameters

```python
# Data
n_samples = 300              # Total samples
n_features = 5               # Number of covariates
n_time_bins = 8              # Discrete time bins
censoring_rate = 0.3         # Natural censoring
artificial_censor_rate = 0.5 # Proportion artificially censored

# Active Learning
probe_depth = 2              # Oracle reveals up to k=2 bins
batch_size = 15              # Query 15 samples per round
n_ensemble = 5               # Ensemble size

# Training
epochs = 50                  # Training epochs
batch_size_train = 32        # Training batch size
learning_rate = 0.001        # Adam learning rate
weight_decay = 0.001         # L2 regularization

# Ensemble Diversity
dropout_rate = 0.2           # MC dropout for uncertainty
bootstrap = True             # Bootstrap sampling per member
initialization_gain = 2.0    # Xavier gain for weight variance
```

---

## Key Classes and Their Roles

| Class | Location | Purpose |
|-------|----------|---------|
| `BayesianSurvivalModel` | `src/models/survival_model.py` | Ensemble of neural networks for survival prediction |
| `SurvivalNet` | `src/models/survival_model.py` | Single neural network module |
| `Oracle` | `src/oracle/oracle.py` | Simulates oracle with probe depth constraint |
| `SurvivalBatchBALD` | `src/acquisition/batchbald.py` | BatchBALD acquisition function |
| `EntropyAcquisition` | `src/acquisition/entropy.py` | Predictive entropy baseline |
| `VarianceAcquisition` | `src/acquisition/variance.py` | Ensemble variance baseline |

---

## Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `generate_survival_data()` | `src/data/synthetic.py` | Generate synthetic survival data |
| `artificially_censor()` | `src/data/synthetic.py` | Apply artificial censoring |
| `split_data()` | `src/data/synthetic.py` | Train/test split |
| `concordance_index()` | `src/evaluation/metrics.py` | C-index evaluation metric |
| `brier_score()` | `src/evaluation/metrics.py` | Brier score evaluation |
| `get_oracle_outcome_probs()` | `src/oracle/oracle.py` | Compute oracle outcome probabilities |

---

## Important Implementation Details

### Survival Loss Function

The model uses discrete survival loss (negative log-likelihood):
- Death event (e=1): `log P(T=t) = log h_t + log S_{t-1}`
- Censored (e=0): `log P(T>t) = log S_t`

Where `h_t` is hazard at time t, `S_t` is survival function.

### BatchBALD Computation

1. **Conditional Entropy**: `H(Y_oracle | theta_k) = -sum_c p(c|theta_k) log p(c|theta_k)`
2. **Entropy of Expected**: `H(E[Y_oracle]) = H(mean over theta)`
3. **Mutual Information**: `I(Y_oracle; theta) = H(E[Y_oracle]) - E[H(Y_oracle | theta)]`
4. **Batch Selection**: Greedy selection maximizing joint mutual information

### Oracle Query Logic

```python
def query(indices, current_time, current_event):
    """
    Query oracle for samples at indices.

    Returns:
        updated_time: New event/censoring time
        updated_event: 1 if death revealed, 0 if still censored
    """
    # Reveals death if it occurs within probe_depth bins
    # Otherwise, extends censoring time by probe_depth
```

---

## Common Pitfalls to Avoid

1. **Ensemble Diversity**: Without proper diversity techniques, BatchBALD fails completely. Always use dropout, bootstrap sampling, and proper initialization.

2. **Shape Mismatches**: Always verify tensor shapes match expected `(K, N, T)` format for predictions and `(K, N, k+1)` for oracle probabilities.

3. **Uncensored Sample Handling**: Already uncensored samples (event=1) should be excluded from acquisition function computation.

4. **Numerical Stability**: Use `scipy.special.xlogy` for entropy computations to handle `0 * log(0) = 0`.

5. **Discrete Time Bins**: Models expect discrete bin indices (integers), not continuous times.

---

## Extending the Codebase

### Adding a New Acquisition Function

1. Create a new file in `src/acquisition/`
2. Implement a class with:
   - `__init__(self, probe_depth: int)`
   - `compute_scores(self, oracle_probs: np.ndarray, current_event: Optional[np.ndarray]) -> np.ndarray`
   - `select_batch(self, oracle_probs: np.ndarray, batch_size: int, current_event: Optional[np.ndarray]) -> np.ndarray`
3. Add tests in `tests/test_acquisition.py`
4. Import in experiment scripts

### Adding a New Dataset

1. Add data loading function in `src/data/` or `Model_stuff/data.py`
2. Ensure output format matches: `X (features), time (discrete bins), event (0/1)`
3. Create an experiment script in `experiments/`

---

## Dependencies

```
numpy>=1.21.0          # Numerical computing
scipy>=1.7.0           # Scientific computing
torch>=1.9.0           # Neural networks
scikit-learn>=0.24.0   # ML utilities
matplotlib>=3.4.0      # Visualization
lifelines>=0.27.0      # Survival analysis
pytest>=6.2.0          # Testing
```

---

## Useful Commands

```bash
# Run specific test file
python tests/test_acquisition.py

# Run main comparison
python experiments/run_comparison.py

# Test on real data
python experiments/test_nacd_continuous.py

# Large-scale experiment
python experiments/large_scale_comparison.py
```

---

## Research Context

This project addresses:
1. **Active learning for survival analysis** - Selecting which censored samples to query
2. **BatchBALD adaptation** - Modifying mutual information computation for oracle constraints
3. **Probe depth constraints** - Realistic modeling of limited follow-up capability
4. **Ensemble uncertainty** - Leveraging Bayesian neural networks for uncertainty estimation

**Key insight**: Simpler entropy baselines are surprisingly competitive with BatchBALD in this setting, especially for small datasets.

---

## Last Updated

- **Date**: 2026-01-25
- **Status**: Experimental validation complete
- **Main Finding**: BatchBALD shows promise but needs larger scale experiments for statistical significance
