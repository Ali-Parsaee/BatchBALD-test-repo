# BatchBALD for Survival Analysis with Probe Depth

This repository implements BatchBALD for survival analysis with oracle probe depth constraints.

## Problem Setting

- Survival data with discrete time bins
- Bayesian models output (K, N, T) predictions
- Oracle reveals information up to probe depth k
- Single-shot active learning evaluation

## Structure

- `src/data/`: Synthetic survival data generation
- `src/models/`: Bayesian survival models
- `src/oracle/`: Oracle simulation with probe depth
- `src/acquisition/`: Acquisition functions (BatchBALD, entropy, variance)
- `src/evaluation/`: Survival metrics
- `experiments/`: Experimental comparison scripts
- `tests/`: Unit and integration tests

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python experiments/run_comparison.py
```
