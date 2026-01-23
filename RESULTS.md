# Experimental Results: BatchBALD for Survival Analysis

## Executive Summary

We implemented and tested a survival-specific BatchBALD acquisition function against entropy and variance baselines in a novel probe-depth-constrained active learning setting.

**Key Finding:** BatchBALD shows marginal advantage (mean Δ C-index: +0.0105) over baselines, but **the difference is NOT statistically significant** (p=0.094).

## Experimental Setup

- **Task**: Survival analysis with discrete time bins
- **Model**: Bayesian ensemble (5 members) with dropout
- **Oracle**: Reveals information up to probe depth k=2
- **Evaluation**: Single-shot active learning (query → retrain → evaluate)
- **Metrics**: Concordance index (C-index), Brier score
- **Runs**: 10 independent experiments with different random seeds

### Parameters
- Samples: 300 (210 train, 90 test)
- Features: 5
- Time bins: 8
- Probe depth: 2
- Batch size: 15
- Artificial censoring rate: 50%
- Ensemble size: 5

## Final Results (10 runs)

| Method | Mean Δ C-index | Std Dev | Win Rate | Best Run | Worst Run |
|--------|----------------|---------|----------|----------|-----------|
| **BatchBALD** | **+0.0105** | 0.0297 | **60%** (6/10) | +0.0615 | -0.0365 |
| Entropy | -0.0154 | 0.0384 | 40% (4/10) | +0.0588 | -0.0657 |
| Variance | -0.0237 | 0.0292 | 10% (1/10) | +0.0027 | -0.1000 |

### Statistical Significance
- **Paired t-test** (BatchBALD vs Entropy): t=1.875, **p=0.094**
- **Conclusion**: NOT statistically significant at α=0.05 level

## Key Insights

### 1. BatchBALD Performance
- ✓ **Highest mean improvement** (+0.0105 vs -0.0154 for Entropy)
- ✓ **Best win rate** (60% vs 40% for Entropy)
- ⚠️ **High variance** (inconsistent across runs)
- ⚠️ **Not statistically significant** (p=0.094)

### 2. Critical Success Factor: Ensemble Diversity

We discovered that **ensemble diversity is crucial** for BatchBALD to work:

| Metric | Without Fixes | With Fixes | Improvement |
|--------|---------------|------------|-------------|
| Average disagreement | 0.026 | 0.161 | **6.2x** |
| Unique predictions/sample | 1.25/5 | 3.60/5 | **2.9x** |
| Mutual information | 0.015 | 0.583 | **39x** |

**Diversity techniques applied:**
- Dropout rate: 0.2
- Bootstrap sampling per ensemble member
- Xavier initialization with gain=2.0
- Light L2 regularization (weight_decay=0.001)

### 3. Why BatchBALD Struggled

Despite theoretical advantages, BatchBALD showed only marginal improvements because:

1. **Limited information from probe depth**: With k=2, oracle can only reveal 2 bins beyond censoring time
2. **Entropy baseline is surprisingly effective**: Simple predictive entropy captures uncertainty well
3. **Small batch size** (15/210 ≈ 7%): BatchBALD's batch diversity advantage is limited
4. **High stochasticity**: Small datasets and neural network training introduce noise

### 4. Entropy Baseline Advantages

Entropy performed surprisingly well:
- **Simpler** to implement and compute
- **More stable** in some runs
- **Computationally cheaper** (no joint entropy calculation)
- **Reveals more deaths** in some experiments (8/10 vs 5/10)

## Theoretical vs Practical Performance

### Theory Predicts BatchBALD Should Win Because:
1. ✓ Accounts for batch diversity (avoids redundant queries)
2. ✓ Uses full ensemble uncertainty (epistemic)
3. ✓ Oracle-aligned (optimizes for what oracle reveals)
4. ✓ Survival-specific (handles probe depth constraint)

### In Practice, Limited Advantage Because:
1. Small datasets reduce statistical power
2. Neural network training noise dominates
3. Probe depth constraint limits information gain
4. Entropy baseline is surprisingly strong

## Conclusions

1. **BatchBALD works in principle** but requires careful tuning
2. **Ensemble diversity is critical** - without it, BatchBALD fails completely
3. **Marginal advantage over baselines** in this setting (not statistically significant)
4. **Entropy is a strong baseline** - simpler and nearly as good

## Recommendations

### Use BatchBALD if:
- Large datasets (>1000 samples)
- Larger batch sizes (>50 queries)
- Computational budget allows
- Ensemble diversity can be ensured

### Use Entropy if:
- Small datasets
- Need simplicity and reliability
- Computational constraints
- Baseline performance is sufficient

## Future Work

To make BatchBALD more competitive:

1. **Increase dataset size**: Test on 1000+ samples
2. **Larger batches**: Try batch sizes of 50-100
3. **Deeper probe depths**: Test k=5, 10
4. **Better diversity**: Explore Bayesian neural networks, variational inference
5. **Multiple rounds**: Test iterative active learning (not just single-shot)
6. **Real survival datasets**: Test on SEER, SUPPORT, etc.

## Reproducibility

All code and experiments are in this repository:

```bash
# Run unit tests
python tests/run_all_tests.py

# Run fast diagnostic (3 runs)
python experiments/fast_diagnostic.py

# Run full comparison (10 runs)
python experiments/final_comparison.py

# Check ensemble diversity
python experiments/check_ensemble_diversity.py
```

## Visualization

![Results Summary](experiments/results_summary.png)

---

**Date**: 2026-01-23
**Status**: Experimental validation complete
**Conclusion**: BatchBALD shows promise but needs larger scale experiments for statistical significance.
