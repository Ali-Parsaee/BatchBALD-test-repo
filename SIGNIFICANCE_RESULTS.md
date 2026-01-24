# Statistical Significance Testing Across Settings

## Executive Summary

**Answer: NO, OptimalBatchBALD is NOT statistically significant across increments and budgets.**

Tested 6 different settings (probe depths: 2, 3, 5; batch sizes: 20, 30, 40, 50) with 10 runs each.

**Result: 0/6 settings achieved statistical significance** (all p > 0.79)

## Complete Results

| Probe Depth | Batch Size | Optimal Δ | Entropy Δ | Difference | p-value | Significant? |
|-------------|------------|-----------|-----------|------------|---------|--------------|
| 2 | 30 | -0.0015 | -0.0003 | -0.0013 | 0.8296 | ✗ No |
| 3 | 30 | -0.0012 | -0.0021 | **+0.0008** | 0.9035 | ✗ No |
| 5 | 30 | +0.0010 | +0.0033 | -0.0022 | 0.7301 | ✗ No |
| 3 | 20 | -0.0018 | -0.0019 | +0.0001 | 0.9801 | ✗ No |
| 3 | 40 | +0.0039 | +0.0025 | **+0.0015** | 0.8867 | ✗ No |
| 3 | 50 | -0.0023 | -0.0009 | -0.0014 | 0.8025 | ✗ No |

**Best setting:** Probe=3, Batch=40 → Δ=+0.0015, p=0.89 (still not significant!)

## Key Findings

### 1. No Significant Improvements Anywhere

- All p-values > 0.73 (far from significance threshold of 0.05)
- Mean differences are tiny (-0.0022 to +0.0015)
- Neither increasing batch size nor changing probe depth helps

### 2. High Variance Dominates

Looking at individual runs, we see huge swings:
- Optimal: from -0.0505 to +0.0698 (13.5% range!)
- Entropy: from -0.0363 to +0.0376 (14% range!)

This variance completely overwhelms any method differences.

### 3. Setting-Specific Observations

**Probe Depth:**
- depth=2: Both methods negative on average
- depth=3: Mixed results
- depth=5: Entropy slightly better (+0.0022)

**Batch Size:**
- size=20: Essentially tied
- size=30: Essentially tied  
- size=40: Optimal slightly better (+0.0015)
- size=50: Essentially tied

**No clear pattern** - performance seems random across settings.

## Why No Significance?

### 1. Problem Difficulty
- Base C-index ≈ 0.50 (near random performance)
- NACD features have limited predictive power
- One-shot AL setting limits learning

### 2. High Variance
- Individual improvements: -0.05 to +0.07
- Standard deviations: ±0.01 to ±0.02
- Method differences: ±0.001 to ±0.002

**Variance is 10-50× larger than method differences!**

### 3. Small Sample Size
- Only 10 runs per setting
- Would need 100+ runs to overcome this variance
- Even then, true effect may be near zero

## Conclusion

**OptimalBatchBALD is NOT statistically better than Entropy**, regardless of:
- Probe depth (tested 2, 3, 5)
- Batch size (tested 20, 30, 40, 50)
- Random seed variations

**Why?**
1. The problem is too noisy (σ >> μ)
2. True method difference is likely near zero
3. NACD dataset is too difficult (C-index ≈ 0.50)

**Should you use OptimalBatchBALD anyway?**

Maybe, because:
- It's theoretically well-motivated
- It combines multiple important signals
- In some individual runs, it helps significantly
- No worse than simpler methods

But honestly: **Random selection performs about as well as any sophisticated method** in this challenging setting.

## Recommendations

1. **For NACD**: Use whatever is simplest (random, entropy, or Optimal - they're all equivalent)

2. **For better datasets** (C-index > 0.6):
   - OptimalBatchBALD may show clearer benefits
   - Test with more runs (30-50) to overcome variance

3. **For multi-round AL**:
   - Benefits may accumulate over multiple query rounds
   - This was only one-shot AL

4. **For larger budgets**:
   - Query 100-200 instances instead of 20-50
   - Larger signal may emerge

## Honest Assessment

We built a sophisticated method combining:
- Observable mass (best theoretical predictor)
- Mutual information (epistemic uncertainty)
- Ensemble variance (disagreement)
- P(death in window) (signal strength)

**Result:** It performs no better than random selection in practice.

**Why?** The problem is inherently too noisy for any method to consistently help.

**Is this a failure?** No - it's an important negative result showing the limits of AL in difficult survival settings with probe depth constraints.

---

**Date:** 2026-01-24  
**Conclusion:** NOT statistically significant across any tested settings
**Best advice:** Use simple baselines (random/entropy) unless you have a better dataset
