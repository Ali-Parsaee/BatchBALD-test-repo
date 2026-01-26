# Experimental Design Fixes: Fair Comparison

## The Problem: Unfair Comparison in Original Experiments

The original `compare_budget20.py` experiment had **critical flaws** that made the comparison unfair:

### Flaw 1: Different Initial Models per Method ❌

**Original code:**
```python
for acq_func, acq_name in acquisition_functions:
    # EACH METHOD TRAINS ITS OWN MODEL!
    model = BayesLinMtlr(...)  # New random initialization
    model = train_model(...)    # Different for each method

    initial_cindex = evaluate_model(...)  # Different starting points!
```

**Problem:** Each acquisition function started from a different model due to:
- Random weight initialization
- Stochastic gradient descent
- Early stopping variance

**Result:**
- BatchBALD average initial C-index: 0.6721
- Random average initial C-index: 0.6926
- **0.0205 point difference before any acquisition!**

### Flaw 2: Cascade Effect ❌

Methods ran sequentially and each one used the previous method's **final model** as its starting point:

```
Trial 1:
BatchBALD:    0.6682 → 0.6754  (+0.0072)
  ↓ Uses BatchBALD's final model!
BatchBALD_opt: 0.6754 → 0.6780  (+0.0026)
  ↓ Uses BatchBALD_opt's final model!
Variance:     0.6781 → 0.6805  (+0.0025)
  ↓
Entropy:      0.6805 → 0.6839  (+0.0034)
  ↓
Random:       0.6839 → 0.6873  (+0.0034)
```

**Problem:** Later methods inherit all previous improvements! Random benefits from ALL prior training.

### Flaw 3: Unfair Pre-filtering ❌

**Original code:**
```python
if 'BatchBALD' in acq_name:
    # Uses method's own model for pre-filtering
    filter_indices = prefilter_pool(model, ...)
```

**Problem:** Each method used its own (different) model for pre-filtering, creating different candidate pools.

## The Solution: Fixed Experimental Design ✅

### Fix 1: Single Shared Initial Model ✅

**New code:**
```python
# Train ONE base model per trial
base_model = train_model(X_train, y_labeled, e_labeled, ...)
initial_cindex = evaluate_model(base_model, ...)

for acq_func, acq_name in acquisition_functions:
    # Create a DEEP COPY for each method
    model = deep_copy_model(base_model, config)

    # Verify same starting point
    verify_cindex = evaluate_model(model, ...)
    assert abs(verify_cindex - initial_cindex) < 1e-6
```

**Result:** All methods now start from **exactly the same model** with the same initial C-index!

### Fix 2: Independent Evaluation ✅

**New code:**
```python
# Each method works on its own copy
model = deep_copy_model(base_model, config)

# Acquire samples using this copy
pool_indices = acq_func(model, ...)

# Update labels (independent for each method)
y_labeled_copy = y_labeled.copy()
e_labeled_copy = e_labeled.copy()
for idx in acquired_indices:
    y_labeled_copy[idx] = ...

# Retrain THIS method's model
model = train_model(model, X_train, y_labeled_copy, e_labeled_copy, ...)
```

**Result:** No cascade effect - each method's results are independent!

### Fix 3: Fair Pre-filtering ✅

**New code:**
```python
# Use the SHARED base model for pre-filtering
if 'BatchBALD' in acq_name:
    filter_indices = prefilter_pool(
        base_model,  # Same model for all methods!
        X_censored, ...
    )
```

**Result:** All methods see the same pre-filtered candidate pool!

## Additional Safeguards

### 1. Explicit Random Seed Control
```python
def artificially_censor_true(times, events, num_initial_samples=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # ...
```

**Purpose:** Ensures censoring is identical across methods in same trial.

### 2. Deep Copy Verification
```python
verify_cindex = evaluate_model(model, X_test, y_test, e_test, time_bins, config)
assert abs(verify_cindex - initial_cindex) < 1e-6, \
    f"Model copy verification failed! {verify_cindex} != {initial_cindex}"
```

**Purpose:** Catches any issues with model copying.

### 3. Independent Label Updates
```python
# Create copies so methods don't interfere
y_labeled_copy = y_labeled.copy()
e_labeled_copy = e_labeled.copy()
```

**Purpose:** Each method's label updates don't affect other methods.

## What This Fixes

### Before (Unfair):
```
BatchBALD:
  Initial: 0.6721 (random init #1)
  Final:   0.6829
  Improvement: +0.0108

Random:
  Initial: 0.6926 (random init #6, inherits all prior improvements)
  Final:   0.6928
  Improvement: +0.0002

Winner: Random (0.6928 > 0.6829)
```

### After (Fair):
```
ALL METHODS:
  Initial: 0.6XXX (SAME initial model)

BatchBALD:
  Improvement: +0.0XXX

Random:
  Improvement: +0.0XXX

Winner: Method with highest improvement!
```

## Expected Changes in Results

### Original (Unfair) Results - Budget=20:
1. Random: 0.6928 (improvement: +0.0002)
2. Entropy: 0.6926 (improvement: +0.0034)
3. Variance: 0.6892 (improvement: +0.0033)
4. BatchBALD: 0.6829 (improvement: +0.0108) ← HIGHEST improvement but last place!

### Expected (Fair) Results - Budget=20:
Based on improvement values, we expect:
1. **BatchBALD**: +0.0108 (highest improvement should win!)
2. Entropy: +0.0034
3. Variance: +0.0033
4. Random: +0.0002

**Hypothesis:** BatchBALD will now WIN because all methods start from the same model, and BatchBALD extracts the most value from acquired samples.

## Verification Checklist

The new experiment verifies:

- ✅ All methods use **same base model** per trial
- ✅ All methods have **same initial C-index** per trial
- ✅ Each method has **independent copy** of base model
- ✅ Pre-filtering uses **shared base model**
- ✅ Random seed is **controlled**
- ✅ Label updates are **independent** per method
- ✅ No cascade effects
- ✅ Deep copy verification via assertion

## Files

- **Original (Flawed):** `compare_budget20.py`
- **Fixed (Fair):** `fair_comparison_budget20.py`
- **Analysis of Flaw:** `analyze_initial_cindices.py`
- **This Document:** `EXPERIMENTAL_FIXES.md`

## Conclusion

The original experiment was **fundamentally flawed** due to:
1. Different random initializations per method
2. Cascade effects from sequential execution
3. Method-specific pre-filtering

The new experiment fixes all these issues, ensuring a **fair comparison** where:
- All methods start from the same initial model
- Results are independent
- We measure what we intend to: **acquisition function effectiveness**

This is the correct way to evaluate active learning acquisition functions!
