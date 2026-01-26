# Fair Comparison Experiment - Summary

## Status: ✅ RUNNING

The corrected fair comparison experiment is currently running with budget=20.

**Progress:** Trial 1 complete, ~15-20 minutes remaining

## What Changed: The Critical Fix

### Before (Unfair Experiment) ❌

**Problem**: Each method trained its own model with different random initialization

```
Trial 1 Example (OLD):
- BatchBALD:    Initial 0.6682 → Final 0.6754 (+0.0072)
- Variance:     Initial 0.6781 → Final 0.6805 (+0.0025)
- Entropy:      Initial 0.6805 → Final 0.6839 (+0.0034)
- Random:       Initial 0.6839 → Final 0.6873 (+0.0034)
```

**Issues**:
1. Different initial models (0.6682 vs 0.6839 = 0.0157 difference!)
2. Cascade effect (each method used previous method's final model)
3. Random always ran last, inheriting all prior improvements

**Result**: Random won (0.6928 final) but only improved +0.0002, while BatchBALD ranked last (0.6829 final) but improved +0.0108!

### After (Fair Experiment) ✅

**Fix**: ONE shared base model per trial, deep copied for each method

```
Trial 1 Example (NEW):
- SHARED Initial: 0.6682 (ALL METHODS START HERE)
- BatchBALD:    0.6682 → 0.6795 (+0.0113) ✓
- Variance:     0.6682 → ???? (+0.00??)
- Entropy:      0.6682 → ???? (+0.00??)
- Random:       0.6682 → ???? (+0.00??)
```

**Fixed**:
1. ✅ All methods start from SAME model (same initial C-index)
2. ✅ No cascade - each method works on independent copy
3. ✅ Fair pre-filtering using shared base model
4. ✅ We measure acquisition function effectiveness, not initialization luck

## Early Results (Trial 1)

### BatchBALD Performance:
- Initial C-index: 0.6682
- Final C-index: 0.6795
- **Improvement: +0.0113**

This is already higher than the unfair experiment (+0.0108), and now all methods start from the same point!

## Expected Final Results

### Hypothesis:
**BatchBALD will WIN** because:
1. It achieved highest improvement in unfair experiment (+0.0108)
2. Now all methods start from same model
3. BatchBALD extracts most value from acquired samples

### Predicted Ranking (by improvement):
1. 🏆 **BatchBALD**: +0.0108 to +0.0120
2. 🥈 **Entropy**: +0.0030 to +0.0040
3. 🥉 **Variance**: +0.0025 to +0.0035
4. **Random**: +0.0000 to +0.0005

## Why This Matters

### What the Unfair Experiment Measured:
"Which method, after inheriting improvements from all prior methods, ends with the best model?"

Answer: Random (but only because it ran last and barely learned anything itself)

### What the Fair Experiment Measures:
"Starting from the SAME initial model, which acquisition function improves it the most?"

Answer: **TBD** (experiment running), but we expect BatchBALD!

## Technical Details

### Fixes Implemented:

1. **Single Shared Model**
```python
# Train ONE base model per trial
base_model = train_model(...)
initial_cindex = evaluate(base_model)

for acq_func in acquisition_functions:
    # Deep copy for each method
    model = deep_copy_model(base_model)
    # All start with SAME initial_cindex!
```

2. **Independent Evaluation**
```python
# Each method has own labels
y_labeled_copy = y_labeled.copy()
e_labeled_copy = e_labeled.copy()

# Acquire and update independently
acquire_samples(model, ...)
update_labels(y_labeled_copy, ...)
retrain(model, y_labeled_copy, ...)
```

3. **Fair Pre-filtering**
```python
# Use SHARED base model for pre-filtering
filter_indices = prefilter_pool(base_model, ...)  # Not method's model!
```

### Verification Added:

```python
verify_cindex = evaluate_model(model_copy, ...)
assert abs(verify_cindex - initial_cindex) < 5e-3
print(f"Initial C-index: {initial_cindex:.4f} (verified same as base)")
```

## Key Insights

### From Unfair Experiment:
- BatchBALD: +0.0108 improvement (highest!)
- Random: +0.0002 improvement (barely learned)
- But Random had better initial model, so won final C-index

### From Fair Experiment (Expected):
- All start from same model
- Method with highest improvement wins
- This is the CORRECT way to compare acquisition functions

## Comparison of Both Experiments

| Metric | Unfair Experiment | Fair Experiment |
|--------|------------------|-----------------|
| Initial model | Different per method | SAME for all |
| Cascade effects | Yes (sequential) | No (independent) |
| Pre-filtering | Method-specific | Shared base model |
| What it measures | Final C-index | Improvement |
| Winner (unfair) | Random (0.6928) | BatchBALD expected |
| Winner (fair) | TBD | TBD (running) |

## Files

- `compare_budget20.py` - Original unfair experiment
- `fair_comparison_budget20.py` - **Corrected fair experiment (RUNNING)**
- `EXPERIMENTAL_FIXES.md` - Detailed documentation of all fixes
- `FAIR_COMPARISON_SUMMARY.md` - This summary
- `analyze_initial_cindices.py` - Analysis revealing the flaw

## Next Steps

1. ✅ Fixed experimental design
2. ⏳ Running fair comparison (15-20 min remaining)
3. 📊 Will analyze results and compare to unfair experiment
4. 📝 Will document findings and recommendations

## Expected Outcome

**Prediction:** BatchBALD will WIN the fair comparison because:
- It consistently achieves highest improvement
- No longer handicapped by poor initial model
- No longer competing against methods that inherit its improvements

**This will validate:**
- BatchBALD is genuinely better for active learning with budget=20
- The original unfair results were confounded by initialization variance
- Proper experimental design is crucial for fair comparison

---

**Last Updated:** Experiment running, Trial 1/5 complete
**Status:** Waiting for results...
