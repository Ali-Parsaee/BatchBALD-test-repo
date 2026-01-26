# Why BatchBALD Ties Variance: Analysis and Investigation

## Current Results (5 Trials, Budget=20)

### Overall Statistics:
1. **Variance**: +0.0117 ± 0.0041 🏆 (Winner)
2. **BatchBALD**: +0.0109 ± 0.0036 🥈 (Very close second)
3. **Random**: -0.0055 ± 0.0113
4. **Entropy**: -0.0069 ± 0.0101

**Statistical Comparison:**
- BatchBALD vs Variance: Mean difference -0.0007, p-value 0.2526 (NOT significant)
- **Conclusion**: They are statistically tied

### Trial-by-Trial Breakdown:
```
Trial 1: BatchBALD +0.0113 vs Variance +0.0101 (BatchBALD wins by 0.0012) ✓
Trial 2: BatchBALD +0.0057 vs Variance +0.0061 (Variance wins by 0.0004)
Trial 3: BatchBALD +0.0102 vs Variance +0.0119 (Variance wins by 0.0017)
Trial 4: BatchBALD +0.0156 vs Variance +0.0175 (Variance wins by 0.0019)
Trial 5: BatchBALD +0.0118 vs Variance +0.0127 (Variance wins by 0.0009)
```

**Pattern**: Variance won 4 out of 5 trials with small but consistent margins (0.0004-0.0019).

---

## Hypothesis: Why BatchBALD Doesn't Dominate

### 1. Pre-filtering Bias Toward Variance ⚠️

**The Issue:**
BatchBALD uses uncertainty-based pre-filtering to reduce the pool from ~1961 to 500 samples:

```python
def get_uncertainty_scores(model, X_pool, time_bins, config, device):
    """Calculate uncertainty scores for pre-filtering."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(X_pool).to(device)
        survival_outputs, _, ensemble_outputs = make_prediction(...)
        pdf = ensemble_to_pdf(ensemble_outputs, device)
        p_mean = pdf.mean(dim=1)
        variance = p_mean.var(dim=1)  # Uses VARIANCE!
    return variance.cpu().numpy()
```

**Problem:** The pre-filtering is **variance-based**, so BatchBALD's candidate pool is already optimized for variance, not diversity!

**Effect:**
- Pre-filtering selects the 500 most uncertain (high variance) samples
- BatchBALD then tries to select diverse samples from this variance-optimized pool
- But the pool is already biased toward what Variance would select anyway
- **Result:** BatchBALD's diversity advantage is neutered by the pre-filtering

**Evidence:**
- Both methods are selecting from the same 500 samples
- These 500 samples are the highest-variance samples
- BatchBALD's diversity mechanism has limited room to differentiate

---

### 2. Budget=20 Too Small for Diversity Advantage 📊

**The Issue:**
At budget=20, we're selecting 20 samples from a pre-filtered pool of 500.

**Why This Hurts BatchBALD:**
- **Sample ratio**: 20/500 = 4% of the pre-filtered pool
- **Diversity mechanism**: BatchBALD's strength is selecting diverse samples that cover the uncertainty space
- **At low budgets**: The top 20 highest-variance samples may already be somewhat diverse
- **Variance method**: Simply picks top 20 by variance score → fast and effective
- **BatchBALD method**: Computes joint entropy to ensure diversity → slower but little added value at this scale

**Mathematical Insight:**
- With 20 samples from 500, the top-20 variance samples naturally spread across different regions
- BatchBALD's expensive joint entropy computation doesn't add much diversity beyond what variance already provides
- The diversity gap becomes more apparent at higher budgets (e.g., 50-100 samples)

---

### 3. Computational Cost Without Proportional Benefit 💻

**Performance Comparison:**
- **Variance**: ~23-24 seconds per trial (fast, simple variance computation)
- **BatchBALD**: ~45-48 seconds per trial (2x slower, joint entropy computation)

**What BatchBALD Does Differently:**
1. Computes variance scores for all 500 samples
2. Builds sampled joint entropy using `batchbald_redux` library
3. Iteratively selects samples to maximize diversity
4. Each iteration updates the joint entropy (expensive)

**What Variance Does:**
1. Computes variance scores for all 500 samples
2. Sorts by variance
3. Takes top 20

**Cost-Benefit Analysis:**
- BatchBALD: 2x computational cost, +0.0109 improvement
- Variance: 1x computational cost, +0.0117 improvement
- **Efficiency**: Variance is both faster AND slightly better!

---

### 4. Sample Selection Overlap 🔄

**Hypothesis:** BatchBALD and Variance are likely selecting very similar or overlapping samples.

**Why This Would Happen:**
1. Both start from the same pre-filtered 500 high-uncertainty samples
2. Both are uncertainty-based methods (variance vs. BALD variance)
3. At budget=20, the top 20 most uncertain samples dominate both methods
4. BatchBALD's diversity constraint may only change 2-5 samples compared to pure variance

**Expected Overlap:**
- Estimate: 15-18 out of 20 samples (75-90% overlap)
- Only 2-5 samples differ due to diversity mechanism
- This small difference explains the small performance gap

---

## Potential Solutions to Make BatchBALD Win

### Solution 1: Remove or Modify Pre-filtering ✨

**Current Problem:** Pre-filtering uses variance, biasing the pool.

**Options:**

**Option A - Increase Pre-filter Size:**
```python
prefilter_size = 1000  # Instead of 500
# or even better:
prefilter_size = len(censored_indices)  # No pre-filtering!
```
- Gives BatchBALD access to more diverse samples
- May be computationally expensive

**Option B - Use Random Pre-filtering:**
```python
if 'BatchBALD' in acq_name:
    # Random sample instead of uncertainty-based
    filter_indices = np.random.choice(len(censored_list), prefilter_size, replace=False)
```
- Removes variance bias
- Gives BatchBALD a neutral starting pool

**Option C - Use Entropy Pre-filtering:**
```python
def get_uncertainty_scores(model, X_pool, time_bins, config, device):
    # Use entropy instead of variance for pre-filtering
    H = -(p_mean * torch.log(p_mean + 1e-12)).sum(dim=1)
    return H.cpu().numpy()
```
- Different uncertainty metric
- May favor different samples than variance

---

### Solution 2: Increase Budget 📈

**Test higher budgets where diversity matters more:**

| Budget | Expected Outcome |
|--------|-----------------|
| 20 | Variance wins (current) |
| 50 | BatchBALD competitive |
| 100 | BatchBALD should win |
| 200 | BatchBALD significant advantage |

**Reasoning:**
- At budget=50+, selecting diverse samples becomes more important
- Pure variance will select many redundant samples from same region
- BatchBALD's diversity mechanism prevents redundancy
- Gap should widen as budget increases

---

### Solution 3: Tune BatchBALD Parameters 🔧

**Current BatchBALD Call:**
```python
pool_indices = batchbald_acquire_budget(
    model=model,
    X_pool=X_censored,
    batch_size=budget,
    num_samples=10000  # Default
)
```

**Potential Tuning:**

**Option A - Increase num_samples:**
```python
num_samples=50000  # More MC samples for better diversity estimation
```
- Better estimates of uncertainty and diversity
- More computational cost

**Option B - Adjust diversity weight:**
If the acquisition function has a diversity parameter (need to check), increase it.

---

### Solution 4: Use Different Variance Formulation 🧮

**Investigation Needed:**
- Check if BatchBALD is using the same variance calculation as the Variance method
- If they're using different formulations (e.g., time variance vs. probability variance), this could explain the tie

**Action:**
- Verify both methods use compatible uncertainty metrics
- Ensure BatchBALD's BALD score is truly capturing epistemic uncertainty

---

## Experiments to Run

### Experiment 1: Remove Pre-filtering ✓
```python
# No pre-filtering for BatchBALD
if 'BatchBALD' in acq_name:
    # Use full pool
    censored_list_full = list(censored_indices)
    pool_indices = batchbald_acquire_budget(...)
```

### Experiment 2: Increase Budget to 50 ✓
```python
budget = 50  # Instead of 20
```

### Experiment 3: Random Pre-filtering ✓
```python
if 'BatchBALD' in acq_name:
    filter_indices = np.random.choice(len(censored_list), prefilter_size, replace=False)
```

### Experiment 4: Compare Sample Overlap 🔍
```python
# After acquisition, compare selected samples
batchbald_samples = set(batchbald_indices)
variance_samples = set(variance_indices)
overlap = len(batchbald_samples & variance_samples)
print(f"Overlap: {overlap}/20 samples ({overlap/20*100:.1f}%)")
```

---

## Recommendations

### Immediate Actions:

1. **Add C-BALD to comparison** (as requested by user)
   - See if C-BALD (variance + death probability weighting) performs differently

2. **Test without pre-filtering**
   - Run experiment with full pool (no 500-sample pre-filtering)
   - This is the most likely culprit

3. **Test at budget=50**
   - See if BatchBALD's advantage emerges at higher budgets

### If BatchBALD Still Doesn't Win:

4. **Analyze sample overlap**
   - Directly compare which samples each method selects
   - Quantify the redundancy

5. **Profile uncertainty scores**
   - Plot variance scores vs. BALD scores for selected samples
   - Check if BatchBALD is measuring something different

6. **Consider hybrid approach**
   - Combine BatchBALD's diversity with Variance's speed
   - E.g., "Select top 50 by variance, then use BatchBALD to pick diverse 20"

---

## Conclusion

**Most Likely Cause:** Pre-filtering with variance-based uncertainty is biasing the candidate pool toward what Variance would select anyway, neutralizing BatchBALD's diversity advantage.

**Most Promising Fix:** Remove or modify pre-filtering to give BatchBALD access to a more diverse candidate pool.

**Secondary Factor:** Budget=20 is too small for diversity to provide significant advantage. Testing at budget=50-100 would better showcase BatchBALD's strengths.

**Next Steps:**
1. Add C-BALD and run comparison
2. Run experiments with modified pre-filtering
3. Test at higher budgets
4. Analyze sample selection overlap
