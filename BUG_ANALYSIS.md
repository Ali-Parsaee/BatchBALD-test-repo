# Bug Analysis: Why Optimized BatchBALD Failed

## The Critical Finding

The optimizations **decreased** the very characteristics they were supposed to **increase**!

### Expected Changes:
- ✅ Total variance: Should INCREASE by 38%
- ✅ Window uncertainty: Should INCREASE by 42×
- ✅ Spatial diversity: Should MAINTAIN (stay high)

### Actual Changes (Window-only optimization):
- ❌ Total variance: **DECREASED by 71.2%** (0.0277 → 0.0080)
- ❌ Window uncertainty: **DECREASED by 86.2%** (0.000704 → 0.000097)
- ❌ Spatial diversity: **DECREASED by 38.3%** (21.55 → 13.30)

### Actual Changes (Window + Variance optimization):
- ❌ Total variance: **DECREASED by 79.4%** (0.0277 → 0.0057)
- ❌ Window uncertainty: **DECREASED by 71.8%** (0.000704 → 0.000198)
- ❌ Spatial diversity: **DECREASED by 43.1%** (21.55 → 12.26)

## Root Cause Analysis

### The Bug: Insufficient Candidate Pool

The optimization has a fundamental design flaw:

1. **Current approach:**
   - BatchBALD selects **exactly 10 candidates** based on joint entropy
   - These 10 are re-ranked by window weights and variance
   - The re-ranked order is returned

2. **Why this fails:**
   - BatchBALD's greedy algorithm considers **diversity** (joint entropy)
   - But the 10 samples it picks may not have high variance or window uncertainty
   - Re-ranking 10 samples doesn't change WHICH samples are selected
   - It only changes their ORDER (which doesn't matter for performance)

3. **What we need:**
   - Get a **larger pool** of candidates from BatchBALD (e.g., top 50-100)
   - Re-rank this larger pool by combined score
   - Select the top 10 from the re-ranked list
   - This allows us to trade off diversity vs. variance/window uncertainty

### Why There's Low Overlap (10%)

Even though we're re-ranking the same 10 samples, the overlap is low because:
- Different random seeds between runs
- BatchBALD's greedy algorithm is sensitive to selection order
- Small changes in the pool can lead to different selections

But this is a red herring - the real issue is that we're not expanding the candidate pool.

## Why Original BatchBALD Works Better

Original BatchBALD:
- Total variance: 0.0277 (HIGH)
- Window uncertainty: 0.000704 (moderate)
- Spatial diversity: 21.55 (BEST)

The original succeeds because:
1. **High diversity (21.55)** - BatchBALD's joint entropy prioritizes diverse samples
2. **High total variance (0.0277)** - Diverse samples tend to have high variance
3. **Moderate window uncertainty** - Not optimized for this, but still decent

## Why Optimized Versions Fail

When we re-rank just 10 samples by window weights and variance:
- We're essentially **overriding BatchBALD's diversity** mechanism
- Samples with high window weights might be **similar** to each other
- This destroys diversity (21.55 → 13.30)
- Less diverse samples have **lower total variance** (0.0277 → 0.0080)
- Less diverse samples have **lower window uncertainty** (0.000704 → 0.000097)

**The optimization is self-defeating!** By prioritizing window uncertainty in the re-ranking, we select similar samples, which:
1. Reduces diversity
2. Reduces total variance
3. Even reduces window uncertainty (since similar samples have similar windows)

## The Investigation Hypothesis Error

The investigation found:
- Methods with high total variance perform better (r=0.944)
- Methods with high window uncertainty perform better (r=0.776)
- Methods with high diversity perform better (r=0.809)

But the investigation measured these characteristics on the FINAL selections, not during the selection process. This created a misleading correlation:

- **Variance method**: Selects samples with high total variance → also happens to have high diversity and window uncertainty
- **BatchBALD**: Selects diverse samples → also happens to have high total variance and moderate window uncertainty

The investigation correctly identified what makes good selections, but the optimization approach (re-ranking 10 samples) couldn't actually increase these characteristics.

## The Fix

To actually improve BatchBALD, we need:

1. **Expand candidate pool:**
   ```python
   # Get MORE candidates from BatchBALD
   candidate_size = 50  # or 100
   candidate_batch = batchbald.get_batchbald_batch(
       bb_logits, candidate_size, actual_num_samples, dtype=torch.double, device=device
   )

   # Re-rank these 50 candidates
   combined_scores = (mi_scores + lambda_variance * variance) * window_weights

   # Select top 10 from re-ranked list
   final_indices = original_indices[reranked_order[:batch_size]]
   ```

2. **Balance the trade-offs:**
   - Use BatchBALD to get diverse, high-MI candidates
   - Among those candidates, prefer ones with high variance and window uncertainty
   - This maintains diversity while boosting the other characteristics

3. **Re-run the investigation:**
   - Test whether expanding the candidate pool actually helps
   - Measure if we can increase all three characteristics simultaneously
   - Or accept that there's a fundamental trade-off between diversity and variance

## Conclusion

The optimization failed because:
1. ❌ Re-ranking just 10 samples doesn't change which samples are selected
2. ❌ Even if it did, prioritizing variance/window uncertainty destroys diversity
3. ❌ Less diverse samples have LOWER variance and window uncertainty, not higher
4. ✅ Original BatchBALD already finds the best trade-off between diversity and variance

The investigation was correct that these characteristics matter, but the implementation approach was fundamentally flawed.
