# Root Cause Analysis: The Missing Preprocessing

## The Real Bug

The optimized version **skips critical preprocessing steps** that the original BatchBALD uses before running the BatchBALD algorithm. This causes BatchBALD to select fundamentally different (and worse) samples.

## Comparison: What Each Version Does

### Original BatchBALD (`Model_stuff/acquisition.py` lines 593-652):
```python
1. Apply _fix_survival_timebins_increment()  # Window collapse
2. Apply OUT-OF-WINDOW DOWNWEIGHTING (lines 598-605)
   - Weight out-of-window bin by censor_ratio
   - Reduces emphasis on far-future uncertainty
3. Apply IN-WINDOW EMPHASIS (lines 608-622)
   - Boost probabilities within reveal window
   - Uses inwindow_gamma parameter
4. Apply TEMPERATURE SMOOTHING (lines 625-628)
   - Smooth probabilities with temperature > 1
   - Reduces overconfidence
5. Compute binary MI from preprocessed probabilities
6. Run BatchBALD with these modified logits
7. Optional diversity filtering
```

### Optimized BatchBALD (`optimized_batchbald.py` lines 131-195):
```python
1. Apply _fix_survival_timebins_increment()  # Window collapse
2. ❌ SKIP out-of-window downweighting
3. ❌ SKIP in-window emphasis
4. ❌ SKIP temperature smoothing
5. Compute binary MI from UN-PROCESSED probabilities
6. Run BatchBALD with these raw logits
7. Compute window weights and variance AFTER selection
8. Re-rank the 10 selected samples
```

## Why This Breaks Everything

### The Problem:
The preprocessing steps in the original BatchBALD are not just cosmetic - they **fundamentally shape which samples BatchBALD selects**:

1. **Out-of-window downweighting**: Reduces the weight of the "unknown/censored" outcome based on censoring time. This guides BatchBALD to prefer samples where the reveal window is more informative.

2. **In-window emphasis**: Boosts probabilities within the reveal window, making BatchBALD prioritize samples with high uncertainty IN THE WINDOW we can actually probe.

3. **Temperature smoothing**: Prevents BatchBALD from overcommitting to very confident predictions, maintaining diversity.

### The Result:
Without these preprocessing steps:
- BatchBALD sees RAW probabilities that don't emphasize the reveal window
- It selects samples based on generic mutual information, not window-specific information
- These samples happen to have:
  - 71% LOWER total variance
  - 86% LOWER window uncertainty
  - 38% LOWER spatial diversity

### Why Re-ranking Doesn't Help:
The optimized version tries to fix this by re-ranking AFTER BatchBALD has already selected 10 poor samples:
```python
# BatchBALD already selected 10 samples based on raw probabilities
# Now we compute window weights and variance for just these 10
# But it's too late - these 10 are already poor choices!
combined_scores = (mi_scores + lambda_variance * variance) * window_weights
```

This is like trying to find the best restaurants by:
1. Randomly selecting 10 restaurants
2. Then ranking those 10 by food quality

Instead of:
1. Ranking ALL restaurants by food quality
2. Then selecting the top 10

## Why The Investigation Was Misleading

The investigation correctly identified that methods with:
- High total variance
- High window uncertainty
- High spatial diversity

...perform better. But it measured these characteristics on the FINAL selections, not during the selection process.

The investigation didn't reveal that the original BatchBALD achieves these characteristics through **preprocessing**, not through post-selection reweighting.

## The Correct Interpretation

The original BatchBALD succeeds because:
1. ✅ Preprocessing guides selection toward window-informative samples
2. ✅ BatchBALD's joint entropy maintains diversity
3. ✅ The combination naturally produces high variance, window uncertainty, and diversity

The optimized version fails because:
1. ❌ Skips preprocessing, so BatchBALD sees wrong signal
2. ❌ Tries to fix with post-hoc reranking
3. ❌ Too late - already selected poor samples

## The Fix Options

### Option 1: Accept Original Is Optimal
- The original BatchBALD already incorporates window-awareness through preprocessing
- The investigation confirmed it has good characteristics
- No need to "optimize" further

### Option 2: Try Preprocessing + Optimization
- Keep all original preprocessing steps
- Run BatchBALD with processed probabilities
- Get a LARGER candidate pool (50-100 samples)
- Re-rank by combined scores (MI + variance) * window_weights
- Select top 10

But Option 2 might not help because:
- The preprocessing already optimizes for window information
- Getting more candidates might dilute BatchBALD's diversity guarantees
- The original might already be at the optimal trade-off point

### Option 3: Investigate Preprocessing Parameters
- The original uses default parameters (temperature=1.25, inwindow_gamma=0.3, etc.)
- Maybe these could be tuned
- But this is parameter tuning, not algorithmic improvement

## Conclusion

**The optimization attempt failed because it skipped the preprocessing that makes BatchBALD work well in the first place.**

The original BatchBALD is not just using "generic" mutual information - it's using mutual information computed from carefully preprocessed probabilities that emphasize window-specific uncertainty.

The investigation's findings were correct (variance, window uncertainty, and diversity matter), but the implementation tried to add these as post-processing instead of recognizing they're already baked into the preprocessing.

**Recommendation**: Accept that the original BatchBALD is already well-optimized for this task. The preprocessing steps effectively incorporate the characteristics identified by the investigation.
