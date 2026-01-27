# CBALD-Diverse Variants: Making It More Powerful

## Goal
Maximize CBALD-Diverse performance by testing intelligent variations on the winning approach.

## Current Baseline
**CBALD-Diverse**: +0.0180 ± 0.0064 (beats C-BALD's +0.0173)

## New Variants Implemented

### 1. CBALD-Diverse-Adaptive
**Hypothesis**: Adapt diversity ratio as batch fills for better exploitation/exploration balance.

**Strategy**:
- **Early selections** (first samples): 90% C-BALD + 10% diversity
  - Prioritize absolute highest-value samples first
  - Minimal diversity constraint when batch is empty
- **Later selections** (final samples): 50% C-BALD + 50% diversity
  - Increase exploration as batch fills
  - More diversity when redundancy risk is higher
- **Smooth transition**: Ratio adapts linearly based on batch progress

**Implementation**:
```python
progress = len(selected) / batch_size  # 0 to 1
current_ratio = 0.1 + (0.5 - 0.1) * progress  # 10% → 50%
combined_score = (1 - current_ratio) * cbald + current_ratio * diversity
```

**Expected Benefit**:
- Gets best exploitation (top value samples) early
- Adds more exploration (diversity) later
- Should beat fixed 30% ratio

### 2. CBALD-Diverse-NoFilter
**Hypothesis**: Remove variance pre-filtering bias to find better samples.

**Strategy**:
- Score **ALL ~1961 censored samples** with C-BALD (no pre-filter)
- Select diverse subset from top candidates
- Removes potential bias from variance-based filtering

**Implementation**:
```python
# No pre-filtering step - score all samples
cbald_scores = cbald_score(model, X_pool_all, ...)  # All 1961
top_candidates = np.argsort(cbald_scores)[-60:]  # Top 3x budget
# Then apply diversity filtering as usual
```

**Expected Benefit**:
- May find high C-BALD score samples missed by variance filter
- Variance and C-BALD may disagree on uncertainty
- Trade-off: Slower (more samples to score)

### 3. CBALD-TwoStage
**Hypothesis**: Guarantee we get absolute best samples, then add diversity.

**Strategy**:
- **Stage 1** (50% of budget): Pure C-BALD exploitation
  - Select top 10 samples by C-BALD score (no diversity)
  - Guarantees highest-value samples included
- **Stage 2** (50% of budget): Diverse exploration
  - Select remaining 10 with diversity filtering
  - Diversity computed relative to Stage 1 selections

**Implementation**:
```python
# Stage 1: Pure exploitation
stage1_indices = top_candidates[:budget//2]  # Top 10 by C-BALD

# Stage 2: Add diversity
stage2_indices = diverse_selection(
    remaining_candidates,
    budget//2,
    already_selected=stage1_indices
)
```

**Expected Benefit**:
- Hard guarantee on best samples
- No risk of diversity pushing out top performers
- Clear separation of exploitation vs exploration

## Comparison Matrix

| Variant | Exploitation Focus | Diversity Mechanism | Pre-filtering | Speed |
|---------|-------------------|---------------------|---------------|-------|
| **CBALD-Diverse** (Original) | Fixed 70% | Greedy JS (30%) | Variance (500) | Fast |
| **CBALD-Adaptive** | Adaptive 90%→50% | Greedy JS (10%→50%) | Variance (500) | Fast |
| **CBALD-NoFilter** | Fixed 70% | Greedy JS (30%) | None (all) | Slow |
| **CBALD-TwoStage** | Guaranteed 50% | Greedy JS (50%) | Variance (500) | Fast |
| **C-BALD** (Reference) | Pure 100% | None | Variance (500) | Fast |

## Theoretical Predictions

### Most Likely to Win: CBALD-Adaptive ⭐
**Reasoning**:
- Keeps C-BALD's proven value weighting
- Better exploitation/exploration balance
- No speed penalty
- Addresses potential weakness (fixed ratio)

**Expected improvement**: +0.0185 to +0.0195 (small gain)

### Dark Horse: CBALD-TwoStage
**Reasoning**:
- Guarantees best samples included
- May be what we actually need (pure value first)
- Simple, interpretable strategy

**Expected improvement**: +0.0180 to +0.0200 (similar or small gain)

### Wildcard: CBALD-NoFilter
**Reasoning**:
- If variance filter is biased, could find better samples
- But: Slower, and variance is usually good proxy
- Higher risk, potentially higher reward

**Expected improvement**: +0.0175 to +0.0210 (wider range, uncertain)

### Baseline: CBALD-Diverse (Original)
**Current**: +0.0180 ± 0.0064
**Expected**: Still competitive, may remain best

## Test Design

**Trials**: 5 (for statistical significance)
**Budget**: 20 samples
**Comparisons**: Paired t-tests vs CBALD-Diverse baseline

**Fair comparison**:
- Same shared base model for all methods
- Same random seeds across trials
- Same evaluation protocol

## Success Criteria

**Primary Goal**: Beat CBALD-Diverse's +0.0180
**Secondary Goal**: Statistical significance (p < 0.05)
**Stretch Goal**: Beat +0.0190 (significant improvement)

## Why These Variants?

**Focus on simple, interpretable improvements**:
1. ✅ No complex mechanisms (no BatchBALD joint entropy)
2. ✅ Build on proven C-BALD scoring
3. ✅ Address specific potential weaknesses
4. ✅ Fast to implement and test

**Not tested** (saved for later if needed):
- CBALD-BatchBALD hybrid (too complex, likely overkill)
- Higher budgets (50, 100) - different research question
- Multiple diversity metrics - diminishing returns

## Expected Outcome

**Best case**: One variant significantly beats CBALD-Diverse
- Demonstrates clear path to improvement
- Provides new state-of-the-art method

**Realistic case**: Small improvements, variants similar
- Validates CBALD-Diverse design is already near-optimal
- Multiple good options for practitioners

**Worst case**: No improvements
- CBALD-Diverse's 30% ratio is already optimal
- Fixed ratio is sufficient for this problem
- Still learned what doesn't help

## Next Steps After Results

**If a variant wins**:
1. Run additional trials (10 total) for robustness
2. Test at different budgets (10, 30, 50)
3. Analyze which samples differ between methods
4. Document new champion in paper/repo

**If original wins**:
1. Document that simple 30% ratio is robust
2. Test sensitivity to diversity_ratio parameter
3. Focus on other improvements (budget size, etc.)

**Either way**:
- Commit winning method(s) to repo
- Update BREAKTHROUGH_RESULTS.md
- Push to GitHub

---

**Status**: Test running (5 trials × 5 methods = 25 training runs)
**Expected completion**: ~30-45 minutes
**Output**: `cbald_variants_results.txt`
