# Investigation: Can BatchBALD Help?

## Question
Does BatchBALD add any value? Can we use it on top of C-BALD for better diversity than CBALD-Diverse?

## Current Results Summary

| Method | Mean Improvement | Rank | Notes |
|--------|-----------------|------|-------|
| **CBALD-Diverse** | +0.0180 ± 0.0064 | 🏆 1st | C-BALD + JS diversity |
| C-BALD | +0.0173 ± 0.0062 | 🥈 2nd | Information value only |
| Variance | +0.0117 ± 0.0041 | 🥉 3rd | Simple baseline |
| **BatchBALD** | +0.0109 ± 0.0036 | 4th | **Worse than Variance!** |

### Key Observation
**BatchBALD performs WORSE than simple Variance** despite being 2x slower (50s vs 22s).

## Analysis: Does BatchBALD Help At All?

### BatchBALD's Strengths (Theoretical)
1. **Joint entropy maximization**: Selects batch that maximizes information gain together
2. **Diversity mechanism**: Avoids redundant samples via mutual information
3. **Principled approach**: Bayesian framework for batch selection

### BatchBALD's Weaknesses (Observed)
1. **No information value weighting**: Treats all uncertain samples equally
2. **Pre-filtering bias**: Uses variance to reduce pool from ~1961 to 500, biasing toward what Variance would select
3. **Budget too small**: At budget=20 (4% of pool), diversity advantage doesn't emerge
4. **Complexity without benefit**: 2x slower than alternatives but performs worse

### Why BatchBALD Underperforms

**Root cause**: BatchBALD optimizes for uncertainty + diversity, but ignores **information value**.

In survival analysis with censored data:
- Some uncertain samples are valuable (high death probability in reveal window)
- Other uncertain samples are less valuable (low death probability in reveal window)
- BatchBALD treats them equally → selects diverse but low-value samples

**Evidence**: BatchBALD (+0.0109) < Variance (+0.0117)
- If diversity was beneficial, BatchBALD should beat Variance
- Since it doesn't, the diversity mechanism is actually harmful at this budget

## Previous Attempts to Combine BatchBALD with C-BALD

### Approach 1: BatchBALD-C v1 (Post-processing reranking)
**Strategy**: Run BatchBALD on 3x candidates, then rerank by multiplicative score
```python
combined_score = batchbald_score * (0.5 + 0.5 * death_prob)
```
**Result**: +0.0086 (FAILED - 4th place, worse than BatchBALD)

### Approach 2: BatchBALD-C v2 (Multiplicative combination)
**Strategy**: Select 2x candidates by C-BALD, then rerank with BatchBALD scores
```python
combined_score = cbald_score * batchbald_score
```
**Result**: +0.0110 (FAILED - barely better than BatchBALD)

### Approach 3: BatchBALD-C v3 (Pre-processing weight)
**Strategy**: Weight input logits by death probability BEFORE BatchBALD
```python
logits[i, :, s:e] *= (0.5 + 0.5 * death_prob[i])
```
**Result**: +0.0109 (FAILED - exact tie with BatchBALD)

### Why All Failed
**BatchBALD's diversity mechanism fights against information value weighting.**

When you try to bias BatchBALD toward high-value samples:
1. High-value samples cluster together (similar predictions)
2. BatchBALD's diversity constraint penalizes this clustering
3. It pulls back toward different samples (lower-value)
4. Net result: No improvement or worse performance

## Can We Use BatchBALD's Joint Entropy for Better Diversity?

### Current CBALD-Diverse Approach
- Select top 3x candidates by C-BALD score
- Greedily pick diverse subset using **pairwise JS divergence**
- Balance: 70% C-BALD score + 30% diversity

**Advantage**: Simple, fast, effective
**Limitation**: JS divergence is pairwise, doesn't account for joint information

### Proposed: CBALD-BatchBALD Hybrid

**Idea**: Use BatchBALD's joint entropy computation on C-BALD's top candidates

**Algorithm**:
```
1. Score all samples with C-BALD (information value)
2. Select top N candidates (N = 3-5x budget)
3. Run BatchBALD on these N candidates (joint entropy diversity)
4. Select final batch of size B
```

**Advantages**:
- C-BALD pre-filtering ensures high information value
- BatchBALD's joint entropy provides principled diversity
- No fighting between information value and diversity (sequential)

**Disadvantages**:
- Much slower (requires full BatchBALD computation)
- BatchBALD may still select lower C-BALD scores for diversity
- Risk of overfitting to diversity at cost of information value

### Theoretical Analysis

**Question**: Will BatchBALD's joint entropy beat JS divergence for diversity?

**Joint Entropy Advantage**:
- Accounts for redundancy across entire batch
- Principled mutual information framework
- Can detect higher-order correlations

**JS Divergence Advantage**:
- Simple pairwise metric
- Fast to compute
- Already proven to work well (+0.0180)

**Hypothesis**: BatchBALD's joint entropy is likely overkill for this problem.

**Reasoning**:
1. At budget=20, pairwise diversity is sufficient
2. CBALD-Diverse already beats C-BALD (+0.0180 vs +0.0173)
3. Adding complexity may not yield significant gains
4. Speed trade-off probably not worth marginal improvement

## Alternative Ideas: Better Than CBALD-Diverse?

### Idea 1: CBALD-Diverse with Adaptive Ratio
**Current**: Fixed 70% C-BALD + 30% diversity
**Proposed**: Adapt ratio based on batch progress
- Early selections: 90% C-BALD + 10% diversity (prioritize value)
- Later selections: 50% C-BALD + 50% diversity (more exploration)

**Hypothesis**: Better balance between exploitation and exploration

### Idea 2: C-BALD with Diversity Constraint
**Current**: Greedy selection with combined score
**Proposed**: Hard constraint on minimum diversity
- Select highest C-BALD sample
- For remaining samples: require JS divergence > threshold
- Among samples meeting threshold, pick highest C-BALD

**Hypothesis**: Ensures diversity without sacrificing too much value

### Idea 3: Two-Stage C-BALD
**Proposed**: Exploit then explore
- Stage 1: Select top B/2 samples by C-BALD score (pure exploitation)
- Stage 2: Select remaining B/2 from rest with diversity filtering

**Hypothesis**: Ensures we get the absolute highest-value samples

### Idea 4: Remove Pre-filtering Bias
**Current**: Pre-filter with variance (1961 → 500)
**Proposed**: Remove pre-filtering entirely
- Score all 1961 samples with C-BALD
- Select diverse subset from top candidates

**Hypothesis**: May find better samples not caught by variance filter

## Experimental Plan

### Priority 1: Quick Wins (No BatchBALD)
Test simple modifications to CBALD-Diverse that don't require BatchBALD:

1. **CBALD-Diverse-Adaptive**: Adaptive ratio (90%→50% over selections)
2. **CBALD-Diverse-NoFilter**: Remove variance pre-filtering
3. **CBALD-TwoStage**: Pure C-BALD for top 50%, diversity for rest

**Expected outcome**: Small improvements, fast to test

### Priority 2: BatchBALD Hybrid (If Quick Wins Succeed)
Only if simple approaches show promise:

4. **CBALD-BatchBALD**: Full joint entropy on C-BALD top candidates

**Expected outcome**: Marginal gain, much slower, probably not worth it

### Priority 3: Understanding BatchBALD's Failure
Diagnostic experiments:

5. **BatchBALD at higher budgets** (50, 100): Does diversity help with more samples?
6. **BatchBALD without pre-filtering**: Is variance bias the problem?
7. **Pure diversity baseline**: Random selection with diversity constraint

**Expected outcome**: Understand when/why BatchBALD fails

## Recommendations

### Short Answer: Does BatchBALD Help?
**No, not in current form.**

BatchBALD's diversity mechanism conflicts with information value optimization. All attempts to combine them failed.

### Should We Use BatchBALD's Joint Entropy?
**Probably not.**

CBALD-Diverse's simple JS divergence already beats C-BALD. BatchBALD's complex joint entropy is likely overkill and much slower.

### Best Path Forward

1. **Test simple CBALD-Diverse variants first** (Priority 1 experiments)
   - Adaptive ratio
   - Remove pre-filtering
   - Two-stage selection

2. **Only try BatchBALD hybrid if** simple approaches show promise
   - And we need that extra 0.5-1% improvement
   - And we're willing to accept slower runtime

3. **Focus on understanding** rather than forcing BatchBALD to work
   - Why does it underperform Variance?
   - When does diversity actually help?
   - What's the optimal budget for diversity benefits?

### Key Insight

**The winning strategy is NOT to use BatchBALD's complex mechanism.**

Instead:
1. Start with proven information value (C-BALD)
2. Add simple diversity filtering (JS divergence)
3. Keep it fast and effective

This is what CBALD-Diverse does, and it works (+0.0180 beats +0.0173).

## Conclusion

**BatchBALD doesn't help** because:
1. It lacks information value weighting
2. Its diversity mechanism conflicts with value optimization
3. At budget=20, diversity doesn't provide sufficient benefit
4. Simple alternatives (Variance) perform better

**CBALD-Diverse succeeds** because:
1. It prioritizes information value first (C-BALD)
2. It adds diversity second (JS divergence)
3. It keeps them separate (no conflict)
4. It's simple, fast, and effective

**Next steps**: Test simple variants of CBALD-Diverse before considering complex BatchBALD approaches.
