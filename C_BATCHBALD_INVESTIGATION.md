# C-BatchBALD Investigation: Why It Underperforms and How to Fix It

**Date:** 2026-01-27
**Goal:** Understand why C-BatchBALD underperforms and improve it to beat C-BALD

---

## 📊 Current Performance Analysis

### Budget 50 Results (The Problem)
| Rank | Method | Improvement | vs C-BALD |
|------|--------|-------------|-----------|
| 🏆 1 | C-BALD | **+0.0285** | baseline |
| 2 | CBALD-TwoStage | +0.0279 | -0.0006 |
| 3 | CBALD-Diverse | +0.0278 | -0.0007 |
| 4 | CBALD-NoFilter | +0.0278 | -0.0007 |
| 5 | CBALD-Adaptive | +0.0265 | -0.0020 |
| ❌ 6 | **C-BatchBALD** | **+0.0179** | **-0.0106** ⚠️ |

**C-BatchBALD loses by 37%!** This is a massive gap.

### Budget 30 Results
| Method | Improvement | vs C-BALD |
|--------|-------------|-----------|
| C-BALD | **+0.0224** | baseline |
| CBALD-Diverse | +0.0219 | -0.0005 (only 2% worse) |

CBALD-Diverse is VERY close at budget=30!

---

## 🔍 Problem 1: Why Does CBALD-Diverse Stop Winning at Higher Budgets?

### The Pre-filtering Bottleneck

**CBALD-Diverse Strategy:**
```python
# Line 1065 in acquisition.py
top_k = min(batch_size * 3, len(X_pool))  # Pre-filter to 3x batch size
top_indices = np.argsort(cbald_scores)[-top_k:][::-1]
```

**The Issue:**
- Budget 10: Top 30 samples → Select 10 → **33% selection rate**
- Budget 20: Top 60 samples → Select 20 → **33% selection rate**
- Budget 30: Top 90 samples → Select 30 → **33% selection rate**
- Budget 50: Top 150 samples → Select 50 → **33% selection rate**

**Why This Hurts at High Budgets:**
1. At budget=50, we're selecting 50 samples from ONLY top 150 candidates
2. With 1961 samples in pool, top 150 is only **7.6%** of data
3. The remaining 43 samples (after selecting #1-7) are ALL from a very narrow high-value slice
4. **Limited diversity opportunities** - all candidates are very similar in C-BALD score
5. C-BALD can pick from top 50 globally → gets THE BEST samples
6. CBALD-Diverse picks from top 150 → dilutes top samples with diversity filtering

### Why C-BALD Wins at High Budgets

**C-BALD Strategy:**
```python
# Just take top N by C-BALD score
sorted_indices = np.argsort(scores)[-batch_size:][::-1]
```

**At budget=50:**
- C-BALD: Take top 50 samples by pure information value
- CBALD-Diverse: Take top 7 + diverse 43 from top 150
- **C-BALD gets THE BEST samples** → higher performance

**Diminishing Returns of Diversity:**
- At small budgets (10-20): Diversity helps avoid redundancy in small batches
- At large budgets (30-50): Information value matters more - you want THE BEST samples
- Diversity becomes less important when batch is large enough to naturally cover variety

---

## 🔍 Problem 2: Why Does C-BatchBALD Perform So Poorly?

### Current C-BatchBALD Implementation

```python
# Line 1472-1534: C-BatchBALD strategy
# Step 1: Pre-filter by C-BALD (top 500)
top_cbald_indices = np.argsort(cbald_scores)[-500:][::-1]

# Step 2: Compute BALD scores on filtered samples
# BALD = H(E[p]) - E[H(p)]  ← GENERIC uncertainty!
bald_scores[i] = h_avg - expected_h

# Step 3: Greedy selection
combined_score = 0.6 * bald_component + 0.4 * diversity_score
```

### The Fatal Flaw: BALD ≠ C-BALD

**C-BALD Score (Lines 1857-1893):**
```python
# Computes DOMAIN-SPECIFIC information value
time_variance = expected_times.var(dim=1)  # Epistemic uncertainty
death_prob = avg_probs[i, s:e].sum()       # Death probability in window
cbald_score = time_variance * (0.5 + 0.5 * death_prob)
```

**BALD Score in C-BatchBALD (Lines 1514-1534):**
```python
# Computes GENERIC uncertainty (no death_prob, no time weighting!)
h_avg = -sum(avg_prob * log(avg_prob))      # Entropy of average
expected_h = mean(-sum(p * log(p)))         # Expected entropy
bald_score = h_avg - expected_h             # Mutual information
```

### Why This Breaks C-BatchBALD

**The Problem:**
1. C-BALD uses **domain knowledge**: `time_variance * (0.5 + 0.5 * death_prob)`
2. BALD uses **generic uncertainty**: `H(E[p]) - E[H(p)]`
3. BALD doesn't know about:
   - Death probability in the increment window (critical for oracle value!)
   - Time variance weighting
   - Censoring structure
4. **C-BatchBALD throws away C-BALD's domain knowledge after pre-filtering!**

**Analogy:**
- C-BALD: "Select samples that are uncertain AND likely to die in our probe window"
- C-BatchBALD: "Pre-filter by above... then ignore it and use generic uncertainty"
- **Result**: Loses the death_prob component that makes C-BALD powerful!

### Quantifying the Loss

From budget 50 results:
- C-BALD: +0.0285 (uses time_variance * death_prob throughout)
- C-BatchBALD: +0.0179 (-37% worse!)
- **Lost 0.0106 improvement by switching from C-BALD scores to BALD scores**

---

## 🎯 C-BALD's Exploitable Limitations

### Limitation 1: No Diversity - Redundant Samples
**Evidence from code:**
```python
# Line 1908-1925: C-BALD just sorts and takes top N
sorted_indices = torch.argsort(scores_t, descending=True)
return list(sorted_indices[:batch_size])  # NO diversity check!
```

**The Issue:**
- At budget=50, C-BALD might select 10 very similar patients
- All with high variance + high death probability
- But overlapping survival patterns → redundant information
- **Opportunity**: Diversity could help IF we don't sacrifice information value

### Limitation 2: Greedy Selection - No Batch Coherence
- C-BALD picks samples independently (no joint optimization)
- Doesn't consider batch-level information gain
- **Opportunity**: BatchBALD's joint entropy SHOULD help... if done right

### Limitation 3: Fixed Pre-filter Size
- CBALD-Diverse uses `3 * batch_size` pre-filter
- C-BatchBALD uses fixed `500` pre-filter
- **At budget=50**: 500 pre-filter is 10x budget (good!)
- **At budget=10**: 500 pre-filter is 50x budget (overkill!)
- **Opportunity**: Adaptive pre-filter size based on budget

---

## 💡 Proposed Solutions

### Solution 1: Use C-BALD Scores Directly (HIGHEST PRIORITY)

**Problem**: C-BatchBALD computes BALD after pre-filtering, losing death_prob
**Fix**: Use C-BALD scores directly in greedy selection

```python
def c_batchbald_v2_acquire(...):
    # Step 1: Compute C-BALD scores for ALL samples
    cbald_scores = cbald_score(model, X_pool, ...)

    # Step 2: Adaptive pre-filter (larger pool for diversity)
    prefilter_k = min(batch_size * 10, len(X_pool))  # 10x budget
    top_indices = np.argsort(cbald_scores)[-prefilter_k:][::-1]

    # Step 3: Get probability distributions for diversity
    ensemble_survival = get_ensemble_predictions(model, X_pool[top_indices])

    # Step 4: Greedy selection with C-BALD scores + diversity
    for iter in range(batch_size):
        for candidate_idx in remaining:
            # Use C-BALD score directly (not BALD!)
            cbald_component = cbald_scores[top_indices[candidate_idx]]

            # Diversity: JS divergence from selected samples
            diversity_score = compute_js_divergence(...)

            # Combine: HIGHER weight on C-BALD (it's proven!)
            combined = 0.8 * cbald_normalized + 0.2 * diversity_score
```

**Key Changes:**
1. ✅ Use C-BALD scores (keep death_prob weighting!)
2. ✅ Larger pre-filter (10x budget instead of fixed 500)
3. ✅ More weight on C-BALD (80/20 instead of 60/40)
4. ✅ Keeps diversity to beat C-BALD's redundancy

**Expected Performance:**
- Should match C-BALD on information value (same scoring function)
- Should beat C-BALD via diversity (removes redundancy)
- Target: **+0.0290** at budget=50 (vs C-BALD's +0.0285)

---

### Solution 2: Budget-Adaptive Diversity Ratio

**Insight from Results:**
- Budget 10: CBALD-Adaptive wins (needs more diversity early)
- Budget 50: C-BALD wins (needs less diversity late)

**Implementation:**
```python
def c_batchbald_adaptive(...):
    # Step 1-3: Same as v2

    # Step 4: Adaptive diversity ratio based on budget
    if batch_size <= 10:
        cbald_weight = 0.7  # 70% C-BALD, 30% diversity
    elif batch_size <= 30:
        cbald_weight = 0.8  # 80% C-BALD, 20% diversity
    else:
        cbald_weight = 0.9  # 90% C-BALD, 10% diversity

    diversity_weight = 1.0 - cbald_weight
    combined = cbald_weight * cbald_normalized + diversity_weight * diversity_score
```

---

### Solution 3: Two-Stage C-BatchBALD

**Insight**: CBALD-TwoStage is #2 at budget=50 (+0.0279)

**Implementation:**
```python
def c_batchbald_twostage(...):
    # Stage 1: Pure C-BALD for first 60% (guaranteed best samples)
    stage1_size = int(batch_size * 0.6)
    top_indices = np.argsort(cbald_scores)[-stage1_size:][::-1]
    selected_indices = list(top_indices)

    # Stage 2: Diversity-filtered C-BALD for remaining 40%
    remaining_pool = [i for i in range(len(X_pool)) if i not in selected_indices]
    prefilter_k = min(stage2_size * 5, len(remaining_pool))

    # Apply diversity filtering to top candidates
    stage2_indices = diversity_greedy_selection(
        cbald_scores[remaining_pool],
        ensemble_probs[remaining_pool],
        selected_probs=ensemble_probs[selected_indices],
        batch_size=stage2_size
    )

    return selected_indices + stage2_indices
```

**Why This Works:**
- First 60%: Gets THE BEST samples (like C-BALD)
- Last 40%: Adds diversity to avoid redundancy
- Balances information value + diversity optimally

---

## 📋 Recommended Testing Plan

### Priority 1: Test C-BatchBALD-V2 (Use C-BALD Scores)
```bash
python test_budget50.py  # With c_batchbald_v2_acquire
```
**Expected:** +0.0285 to +0.0295 (match or beat C-BALD)

### Priority 2: Test Budget-Adaptive Version
```bash
python test_all_budgets.py  # Budgets 10, 20, 30, 50 with adaptive ratios
```
**Expected:** Consistent top-3 across all budgets

### Priority 3: Test Two-Stage Version
```bash
python test_budget50.py  # With c_batchbald_twostage
```
**Expected:** +0.0280 to +0.0290 (similar to CBALD-TwoStage)

---

## 🎓 Summary

### Why CBALD-Diverse Stops Winning
1. **Pre-filter bottleneck**: Only 3x budget limits high-budget diversity
2. **Diminishing returns**: Diversity less important at large budgets
3. **C-BALD advantage**: Takes THE BEST samples without diversity dilution

### Why C-BatchBALD Fails
1. **Fatal flaw**: Uses BALD (generic) instead of C-BALD scores (domain-specific)
2. **Loses death_prob**: BALD doesn't know about death probability in window
3. **Wrong ratio**: 60/40 isn't aggressive enough on information value

### How to Fix C-BatchBALD
1. ✅ **Use C-BALD scores directly** in greedy selection (not BALD!)
2. ✅ **Larger pre-filter**: 10x budget (not fixed 500)
3. ✅ **Higher C-BALD weight**: 80/20 or 90/10 (not 60/40)
4. ✅ **Adaptive ratios**: More diversity at small budgets, less at large budgets

### Expected Impact
- C-BatchBALD-V2 should **beat C-BALD** by 1-2% at budget=50
- Target: **+0.0290** vs C-BALD's +0.0285
- Achieves project goal of "making BatchBALD work" by properly integrating diversity with C-BALD's domain knowledge
