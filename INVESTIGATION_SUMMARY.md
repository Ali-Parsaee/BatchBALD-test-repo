# Investigation Summary: Why C-BatchBALD Fails and How We Fixed It

**Date:** 2026-01-27
**Status:** Investigation Complete - Ready for Testing

---

## 🔍 Investigation Goals

1. Why does CBALD-Diverse stop beating C-BALD at higher budgets (30, 50)?
2. Why does C-BatchBALD come last at budget=50?
3. How can we improve C-BatchBALD to beat C-BALD?

---

## 📊 The Problems We Discovered

### Problem 1: CBALD-Diverse's Pre-filtering Bottleneck

**Current Results:**
- Budget 20: CBALD-Diverse wins (+0.0180 vs C-BALD +0.0173) ✅
- Budget 30: C-BALD wins (+0.0224 vs CBALD-Diverse +0.0219) ❌
- Budget 50: C-BALD wins (+0.0285 vs CBALD-Diverse +0.0278) ❌

**Root Cause:**
```python
# Line 1065 in acquisition.py
top_k = min(batch_size * 3, len(X_pool))  # Pre-filter to 3x batch size
```

**The Issue:**
- At budget=50, CBALD-Diverse pre-filters to top 150 samples (7.6% of 1961 total)
- Then selects 50 from those 150 using diversity
- **C-BALD takes top 50 globally** → gets THE BEST samples
- **CBALD-Diverse takes diverse 50 from top 150** → dilutes quality with diversity

**Why This Matters:**
- Small budgets: Diversity helps avoid redundancy
- Large budgets: Information value matters more - you want THE BEST samples
- At budget=50, getting THE BEST 50 is better than getting diverse top-150

---

### Problem 2: C-BatchBALD's Fatal Flaw

**Current Results:**
- Budget 50: C-BALD wins (+0.0285)
- Budget 50: C-BatchBALD LAST PLACE (+0.0179) - **37% worse!**

**The Fatal Flaw:**

C-BatchBALD throws away C-BALD's domain knowledge!

**C-BALD Formula (Lines 1857-1893):**
```python
time_variance = expected_times.var(dim=1)     # Epistemic uncertainty
death_prob = avg_probs[i, s:e].sum()          # Death probability in window
cbald_score = time_variance * (0.5 + 0.5 * death_prob)  # DOMAIN KNOWLEDGE!
```

**C-BatchBALD's Mistake (Lines 1514-1534):**
```python
# Step 1: Pre-filter by C-BALD ✅
top_indices = np.argsort(cbald_scores)[-500:]

# Step 2: Compute BALD scores ❌ LOSES DEATH_PROB!
bald_score = H(E[p]) - E[H(p)]  # Generic uncertainty, no death_prob!

# Step 3: Use BALD in selection ❌
combined = 0.6 * bald_normalized + 0.4 * diversity
```

**Why This Breaks Everything:**
1. C-BALD uses **death probability in the increment window** - critical for oracle value!
2. BALD uses **generic uncertainty** - doesn't know about censoring or death windows
3. After pre-filtering, C-BatchBALD **ignores death_prob** and uses generic BALD
4. **Result**: Loses the domain knowledge that makes C-BALD powerful!

**Quantified Impact:**
- C-BALD: +0.0285 (uses death_prob throughout)
- C-BatchBALD: +0.0179 (loses death_prob after pre-filtering)
- **Loss: 0.0106 improvement (-37%)**

---

## 💡 The Solutions We Implemented

### Solution 1: C-BatchBALD-V2 (RECOMMENDED)

**Key Fixes:**
1. ✅ Use C-BALD scores directly (not BALD) → preserves death_prob
2. ✅ Adaptive pre-filter (10x budget, not fixed 500)
3. ✅ Higher C-BALD weight (80/20, not 60/40)

**Implementation (Lines 1606-1735):**
```python
def c_batchbald_v2_acquire(..., cbald_weight=0.8):
    # Step 1: Compute C-BALD scores for ALL samples
    cbald_scores = cbald_score(model, X_pool, ...)

    # Step 2: Adaptive pre-filter (10x budget for diversity pool)
    prefilter_k = min(batch_size * 10, len(X_pool))
    top_indices = np.argsort(cbald_scores)[-prefilter_k:][::-1]

    # Step 3: Get ensemble predictions for diversity
    ensemble_survival = get_ensemble_predictions(...)

    # Step 4: Greedy selection using C-BALD scores + diversity
    for candidate in remaining:
        # Use C-BALD score directly (keeps death_prob!)
        cbald_component = cbald_scores[candidate]

        # Compute diversity
        diversity_score = js_divergence_from_selected(...)

        # Combine: MORE weight on C-BALD
        combined = 0.8 * cbald_component + 0.2 * diversity_score
```

**Why This Works:**
- Uses C-BALD scores throughout → keeps death_prob weighting ✅
- Larger pre-filter (10x budget) → more diversity opportunities ✅
- Higher C-BALD weight → prioritizes information value ✅
- Should match C-BALD on value + beat it on diversity ✅

**Expected Performance:**
- **Target: +0.0290 at budget=50** (vs C-BALD's +0.0285)
- Should beat C-BALD by 1-2% by removing redundancy

---

### Solution 2: C-BatchBALD-Adaptive

**Key Insight:**
- Budget 10: CBALD-Adaptive wins (needs more diversity)
- Budget 50: C-BALD wins (needs more information value)

**Implementation (Lines 1738-1768):**
```python
def c_batchbald_adaptive_acquire(...):
    # Budget-aware diversity ratios
    if batch_size <= 10:
        cbald_weight = 0.7  # 70% C-BALD, 30% diversity
    elif batch_size <= 30:
        cbald_weight = 0.8  # 80% C-BALD, 20% diversity
    else:
        cbald_weight = 0.9  # 90% C-BALD, 10% diversity

    # Use V2 with adaptive weight
    return c_batchbald_v2_acquire(..., cbald_weight=cbald_weight)
```

**Why This Works:**
- Small budgets: More diversity (avoid redundancy in small batches)
- Medium budgets: Balanced (sweet spot like CBALD-Diverse)
- Large budgets: More value (like C-BALD, but with small diversity bonus)

**Expected Performance:**
- Consistent top-3 across ALL budgets
- Budget 10: +0.0110 (match CBALD-Adaptive)
- Budget 20: +0.0185 (beat CBALD-Diverse)
- Budget 30: +0.0225 (match C-BALD)
- Budget 50: +0.0290 (beat C-BALD)

---

## 🎯 C-BALD's Exploitable Limitations

We identified 3 weaknesses in C-BALD that diversity can exploit:

### 1. No Diversity Check
```python
# C-BALD just sorts and takes top N
sorted_indices = torch.argsort(scores, descending=True)
return list(sorted_indices[:batch_size])  # NO diversity!
```

**Issue**: Might select 10 very similar patients with redundant information

### 2. No Batch Coherence
- C-BALD picks samples independently
- Doesn't consider batch-level information gain
- **Opportunity**: Diversity filtering can remove redundancy

### 3. Fixed Strategy Across Budgets
- Same greedy selection for budget=10 and budget=50
- Doesn't adapt to budget size
- **Opportunity**: Adaptive methods can optimize per budget

---

## 📋 Next Steps: Testing Plan

### Test 1: Validate C-BatchBALD-V2 Fixes (PRIORITY 1)

**Goal**: Confirm V2 beats C-BALD at budget=50

```bash
# Update test_budget50.py to include c_batchbald_v2_acquire
python test_budget50.py
```

**Expected Results:**
| Method | Target | vs C-BALD |
|--------|--------|-----------|
| C-BALD | +0.0285 | baseline |
| C-BatchBALD (old) | +0.0179 | -37% ❌ |
| **C-BatchBALD-V2** | **+0.0290** | **+2%** ✅ |

**Success Criteria:**
- ✅ C-BatchBALD-V2 beats C-BALD by 0.0005 or more
- ✅ Uses C-BALD scores (check logs: "using C-BALD + diversity")
- ✅ Adaptive pre-filter (10x budget = 500 samples at budget=50)

---

### Test 2: Validate Adaptive Across All Budgets (PRIORITY 2)

**Goal**: Confirm adaptive ratios help across budgets

```bash
# Update test scripts to include c_batchbald_adaptive_acquire
python test_budget10.py
python test_budget20.py
python test_budget30.py
python test_budget50.py
```

**Expected Results:**
| Budget | Current Winner | C-BatchBALD-Adaptive Target | vs Winner |
|--------|----------------|----------------------------|-----------|
| 10 | CBALD-Adaptive +0.0112 | +0.0115 | +3% ✅ |
| 20 | CBALD-Diverse +0.0180 | +0.0185 | +3% ✅ |
| 30 | C-BALD +0.0224 | +0.0225 | +0.4% ✅ |
| 50 | C-BALD +0.0285 | +0.0290 | +2% ✅ |

**Success Criteria:**
- ✅ C-BatchBALD-Adaptive in top-2 at ALL budgets
- ✅ Beats current winner at 2+ budgets
- ✅ Most consistent method overall

---

### Test 3: Sensitivity Analysis (PRIORITY 3)

**Goal**: Optimize hyperparameters

```bash
# Test different cbald_weight values
python test_sensitivity.py --weights 0.7,0.75,0.8,0.85,0.9
python test_sensitivity.py --prefilter_multipliers 5,7,10,15,20
```

**Parameters to Test:**
1. `cbald_weight`: Test 0.7, 0.75, 0.8, 0.85, 0.9
2. `prefilter_multiplier`: Test 5x, 7x, 10x, 15x, 20x budget

**Expected Insights:**
- Optimal `cbald_weight` likely 0.8-0.85 for budget=50
- Optimal `prefilter_multiplier` likely 10x-15x budget

---

## 🎓 Key Insights Summary

### Why Original C-BatchBALD Failed
1. **Threw away domain knowledge**: Used BALD instead of C-BALD scores
2. **Lost death_prob weighting**: BALD doesn't know about death windows
3. **Wrong ratio**: 60/40 wasn't aggressive enough on information value
4. **Fixed pre-filter**: 500 samples regardless of budget size

### How V2 Fixes It
1. **Keeps domain knowledge**: Uses C-BALD scores directly
2. **Preserves death_prob**: death_prob * time_variance used in selection
3. **Better ratio**: 80/20 prioritizes proven C-BALD scoring
4. **Adaptive pre-filter**: 10x budget gives better diversity pool

### Why We Expect V2 to Win
1. **Matches C-BALD on information value** (same scoring function)
2. **Beats C-BALD on diversity** (removes redundant samples)
3. **Exploits C-BALD's weakness** (no diversity checking)
4. **Target: +1-2% improvement** at budget=50

---

## 📁 Files Modified

1. **Model_stuff/acquisition.py**:
   - Added `c_batchbald_v2_acquire()` (lines 1606-1735)
   - Added `c_batchbald_adaptive_acquire()` (lines 1738-1768)

2. **C_BATCHBALD_INVESTIGATION.md**:
   - Complete technical analysis
   - Root cause identification
   - Solution proposals with code

3. **INVESTIGATION_SUMMARY.md** (this file):
   - Executive summary
   - Testing plan
   - Expected results

---

## ✅ Ready for Testing

All code has been implemented. Next step: Update test scripts and run experiments!

**Recommendation**: Start with test_budget50.py to validate the core fix before testing across all budgets.
