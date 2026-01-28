# C-BALD Formulation Comparison

**Date:** 2026-01-27
**Status:** Running experiments
**Goal:** Compare three C-BALD formulations to determine which is best

---

## 🎯 The Question

**Which C-BALD formulation performs best?**

We have three different implementations:
1. **C-BALD (Simplified)** - Our current winner
2. **BatchBALD (Pure)** - Standard BatchBALD
3. **CBALD_True (Paper-correct)** - Theoretically correct from paper

---

## 📊 Three Formulations

### 1. C-BALD (Simplified) ✅ Current Winner

**Formula:**
```python
score = time_variance * (0.5 + 0.5 * death_prob)
```

**How it works:**
- `time_variance`: Epistemic uncertainty in expected survival time
- `death_prob`: Probability of death in the increment window
- Simple multiplicative combination

**Known Performance:**
- Budget 30: +0.0224 (1st place)
- Budget 50: +0.0285 (1st place)
- **Wins at high budgets**

**Advantages:**
- Simple, interpretable formula
- Fast to compute
- Proven winner

**Limitations:**
- Not theoretically derived
- Heuristic combination of components

---

### 2. BatchBALD (Pure) ❌ Known to Underperform

**Formula:**
```python
# Joint entropy maximization
score = I(y_batch; θ | x_batch)
```

**How it works:**
- Maximizes joint mutual information over the batch
- Uses BALD scores with diversity
- No censoring-specific components

**Known Performance:**
- Budget 20: +0.0109 (WORSE than Variance at +0.0117)
- **Loses to simple baselines**

**Why it fails:**
- Doesn't use death probability weighting
- Doesn't account for censoring structure
- Generic uncertainty, not domain-specific

---

### 3. CBALD_True (Paper-correct) ❓ Unknown Performance

**Formula:**
```python
score = I(l; θ | x) + I(y; θ | l, x)
```

**Where:**
- `I(l; θ | x)`: Mutual information about censoring indicator
  - l=1 means uncensored (death observed)
  - l=0 means censored
- `I(y; θ | l, x)`: Mutual information about outcome, conditioned on censoring
  - Blends uncensored and censored MI

**How it works:**
1. Compute λ_ens: P(uncensored | x, θ) for each ensemble member
2. Compute I(l; θ | x) = H(E[λ]) - E[H(λ)] (censoring indicator MI)
3. Compute I_y_unc: Standard BALD for uncensored labels
4. Compute I_y_cens: MI for censored observations (survival mass)
5. Blend: I(y; θ | l, x) = λ̄ * I_y_unc + (1-λ̄) * I_y_cens
6. Total: C-BALD = I(l; θ | x) + I(y; θ | l, x)

**Known Performance:**
- ❓ Never tested before
- This is the first time we're running it

**Advantages:**
- Theoretically correct from C-BALD paper
- Properly decomposes mutual information
- Accounts for censoring structure explicitly

**Potential Issues:**
- More complex to compute
- May overweight censoring indicator
- Unknown if theory beats practice

---

## 🔬 Experiment Design

### Test Configuration

**Budgets:** 10, 20, 30
**Trials:** 5 per method per budget
**Metric:** C-index improvement over initial model

### Why These Budgets?

- **Budget 10:** Small budget - tests early stage performance
- **Budget 20:** Medium budget - where CBALD-Diverse beat C-BALD (+0.0180 vs +0.0173)
- **Budget 30:** Large budget - where C-BALD wins (+0.0224)

### Hypotheses

**H1:** C-BALD (simplified) beats BatchBALD at all budgets
- **Rationale:** C-BALD uses death_prob, BatchBALD doesn't
- **Known:** C-BALD beats BatchBALD at budget=20

**H2:** CBALD_True performs similarly to C-BALD (simplified)
- **Rationale:** Both capture censoring structure
- **Alternative:** Paper-correct might beat simplified

**H3:** BatchBALD comes last at all budgets
- **Rationale:** No domain knowledge, proven to underperform
- **Known:** BatchBALD +0.0109 vs Variance +0.0117

---

## 📋 Expected Outcomes

### Scenario A: Simplified C-BALD Wins (Most Likely)
```
Budget 10:  C-BALD > CBALD_True > BatchBALD
Budget 20:  C-BALD > CBALD_True > BatchBALD
Budget 30:  C-BALD > CBALD_True > BatchBALD
```

**Conclusion:** Simple heuristic beats theory
**Action:** Keep using simplified C-BALD

---

### Scenario B: CBALD_True Wins
```
Budget 10:  CBALD_True > C-BALD > BatchBALD
Budget 20:  CBALD_True > C-BALD > BatchBALD
Budget 30:  CBALD_True > C-BALD > BatchBALD
```

**Conclusion:** Theory wins over heuristic!
**Action:** Replace C-BALD with CBALD_True everywhere

---

### Scenario C: Mixed Results
```
Budget 10:  CBALD_True > C-BALD > BatchBALD
Budget 20:  C-BALD > CBALD_True > BatchBALD
Budget 30:  C-BALD > CBALD_True > BatchBALD
```

**Conclusion:** Budget-dependent performance
**Action:** Use different formulations for different budgets

---

## 🎯 Key Questions to Answer

1. **Does CBALD_True beat simplified C-BALD?**
   - If yes: Theory > Practice, adopt CBALD_True
   - If no: Heuristic is good enough

2. **How much does C-BALD beat BatchBALD by?**
   - Quantify the advantage of domain knowledge
   - Confirm BatchBALD is not suitable for this problem

3. **Is the gap consistent across budgets?**
   - Does C-BALD's advantage grow/shrink with budget?
   - Are there budgets where BatchBALD is competitive?

4. **Should we combine CBALD_True with diversity?**
   - If CBALD_True is good, try CBALD_True-Diverse
   - Could be the ultimate method

---

## 📁 Files

**Experiment Script:**
- `compare_cbald_variants.py` - Main comparison script

**Implementation Files:**
- `Model_stuff/acquisition.py` - All three methods:
  - Lines 2009-2053: `cbald_censored_regression()` - C-BALD simplified
  - Lines 900-994: `batchbald_acquire_budget()` - BatchBALD
  - Lines 1759-1935: `cbald_true_acquire()` - CBALD_True

**Results:**
- `compare_results.txt` - Full experiment output
- This document will be updated with findings

---

## ⏳ Status

**Currently Running:**
- Budget 10: 3 methods × 5 trials = 15 runs
- Budget 20: 3 methods × 5 trials = 15 runs
- Budget 30: 3 methods × 5 trials = 15 runs
- **Total: 45 trials**

Estimated time: ~2-3 hours

---

## 🎓 Why This Matters

This comparison will answer the fundamental question: **Does theoretical correctness matter for active learning in survival analysis?**

- If **CBALD_True wins**: Theory guides us to better methods
- If **C-BALD wins**: Practical heuristics can outperform theory
- Either way: We confirm BatchBALD is not suitable for this domain

The answer will determine our final recommendation for the best active learning method for censored survival data.
