# Analysis: What Matters for Survival Active Learning with Probe Depth

## Setting Recap
- One-shot active learning (not iterative)
- Probe depth k limits oracle queries
- Oracle can only reveal up to k bins ahead
- Success metric: C-index improvement after one query round

## Key Factors for Success

### 1. **Informativeness about Model Uncertainty** (CRITICAL)
- **Why**: In Bayesian AL, we want to reduce model uncertainty
- **How BatchBALD helps**: I(Y_oracle; θ) directly measures this
- **Why better than entropy**: Entropy only measures prediction uncertainty, not parameter uncertainty
- **Advantage**: BatchBALD accounts for what THE MODEL is uncertain about, not just what the prediction is uncertain about

### 2. **Likelihood of Oracle Revealing True Death** (IMPORTANT)
- **Why**: Only revealed deaths provide strong signals; extended censoring is weaker
- **Problem**: If we query someone censored at bin 3 with true death at bin 20 and k=3, we only learn they're censored at bin 6 (weak signal)
- **Opportunity**: If we query someone censored at bin 3 with true death at bin 5 and k=3, we learn exact death time (strong signal)
- **How to leverage**:
  - Prefer querying instances where P(death within probe window | censored) is HIGH
  - This means: P(c+1 ≤ T ≤ c+k | T > c) should be high

### 3. **Diversity** (SOMEWHAT IMPORTANT for batches)
- **Why**: In batch selection, redundant queries waste budget
- **How BatchBALD helps**: Joint entropy approximation reduces redundancy
- **Less critical**: In one-shot setting with small batches, less critical than iterative AL

### 4. **Earlier vs Later Censored Instances** (CONTEXT-DEPENDENT)
- **Earlier censored (low c)**:
  - Pros: More room to learn (can reveal deaths in many bins)
  - Cons: If true death is very late, oracle won't help much
- **Later censored (high c)**:
  - Pros: Already have more information
  - Cons: Less room for oracle to add value
- **Verdict**: NEITHER is universally better; depends on where model is most uncertain

### 5. **Instance Density/Representativeness** (LESS IMPORTANT)
- Not critical for one-shot AL with Bayesian models
- BatchBALD already captures epistemic uncertainty

## Why Entropy/Variance Might Be Doing Well

Looking at your existing results, entropy and variance are performing well because:

1. **High entropy ≈ High P(death in probe window)**
   - If a censored instance has high entropy, the model is spreading probability across many bins
   - Some of that probability mass is likely within the probe window
   - So entropy is indirectly capturing "likelihood of revealing death"

2. **Variance captures disagreement**
   - High variance means ensemble members disagree
   - This often correlates with instances where oracle could disambiguate

3. **Simple and robust**
   - No complex calculations that could go wrong
   - Direct measure of uncertainty

## How to Make BatchBALD Dominate

BatchBALD should win by combining:

### **A. Mutual Information** (captures epistemic uncertainty)
Rather than just H(Y_oracle), compute I(Y_oracle; θ)

### **B. Expected Informativeness**
Weight by P(oracle reveals death) because revealed deaths provide stronger training signal than extended censoring

Proposed score:
```
score = I(Y_oracle; θ) × P(reveal death within window)

where:
P(reveal death) = Σ_{t=c+1}^{c+k} p_t  (sum of death probs in probe window)
```

### **C. Proper Batch Selection**
Use greedy BatchBALD to avoid redundancy

### **D. Censoring-Aware Renormalization** (already done)
Continue renormalizing based on current censoring time

## Hypothesis

**Entropy/Variance are doing well because they correlate with "likely to reveal death"**

**BatchBALD can dominate by EXPLICITLY combining:**
1. Model uncertainty (mutual information)
2. Likelihood of strong signal (P(reveal death))
3. Batch diversity (greedy selection)

This triple combination should beat single-factor methods.

## Next Steps

1. Implement "weighted BatchBALD" with P(reveal death) multiplier
2. Compare against entropy, variance, and unweighted BatchBALD
3. Analyze which component contributes most to performance
4. Test on NACD dataset
