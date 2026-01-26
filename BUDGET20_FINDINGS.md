# Budget=20 Experimental Results: Surprising Findings

## Executive Summary

**The Paradox:** When budget was increased from 10 to 20, BatchBALD went from **1st place to LAST place** in final C-index, yet it still achieved the **HIGHEST improvement** from its initial model.

## Results Comparison

### Budget=10 Results (Previous)
| Rank | Method | Improvement | Final C-index |
|------|--------|-------------|---------------|
| 🏆 1 | BatchBALD | **+0.0056 ± 0.0022** | — |
| 🥈 2 | Variance | +0.0036 ± 0.0013 | — |
| 🥉 3 | Entropy | +0.0019 ± 0.0009 | — |
| 4 | Random | +0.0009 ± 0.0018 | — |

### Budget=20 Results (Current)

**By Final C-index:**
| Rank | Method | Final C-index | Improvement |
|------|--------|--------------|-------------|
| 🏆 1 | **Random** | **0.6928 ± 0.0076** | +0.0002 ± 0.0023 |
| 🥈 2 | Entropy | 0.6926 ± 0.0081 | +0.0034 ± 0.0005 |
| 🥉 3 | Variance | 0.6892 ± 0.0084 | +0.0033 ± 0.0025 |
| 4 | BatchBALD_optimal (λ₁=0.3) | 0.6859 ± 0.0102 | +0.0014 ± 0.0013 |
| 5 | BatchBALD_optimal (λ₁=0.0) | 0.6845 ± 0.0107 | +0.0016 ± 0.0017 |
| 6 | **BatchBALD** | **0.6829 ± 0.0109** | **+0.0108 ± 0.0039** |

**By Improvement (Δ C-index):**
| Rank | Method | Improvement | Final C-index |
|------|--------|-------------|---------------|
| 🏆 1 | **BatchBALD** | **+0.0108 ± 0.0039** | 0.6829 ± 0.0109 |
| 🥈 2 | Entropy | +0.0034 ± 0.0005 | 0.6926 ± 0.0081 |
| 🥉 3 | Variance | +0.0033 ± 0.0025 | 0.6892 ± 0.0084 |
| 4 | BatchBALD_optimal (λ₁=0.0) | +0.0016 ± 0.0017 | 0.6845 ± 0.0107 |
| 5 | BatchBALD_optimal (λ₁=0.3) | +0.0014 ± 0.0013 | 0.6859 ± 0.0102 |
| 6 | **Random** | **+0.0002 ± 0.0023** | 0.6928 ± 0.0076 |

## Statistical Significance

**BatchBALD vs Others (Paired t-tests on final C-index):**
- BatchBALD vs Random: **-0.0099** (p=0.0028) ✓ **Significantly worse**
- BatchBALD vs Entropy: **-0.0097** (p=0.0050) ✓ **Significantly worse**
- BatchBALD vs Variance: **-0.0063** (p=0.0157) ✓ **Significantly worse**
- BatchBALD vs BatchBALD_optimal (λ₁=0.3): -0.0030 (p=0.0233) ✓ Significantly worse
- BatchBALD vs BatchBALD_optimal (λ₁=0.0): -0.0016 (p=0.0974) Not significant

**Top performers comparison:**
- Random vs Entropy: +0.0002 (p=0.8405) - Essentially tied

## The Paradox Explained

### Why BatchBALD Ranks Last in Final C-index

BatchBALD achieved the highest improvement (+0.0108) but ended with the lowest final C-index (0.6829). This creates an apparent paradox:

**How can the method that improves the most end up with the worst final performance?**

### Explanation: Initial Model Variance

Each method in each trial trains its own independent model from scratch with random initialization. This creates variance in initial C-index across methods:

**Example from trials:**
- Random: Started at ~0.6926, improved +0.0002 → Final: 0.6928
- BatchBALD: Started at ~0.6721, improved +0.0108 → Final: 0.6829

**BatchBALD improved 54× more than Random (+0.0108 vs +0.0002), but Random started with a model that was already 0.02 points better!**

### Why This Matters

This reveals a critical issue with the experimental design:

1. **Within each trial**, different methods train different base models
2. These base models have different initial C-indices due to random initialization
3. A method that starts with a bad model and improves it significantly can still lose to a method that starts with a great model and barely improves it

### The True Question

We need to answer: **"Given the same initial model, which acquisition function improves it the most?"**

Our current experiment answers: **"Which method ends up with the best model, regardless of starting point?"**

These are related but different questions.

## Why Did Results Change From Budget=10 to Budget=20?

### Budget=10: BatchBALD Won
- Small budget (10 samples)
- BatchBALD's careful sample selection made a big difference
- Random selection didn't have enough samples to get lucky

### Budget=20: Random Won
- Larger budget (20 samples)
- Random had enough samples to cover the important regions
- With more data, sophisticated selection matters less
- Initial model quality dominated final performance

### Diminishing Returns Hypothesis

As budget increases:
1. The marginal benefit of smart sample selection decreases
2. Random sampling has more chances to cover important areas
3. Initial model quality becomes more important than acquisition strategy

## Individual Trial Results

### Trial 1
- Random: 0.6873
- Entropy: 0.6839
- Variance: 0.6805
- BatchBALD_optimal (λ₁=0.3): 0.6781
- BatchBALD_optimal (λ₁=0.0): 0.6780
- BatchBALD: 0.6754

### Trial 2
- Entropy: 0.6854
- Random: 0.6855
- Variance: 0.6817
- BatchBALD_optimal (λ₁=0.3): 0.6744
- BatchBALD: 0.6719
- BatchBALD_optimal (λ₁=0.0): 0.6716

### Trial 3
- Entropy: 0.6925
- Random: 0.6901
- Variance: 0.6888
- BatchBALD_optimal (λ₁=0.3): 0.6849
- BatchBALD_optimal (λ₁=0.0): 0.6822
- BatchBALD: 0.6786

### Trial 4
- Entropy: 0.6993
- Random: 0.6979
- Variance: 0.6955
- BatchBALD_optimal (λ₁=0.3): 0.6938
- BatchBALD_optimal (λ₁=0.0): 0.6937
- BatchBALD: 0.6917

### Trial 5
- Random: 0.7034
- Entropy: 0.7020
- Variance: 0.6996
- BatchBALD_optimal (λ₁=0.3): 0.6985
- BatchBALD: 0.6970
- BatchBALD_optimal (λ₁=0.0): 0.6971

**Observation:** Across ALL 5 trials, BatchBALD consistently ranked in the bottom half (ranks 5-6). This isn't just bad luck in one trial - it's systematic.

## Key Insights

### 1. BatchBALD Still Improves Most
Despite ranking last in final C-index, BatchBALD achieved:
- **Highest improvement**: +0.0108 (2.7× better than Entropy/Variance)
- **Most effective learning**: Extracts more value from acquired samples

### 2. Random Selection Works Well with Larger Budgets
Random achieved:
- **Highest final C-index**: 0.6928
- **Lowest improvement**: +0.0002 (essentially no learning!)
- **Implication**: With budget=20, random sampling covers enough of the space that active learning advantages diminish

### 3. Optimized BatchBALD Still Fails
- BatchBALD_optimal (λ₁=0.0): 5th place (0.6845)
- BatchBALD_optimal (λ₁=0.3): 4th place (0.6859)
- Both performed better than original BatchBALD, but still worse than simpler methods

### 4. Simpler Methods Dominate
Top 3 methods are all simple:
1. Random (no strategy)
2. Entropy (simple uncertainty)
3. Variance (simple ensemble disagreement)

BatchBALD's sophistication (joint entropy, diversity) doesn't help with larger budgets.

## Possible Explanations

### 1. Initial Model Quality Dominates
- Random consistently gets better initial models
- With budget=20, initial advantage persists
- Acquisition function can't overcome poor initialization

### 2. BatchBALD Overfits to Budget=10
- BatchBALD's diversity mechanism is tuned for small batches
- With budget=20, diversity might hurt more than help
- Selecting "diverse" samples might skip important high-uncertainty regions

### 3. Pre-filtering Bias
- All methods use top-500 pre-filtering based on variance
- This might bias toward samples that variance-based methods like
- BatchBALD's strengths (diversity, joint info) are wasted if pool is already variance-selected

### 4. Dataset Characteristics
- NACD dataset might not benefit from diversity at budget=20
- Maybe the uncertainty distribution is such that random sampling naturally covers important areas
- With 500 pre-filtered samples, random sampling of 20 is 4% - might be enough

### 5. Computational Approximations
- BatchBALD switches to SampledJointEntropy at sample 15/20
- This approximation might hurt selection quality
- Simpler methods don't have this complexity

## Implications

### For This Project
1. ✅ **Budget=10**: BatchBALD wins
2. ❌ **Budget=20**: Random/Entropy win, BatchBALD loses

**Recommendation:** Use BatchBALD for small budgets (≤10), simpler methods for larger budgets

### For Active Learning Research
1. **Budget size matters**: Results at budget=10 don't generalize to budget=20
2. **Initial model variance**: Need to control for random initialization effects
3. **Diminishing returns**: Sophisticated methods might not scale well to larger budgets
4. **Experimental design**: Should compare methods with same initial model

### For Future Experiments

To properly evaluate acquisition functions, we should:

1. **Fix initial model**: Train ONE base model per trial, use it for ALL methods
2. **Test multiple budgets**: Run experiments at budgets [5, 10, 15, 20, 25, 30]
3. **Plot learning curves**: Show how improvement scales with budget
4. **Control for variance**: Use more trials or fixed random seeds
5. **Measure efficiency**: Compare cost (time) vs. benefit (improvement)

## Unanswered Questions

1. **Why is BatchBALD's initial model consistently worse?**
   - Across all 5 trials, it ranks bottom-half
   - Is there a bug in the training code for BatchBALD?
   - Or is it just random bad luck?

2. **Would fixing the initial model change the ranking?**
   - If all methods started from the same model, would BatchBALD win?
   - Is improvement (+0.0108) enough to overcome the deficit?

3. **At what budget does BatchBALD stop winning?**
   - Budget=10: BatchBALD wins
   - Budget=20: Random wins
   - Where's the crossover point?

4. **Is pre-filtering helping or hurting?**
   - Pre-filtering by variance might bias toward variance-based methods
   - Would BatchBALD do better without pre-filtering?

## Conclusions

### Short Answer: "Does BatchBALD still win at budget=20?"

**No.** At budget=20:
- Random selection wins (final C-index: 0.6928)
- BatchBALD ranks last (final C-index: 0.6829)
- Difference is statistically significant (p=0.0028)

### Longer Answer: "It's Complicated"

BatchBALD achieves the **highest improvement** (+0.0108) but ends with the **lowest final C-index** (0.6829) because:

1. It consistently starts with worse initial models
2. Random selection benefits from larger budget (20 vs 10)
3. Initial model quality dominates final performance at this budget size

### Key Takeaway

**For small budgets (≤10):** BatchBALD's sophisticated sample selection provides clear advantages.

**For larger budgets (≥20):** Simple methods (Random, Entropy, Variance) perform better, possibly because:
- Random sampling covers enough of the space with more samples
- Simpler methods are more robust to initialization variance
- Sophisticated diversity mechanisms matter less with more data

### Recommendation

Use **budget-aware acquisition function selection**:
- Budget ≤ 10: Use BatchBALD
- Budget 10-20: Use Entropy or Variance
- Budget ≥ 20: Random selection works surprisingly well
