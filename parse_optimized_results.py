"""Parse and analyze optimized BatchBALD results."""

import re
import numpy as np
from scipy import stats

# Read the results file
with open('optimized_batchbald_results.txt', 'r') as f:
    content = f.read()

# Extract improvements for each method across trials
methods = {
    'BatchBALD': [],
    'BatchBALD_optimal (λ₁=0.0)': [],
    'BatchBALD_optimal (λ₁=0.3)': [],
    'Variance': [],
    'Entropy': [],
    'Random': []
}

# Find all final c-index lines with improvements
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'Final C-index:' in line and '(Δ=' in line:
        # Extract the delta value
        match = re.search(r'\(Δ=([+-]?\d+\.\d+)\)', line)
        if match:
            improvement = float(match.group(1))

            # Determine which method this belongs to
            if i > 0:
                prev_line = lines[i-1]
                if '[BatchBALD_optimal (λ₁=0.0)]' in prev_line:
                    methods['BatchBALD_optimal (λ₁=0.0)'].append(improvement)
                elif '[BatchBALD_optimal (λ₁=0.3)]' in prev_line:
                    methods['BatchBALD_optimal (λ₁=0.3)'].append(improvement)
                elif '[BatchBALD]' in prev_line:
                    methods['BatchBALD'].append(improvement)
                elif '[Variance]' in prev_line:
                    methods['Variance'].append(improvement)
                elif '[Entropy]' in prev_line:
                    methods['Entropy'].append(improvement)
                elif '[Random]' in prev_line:
                    methods['Random'].append(improvement)

print("="*70)
print("OPTIMIZED BATCHBALD RESULTS - PROPER ANALYSIS")
print("="*70)
print()

# Compute statistics
results = {}
for method, improvements in methods.items():
    if len(improvements) > 0:
        results[method] = {
            'improvements': improvements,
            'mean': np.mean(improvements),
            'std': np.std(improvements),
            'n': len(improvements)
        }
        print(f"{method}:")
        print(f"  Improvements per trial: {[f'{x:+.4f}' for x in improvements]}")
        print(f"  Mean: {results[method]['mean']:+.4f} ± {results[method]['std']:.4f}")
        print()

# Sort by mean improvement
sorted_methods = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)

print("="*70)
print("RANKING BY MEAN IMPROVEMENT")
print("="*70)
for rank, (method, stats_dict) in enumerate(sorted_methods, 1):
    emoji = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
    print(f"{rank}. {method:<35} {stats_dict['mean']:+.4f} ± {stats_dict['std']:.4f} {emoji}")

print()
print("="*70)
print("KEY COMPARISONS")
print("="*70)

# Compare optimized vs original BatchBALD
if 'BatchBALD' in results and 'BatchBALD_optimal (λ₁=0.0)' in results:
    bb_orig = results['BatchBALD']['improvements']
    bb_opt_window = results['BatchBALD_optimal (λ₁=0.0)']['improvements']
    diff = np.mean(bb_opt_window) - np.mean(bb_orig)
    t_stat, p_val = stats.ttest_rel(bb_opt_window, bb_orig)
    print(f"\n1. BatchBALD_optimal (λ₁=0.0) vs BatchBALD:")
    print(f"   Mean difference: {diff:+.4f}")
    print(f"   t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
    if p_val < 0.05:
        print(f"   ✓ Statistically significant improvement!")
    else:
        print(f"   ✗ Not statistically significant")

# Compare window-only vs window+variance
if 'BatchBALD_optimal (λ₁=0.0)' in results and 'BatchBALD_optimal (λ₁=0.3)' in results:
    bb_window = results['BatchBALD_optimal (λ₁=0.0)']['improvements']
    bb_both = results['BatchBALD_optimal (λ₁=0.3)']['improvements']
    diff = np.mean(bb_both) - np.mean(bb_window)
    t_stat, p_val = stats.ttest_rel(bb_both, bb_window)
    print(f"\n2. BatchBALD_optimal (λ₁=0.3) vs BatchBALD_optimal (λ₁=0.0):")
    print(f"   Mean difference: {diff:+.4f}")
    print(f"   t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
    print(f"   → Adding variance term (λ₁=0.3) {'helps' if diff > 0 else 'hurts'}")

# Compare best optimized vs Variance
if 'BatchBALD_optimal (λ₁=0.3)' in results and 'Variance' in results:
    bb_best = results['BatchBALD_optimal (λ₁=0.3)']['improvements']
    variance = results['Variance']['improvements']
    diff = np.mean(bb_best) - np.mean(variance)
    t_stat, p_val = stats.ttest_ind(bb_best, variance)
    print(f"\n3. BatchBALD_optimal (λ₁=0.3) vs Variance:")
    print(f"   Mean difference: {diff:+.4f}")
    print(f"   t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")
    if diff > 0:
        print(f"   ✓ Optimized BatchBALD beats Variance!")
    else:
        print(f"   ✗ Variance still wins")

print()
