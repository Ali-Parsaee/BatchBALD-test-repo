"""Parse budget=20 experimental results."""

import re
import numpy as np
from scipy import stats

# Read the output file
with open('/tmp/claude/-home-user-BatchBALD-test-repo/tasks/b9f025b.output', 'r') as f:
    content = f.read()

# Extract results for each method
methods = ['BatchBALD', 'BatchBALD_optimal (λ₁=0.0)', 'BatchBALD_optimal (λ₁=0.3)',
           'Variance', 'Entropy', 'Random']

results = {method: {'final': [], 'improvement': []} for method in methods}

# Parse trial sections
trials = re.findall(r'Trial \d+/\d+.*?Trial \d+ completed', content, re.DOTALL)
if not trials:
    # Try to get last trial
    last_trial_match = re.search(r'Trial 5/5(.*)', content, re.DOTALL)
    if last_trial_match:
        trials = re.findall(r'Trial \d+/\d+.*?(?=Trial \d+/\d+|$)', content, re.DOTALL)

# Parse each line with results
for line in content.split('\n'):
    for method in methods:
        if f'[{method}] Final C-index:' in line:
            match = re.search(r'Final C-index: ([\d.]+) \(Δ=([+-][\d.]+)\)', line)
            if match:
                final_cindex = float(match.group(1))
                improvement = float(match.group(2))
                results[method]['final'].append(final_cindex)
                results[method]['improvement'].append(improvement)

print("="*80)
print("BUDGET=20 RESULTS - DETAILED ANALYSIS")
print("="*80)
print()

# Calculate statistics
summary = {}
for method in methods:
    if len(results[method]['final']) > 0:
        final_cindices = np.array(results[method]['final'])
        improvements = np.array(results[method]['improvement'])

        summary[method] = {
            'mean_final': np.mean(final_cindices),
            'std_final': np.std(final_cindices, ddof=1),
            'mean_improvement': np.mean(improvements),
            'std_improvement': np.std(improvements, ddof=1),
            'n_trials': len(final_cindices),
            'final_values': final_cindices,
            'improvement_values': improvements
        }

        print(f"{method}:")
        print(f"  N trials: {len(final_cindices)}")
        print(f"  Final C-index: {np.mean(final_cindices):.4f} ± {np.std(final_cindices, ddof=1):.4f}")
        print(f"  Improvement: {np.mean(improvements):+.4f} ± {np.std(improvements, ddof=1):.4f}")
        print(f"  Individual finals: {[f'{x:.4f}' for x in final_cindices]}")
        print()

# Sort by mean final C-index
sorted_methods = sorted(summary.items(), key=lambda x: x[1]['mean_final'], reverse=True)

print("="*80)
print("RANKING BY MEAN FINAL C-INDEX")
print("="*80)
for rank, (method, stats_dict) in enumerate(sorted_methods, 1):
    emoji = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
    print(f"{rank}. {method:<40} {stats_dict['mean_final']:.4f} ± {stats_dict['std_final']:.4f} {emoji}")
print()

# Sort by mean improvement
sorted_by_improvement = sorted(summary.items(), key=lambda x: x[1]['mean_improvement'], reverse=True)

print("="*80)
print("RANKING BY MEAN IMPROVEMENT")
print("="*80)
for rank, (method, stats_dict) in enumerate(sorted_by_improvement, 1):
    emoji = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
    print(f"{rank}. {method:<40} {stats_dict['mean_improvement']:+.4f} ± {stats_dict['std_improvement']:.4f} {emoji}")
print()

# Statistical comparisons
print("="*80)
print("STATISTICAL COMPARISONS (Paired t-tests)")
print("="*80)

# Compare BatchBALD vs all others
if 'BatchBALD' in summary and summary['BatchBALD']['n_trials'] >= 5:
    bb_final = summary['BatchBALD']['final_values']

    print("\nBatchBALD vs Others:")
    print("-" * 80)
    for method in ['Variance', 'Entropy', 'Random', 'BatchBALD_optimal (λ₁=0.0)', 'BatchBALD_optimal (λ₁=0.3)']:
        if method in summary and summary[method]['n_trials'] >= 5:
            other_final = summary[method]['final_values']

            # Paired t-test
            t_stat, p_val = stats.ttest_rel(bb_final, other_final)
            diff = np.mean(bb_final) - np.mean(other_final)
            sig = "✓ Significant" if p_val < 0.05 else "  Not significant"

            print(f"\nBatchBALD vs {method}:")
            print(f"  Mean difference: {diff:+.4f}")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_val:.4f} {sig}")

# Compare top performers
print("\n" + "="*80)
print("KEY COMPARISONS")
print("="*80)

# Random vs Entropy
if 'Random' in summary and 'Entropy' in summary:
    if summary['Random']['n_trials'] >= 5 and summary['Entropy']['n_trials'] >= 5:
        t_stat, p_val = stats.ttest_rel(
            summary['Random']['final_values'],
            summary['Entropy']['final_values']
        )
        diff = summary['Random']['mean_final'] - summary['Entropy']['mean_final']
        print(f"\nRandom vs Entropy:")
        print(f"  Mean difference: {diff:+.4f}")
        print(f"  t-statistic: {t_stat:.3f}, p-value: {p_val:.4f}")

# Variance vs BatchBALD
if 'Variance' in summary and 'BatchBALD' in summary:
    if summary['Variance']['n_trials'] >= 5 and summary['BatchBALD']['n_trials'] >= 5:
        t_stat, p_val = stats.ttest_rel(
            summary['Variance']['final_values'],
            summary['BatchBALD']['final_values']
        )
        diff = summary['Variance']['mean_final'] - summary['BatchBALD']['mean_final']
        sig = "✓ Significant" if p_val < 0.05 else ""
        print(f"\nVariance vs BatchBALD:")
        print(f"  Mean difference: {diff:+.4f}")
        print(f"  t-statistic: {t_stat:.3f}, p-value: {p_val:.4f} {sig}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print()
print("Budget=10 Results (previous):")
print("  1. BatchBALD: +0.0056 ± 0.0022 (WINNER)")
print("  2. Variance:  +0.0036 ± 0.0013")
print("  3. Entropy:   +0.0019 ± 0.0009")
print()
print("Budget=20 Results (current):")
if len(sorted_methods) >= 3:
    for i, (method, stats_dict) in enumerate(sorted_methods[:3], 1):
        print(f"  {i}. {method}: {stats_dict['mean_final']:.4f} ± {stats_dict['std_final']:.4f}")
print()

if 'BatchBALD' in summary:
    bb_rank = [i for i, (m, _) in enumerate(sorted_methods, 1) if m == 'BatchBALD'][0]
    print(f"BatchBALD ranking: #{bb_rank} (was #1 with budget=10)")
    print(f"BatchBALD improvement: {summary['BatchBALD']['mean_improvement']:+.4f}")
print()
