"""
Final comprehensive comparison with multiple seeds.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from fast_diagnostic import run_fast_experiment


if __name__ == '__main__':
    print("\n" + "="*80)
    print("FINAL COMPREHENSIVE COMPARISON")
    print("="*80)
    print("\nRunning 10 experiments with different random seeds...\n")

    all_results = []
    seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    for seed in seeds:
        print(f"\n{'#'*80}")
        print(f"# RUN {len(all_results) + 1}/10 (seed={seed})")
        print(f"{'#'*80}")

        results = run_fast_experiment(seed)
        all_results.append(results)

    # Comprehensive summary
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS (10 RUNS)")
    print(f"{'='*80}\n")

    methods = ['BatchBALD', 'Entropy', 'Variance']

    print(f"{'Method':<12} {'Mean Δ C-idx':<15} {'Std':<10} {'Wins':<8} {'Win %':<8}")
    print("-" * 70)

    results_summary = {}
    for method in methods:
        improvements = [r[method]['improvement'] for r in all_results]
        mean_imp = np.mean(improvements)
        std_imp = np.std(improvements)
        wins = sum(1 for imp in improvements if imp > 0)
        win_pct = wins / len(improvements) * 100

        results_summary[method] = {
            'mean': mean_imp,
            'std': std_imp,
            'wins': wins,
            'win_pct': win_pct
        }

        print(f"{method:<12} {mean_imp:+.4f}         {std_imp:.4f}    "
              f"{wins}/10    {win_pct:.1f}%")

    # Determine winner
    print(f"\n{'='*80}")
    best_method = max(methods, key=lambda m: results_summary[m]['mean'])
    second_method = sorted(methods, key=lambda m: results_summary[m]['mean'], reverse=True)[1]

    print(f"WINNER: {best_method}")
    print(f"  Average improvement: {results_summary[best_method]['mean']:+.4f}")
    print(f"  Win rate: {results_summary[best_method]['win_pct']:.1f}%")

    print(f"\nRunner-up: {second_method}")
    print(f"  Average improvement: {results_summary[second_method]['mean']:+.4f}")
    print(f"  Win rate: {results_summary[second_method]['win_pct']:.1f}%")

    # Statistical significance check (simple t-test)
    from scipy.stats import ttest_rel

    best_improvements = [r[best_method]['improvement'] for r in all_results]
    second_improvements = [r[second_method]['improvement'] for r in all_results]

    t_stat, p_value = ttest_rel(best_improvements, second_improvements)

    print(f"\n{'='*80}")
    print(f"Statistical Significance (paired t-test):")
    print(f"{best_method} vs {second_method}: t={t_stat:.3f}, p={p_value:.3f}")

    if p_value < 0.05:
        print(f"✓ Statistically significant difference (p < 0.05)")
    else:
        print(f"✗ NOT statistically significant (p >= 0.05)")

    print(f"{'='*80}\n")

    # Show detailed breakdown
    print("\nDetailed Results:")
    print("-" * 80)
    print(f"{'Run':<6} {'BatchBALD':<15} {'Entropy':<15} {'Variance':<15}")
    print("-" * 80)

    for i, results in enumerate(all_results):
        print(f"{i+1:<6} {results['BatchBALD']['improvement']:+.4f}          "
              f"{results['Entropy']['improvement']:+.4f}          "
              f"{results['Variance']['improvement']:+.4f}")

    print("="*80)
