"""
Streamlined large-scale comparison (optimized for speed).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.synthetic import generate_survival_data, artificially_censor, split_data
from src.models.survival_model import BayesianSurvivalModel
from src.oracle.oracle import Oracle
from src.acquisition.batchbald import SurvivalBatchBALD
from src.acquisition.entropy import EntropyAcquisition
from src.acquisition.variance import VarianceAcquisition
from src.evaluation.metrics import concordance_index
from scipy.stats import ttest_rel


def run_experiment(n_samples, batch_size, probe_depth, random_state=42):
    """Streamlined experiment."""

    # Generate data
    data = generate_survival_data(
        n_samples=n_samples,
        n_features=10,
        n_time_bins=15,
        censoring_rate=0.3,
        random_state=random_state
    )

    train_data, test_data = split_data(data, train_size=0.7, random_state=random_state)
    true_train_time = train_data['time'].copy()
    true_train_event = train_data['event'].copy()

    # Artificially censor
    artificial_time, artificial_event = artificially_censor(
        train_data['time'], train_data['event'],
        proportion=0.5, random_state=random_state
    )

    # Train initial model (fewer epochs for speed)
    initial_model = BayesianSurvivalModel(
        n_features=10, n_time_bins=15, n_ensemble=3, hidden_size=32
    )
    initial_model.fit(
        train_data['X'], artificial_time, artificial_event,
        epochs=30, batch_size=32, verbose=False
    )

    test_preds = initial_model.predict_proba(test_data['X'])
    initial_c_index = concordance_index(test_preds, test_data['time'], test_data['event'])

    # Oracle and predictions
    oracle = Oracle(true_train_time, true_train_event, probe_depth)
    train_preds = initial_model.predict_proba(train_data['X'])

    results = {'initial': initial_c_index}

    # Test methods
    for method_name, acq_func in [
        ('BatchBALD', SurvivalBatchBALD(probe_depth=probe_depth)),
        ('Entropy', EntropyAcquisition()),
        ('Variance', VarianceAcquisition())
    ]:
        # Select batch
        if method_name == 'BatchBALD':
            oracle_probs = oracle.get_oracle_outcome_probs_ensemble(
                train_preds, artificial_time, artificial_event
            )
            selected = acq_func.select_batch(oracle_probs, batch_size, artificial_event)
        else:
            selected = acq_func.select_batch(train_preds, batch_size, artificial_event)

        # Query oracle
        updated_time, updated_event = oracle.query(selected, artificial_time, artificial_event)
        n_revealed = ((updated_event - artificial_event) > 0).sum()

        # Retrain
        updated_model = BayesianSurvivalModel(
            n_features=10, n_time_bins=15, n_ensemble=3, hidden_size=32
        )
        updated_model.fit(
            train_data['X'], updated_time, updated_event,
            epochs=30, batch_size=32, verbose=False
        )

        # Evaluate
        updated_preds = updated_model.predict_proba(test_data['X'])
        updated_c_index = concordance_index(updated_preds, test_data['time'], test_data['event'])

        results[method_name] = {
            'c_index': updated_c_index,
            'improvement': updated_c_index - initial_c_index,
            'n_revealed': n_revealed
        }

    return results


def main():
    print("\n" + "="*100)
    print("STREAMLINED LARGE-SCALE COMPARISON")
    print("="*100 + "\n")

    # Test configurations
    configs = [
        # (n_samples, batch_size, probe_depth, n_runs, description)
        (500, 30, 2, 5, "Small dataset, small batch, small probe"),
        (1000, 50, 2, 5, "Medium dataset, medium batch, small probe"),
        (1000, 80, 2, 5, "Medium dataset, large batch, small probe"),
        (1000, 50, 5, 5, "Medium dataset, medium batch, medium probe"),
        (1000, 50, 8, 5, "Medium dataset, medium batch, large probe"),
        (1500, 75, 5, 3, "Large dataset, large batch, medium probe"),
    ]

    all_results = []

    for config_idx, (n_samples, batch_size, probe_depth, n_runs, desc) in enumerate(configs):
        print(f"\n{'#'*100}")
        print(f"CONFIG {config_idx + 1}/{len(configs)}: {desc}")
        print(f"N={n_samples}, Batch={batch_size}, Probe={probe_depth}, Runs={n_runs}")
        print(f"{'#'*100}\n")

        config_results = []

        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}...", end=" ", flush=True)
            results = run_experiment(n_samples, batch_size, probe_depth, random_state=42 + run)
            config_results.append(results)

            bald = results['BatchBALD']['improvement']
            ent = results['Entropy']['improvement']
            print(f"BALD={bald:+.4f}, Ent={ent:+.4f}")

        # Summary
        bald_imps = [r['BatchBALD']['improvement'] for r in config_results]
        ent_imps = [r['Entropy']['improvement'] for r in config_results]
        var_imps = [r['Variance']['improvement'] for r in config_results]

        bald_mean, bald_std = np.mean(bald_imps), np.std(bald_imps)
        ent_mean, ent_std = np.mean(ent_imps), np.std(ent_imps)
        var_mean, var_std = np.mean(var_imps), np.std(var_imps)

        bald_wins = sum(1 for x in bald_imps if x > 0)
        ent_wins = sum(1 for x in ent_imps if x > 0)

        if n_runs >= 3:
            t_stat, p_value = ttest_rel(bald_imps, ent_imps)
        else:
            p_value = 1.0

        print(f"\n  RESULTS:")
        print(f"  BatchBALD: {bald_mean:+.4f} ± {bald_std:.4f} (wins: {bald_wins}/{n_runs})")
        print(f"  Entropy:   {ent_mean:+.4f} ± {ent_std:.4f} (wins: {ent_wins}/{n_runs})")
        print(f"  Variance:  {var_mean:+.4f} ± {var_std:.4f}")
        print(f"  t-test p-value: {p_value:.4f}")

        winner = "BatchBALD" if bald_mean > ent_mean else "Entropy"
        sig = "YES ✓" if p_value < 0.05 else "NO"

        if p_value < 0.05 and bald_mean > ent_mean:
            print(f"  🎉 BatchBALD SIGNIFICANTLY beats Entropy!")
        elif bald_mean > ent_mean:
            print(f"  ⚠️  BatchBALD wins marginally (not significant)")
        else:
            print(f"  ❌ Entropy wins")

        all_results.append({
            'config': (n_samples, batch_size, probe_depth),
            'desc': desc,
            'bald_mean': bald_mean,
            'ent_mean': ent_mean,
            'var_mean': var_mean,
            'p_value': p_value,
            'bald_wins': bald_wins,
            'n_runs': n_runs
        })

    # Final summary
    print(f"\n\n{'='*100}")
    print("FINAL SUMMARY")
    print(f"{'='*100}\n")

    print(f"{'Config':<45} {'N':<6} {'B':<5} {'k':<4} {'BALD Δ':<12} {'Ent Δ':<12} {'Winner':<10} {'Sig?'}")
    print("-"*100)

    sig_wins = 0
    for r in all_results:
        n, b, k = r['config']
        bald_mean, ent_mean = r['bald_mean'], r['ent_mean']
        p_val = r['p_value']

        winner = "BALD" if bald_mean > ent_mean else "Entropy"
        sig = "YES ✓" if p_val < 0.05 else "NO"

        if p_val < 0.05 and bald_mean > ent_mean:
            sig_wins += 1
            marker = "🎉"
        elif bald_mean > ent_mean:
            marker = "⚠️ "
        else:
            marker = "  "

        print(f"{marker} {r['desc']:<42} {n:<6} {b:<5} {k:<4} "
              f"{bald_mean:+.4f}      {ent_mean:+.4f}      {winner:<10} {sig}")

    print("="*100)
    print(f"\n🎯 BatchBALD SIGNIFICANTLY wins in {sig_wins}/{len(all_results)} configurations\n")

    if sig_wins >= len(all_results) / 2:
        print("✅ CONCLUSION: BatchBALD CONVINCINGLY beats baselines at scale!")
    elif sig_wins > 0:
        print("⚠️  CONCLUSION: BatchBALD wins in some configs but not consistently")
    else:
        print("❌ CONCLUSION: BatchBALD does NOT significantly beat baselines")

    # Show where BatchBALD wins
    print(f"\n{'='*100}")
    print("KEY INSIGHTS:")
    print(f"{'='*100}")

    # Find best config for BatchBALD
    best_config = max(all_results, key=lambda x: x['bald_mean'])
    print(f"\nBest BatchBALD performance:")
    print(f"  Config: {best_config['desc']}")
    print(f"  Improvement: {best_config['bald_mean']:+.4f}")
    print(f"  p-value: {best_config['p_value']:.4f}")

    # Check trends
    print(f"\nTrends:")

    # Larger probe depth helps?
    small_probe = [r for r in all_results if r['config'][2] <= 2]
    large_probe = [r for r in all_results if r['config'][2] >= 5]

    if small_probe and large_probe:
        small_probe_bald = np.mean([r['bald_mean'] for r in small_probe])
        large_probe_bald = np.mean([r['bald_mean'] for r in large_probe])
        print(f"  Small probe depth (k≤2): BALD Δ = {small_probe_bald:+.4f}")
        print(f"  Large probe depth (k≥5): BALD Δ = {large_probe_bald:+.4f}")
        if large_probe_bald > small_probe_bald + 0.01:
            print(f"  → Larger probe depth HELPS BatchBALD! ✓")

    # Larger batch helps?
    small_batch = [r for r in all_results if r['config'][1] <= 50]
    large_batch = [r for r in all_results if r['config'][1] > 50]

    if small_batch and large_batch:
        small_batch_bald = np.mean([r['bald_mean'] for r in small_batch])
        large_batch_bald = np.mean([r['bald_mean'] for r in large_batch])
        print(f"  Small batch (≤50): BALD Δ = {small_batch_bald:+.4f}")
        print(f"  Large batch (>50): BALD Δ = {large_batch_bald:+.4f}")
        if large_batch_bald > small_batch_bald + 0.01:
            print(f"  → Larger batch size HELPS BatchBALD! ✓")

    print("\n" + "="*100 + "\n")


if __name__ == '__main__':
    main()
