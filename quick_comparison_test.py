"""
Quick test of the comparison setup (3 trials only).
"""
import sys
sys.path.insert(0, '.')

from compare_acquisition_functions import run_comparison

if __name__ == "__main__":
    print("Running quick comparison test (3 trials)...\n")

    results, summary = run_comparison(
        n_trials=3,
        initial_samples=200,
        budget=30,
        increment=60,
        verbose=True
    )

    print("\n✓ Quick test completed successfully!")
