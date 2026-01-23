"""
Run all unit tests.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("="*80)
print("RUNNING ALL TESTS")
print("="*80)

# Import and run all tests
from test_data import (
    test_generate_survival_data,
    test_artificial_censoring,
    test_split_data
)

from test_oracle import (
    test_oracle_query,
    test_oracle_outcome_probs,
    test_oracle_ensemble_probs
)

from test_acquisition import (
    test_batchbald,
    test_entropy_acquisition,
    test_variance_acquisition,
    test_acquisition_comparison
)

# Run data tests
print("\n" + "="*80)
print("DATA TESTS")
print("="*80)
test_generate_survival_data()
test_artificial_censoring()
test_split_data()

# Run oracle tests
print("\n" + "="*80)
print("ORACLE TESTS")
print("="*80)
test_oracle_query()
test_oracle_outcome_probs()
test_oracle_ensemble_probs()

# Run acquisition tests
print("\n" + "="*80)
print("ACQUISITION TESTS")
print("="*80)
test_batchbald()
test_entropy_acquisition()
test_variance_acquisition()
test_acquisition_comparison()

print("\n" + "="*80)
print("ALL TESTS PASSED!")
print("="*80)
