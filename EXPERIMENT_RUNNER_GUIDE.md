# Experiment Runner Guide

## Quick Start

The `run_experiment.py` script lets you run custom active learning experiments with any configuration.

### Basic Usage

```bash
# Run with defaults (CBALD-Diverse vs C-BALD, budget=20, 5 trials)
python run_experiment.py

# Test CBALD-Adaptive with budget=30
python run_experiment.py --methods cbald_adaptive --budget 30

# Compare all CBALD variants
python run_experiment.py --methods cbald_diverse cbald_adaptive cbald_nofilter cbald_twostage --budget 20

# Run all methods with budget=50
python run_experiment.py --methods all --budget 50 --trials 10
```

## Available Acquisition Functions

| Method Key | Full Name | Description | Performance |
|-----------|-----------|-------------|-------------|
| `cbald_diverse` | CBALD-Diverse | C-BALD + diversity filtering | +0.0180 ± 0.0064 🏆 |
| `cbald_adaptive` | CBALD-Adaptive | Adaptive ratio (90%→50%) | +0.0166 ± 0.0000 |
| `cbald_nofilter` | CBALD-NoFilter | No pre-filtering bias | +0.0166 ± 0.0000 |
| `cbald_twostage` | CBALD-TwoStage | Exploit then explore | +0.0166 ± 0.0000 |
| `c_bald` | C-BALD | Information value weighting | +0.0173 ± 0.0062 🥈 |
| `batchbald` | BatchBALD | Joint entropy diversity | +0.0109 ± 0.0036 |
| `variance` | Variance | Epistemic uncertainty | +0.0117 ± 0.0041 |
| `entropy` | Entropy | Predictive uncertainty | N/A |
| `random` | Random | Random selection baseline | N/A |

## Command-Line Options

### Required Arguments

None! All have sensible defaults.

### Optional Arguments

**`--methods METHOD [METHOD ...]`**
- Acquisition methods to test
- Choices: `cbald_diverse`, `cbald_adaptive`, `cbald_nofilter`, `cbald_twostage`, `c_bald`, `batchbald`, `variance`, `entropy`, `random`, `all`
- Default: `cbald_diverse c_bald`
- Examples:
  ```bash
  --methods cbald_diverse  # Single method
  --methods cbald_diverse c_bald variance  # Multiple methods
  --methods all  # All available methods
  ```

**`--budget NUM`**
- Number of samples to acquire per trial
- Default: `20`
- Recommended range: 10-100
- Examples:
  ```bash
  --budget 10   # Small budget (fast)
  --budget 50   # Large budget (more data)
  --budget 100  # Very large budget (comprehensive)
  ```

**`--increment NUM`**
- Survival time window increment (for C-BALD and variants)
- Default: `60`
- Typical range: 30-120
- Examples:
  ```bash
  --increment 30   # Smaller window
  --increment 90   # Larger window
  ```

**`--initial-samples NUM`**
- Number of initial uncensored samples to start with
- Default: `50`
- Affects starting model quality
- Examples:
  ```bash
  --initial-samples 25   # Harder problem (less initial data)
  --initial-samples 100  # Easier problem (more initial data)
  ```

**`--trials NUM`**
- Number of trials to run for statistical significance
- Default: `5`
- Recommended minimum: 3
- Examples:
  ```bash
  --trials 3    # Quick test
  --trials 10   # High confidence
  ```

**`--prefilter-size NUM`**
- Size of pre-filtered pool (for methods that use it)
- Default: `500`
- Only affects certain methods (BatchBALD, some CBALD variants)
- Examples:
  ```bash
  --prefilter-size 1000  # Larger pool
  ```

**`--seed NUM`**
- Random seed for reproducibility
- Default: `42`
- Examples:
  ```bash
  --seed 123  # Different random initialization
  ```

**`--list-methods`**
- List all available methods and exit
- Usage:
  ```bash
  python run_experiment.py --list-methods
  ```

## Example Use Cases

### 1. Quick Test of New Method
Test a single method quickly:
```bash
python run_experiment.py --methods cbald_adaptive --budget 10 --trials 3
```

### 2. Comprehensive Comparison
Compare all CBALD variants with statistical significance:
```bash
python run_experiment.py --methods cbald_diverse cbald_adaptive cbald_nofilter cbald_twostage c_bald --budget 20 --trials 10
```

### 3. Budget Sensitivity Analysis
Test how performance scales with budget:
```bash
# Small budget
python run_experiment.py --methods cbald_diverse c_bald --budget 10 --trials 5

# Medium budget
python run_experiment.py --methods cbald_diverse c_bald --budget 30 --trials 5

# Large budget
python run_experiment.py --methods cbald_diverse c_bald --budget 50 --trials 5
```

### 4. All Methods Benchmark
Run comprehensive benchmark of all methods:
```bash
python run_experiment.py --methods all --budget 20 --trials 5
```

### 5. Reproducibility Test
Verify results with specific seed:
```bash
python run_experiment.py --methods cbald_diverse --budget 20 --trials 5 --seed 100
```

## Output Format

The script outputs:

1. **Configuration Summary**
   - Methods being tested
   - Budget, increment, initial samples
   - Number of trials

2. **Per-Trial Progress**
   - Base model training
   - Each method's acquisition and retraining
   - C-index improvements

3. **Final Results**
   - Rankings by mean improvement
   - Mean ± standard deviation for each method
   - Top 3 methods get medals 🏆🥈🥉

Example output:
```
RESULTS
======================================================================

Rankings (by mean improvement):
----------------------------------------------------------------------
1. CBALD-Diverse                       +0.0180 ± 0.0064 🏆
2. C-BALD                              +0.0173 ± 0.0062 🥈
3. CBALD-Adaptive                      +0.0166 ± 0.0000 🥉
4. Variance                            +0.0117 ± 0.0041

Experiment complete!
```

## Tips

1. **Start Small**: Use `--trials 3 --budget 10` for quick tests
2. **Be Patient**: Each trial takes ~2-3 minutes per method
3. **Use Seeds**: Use `--seed` for reproducible results
4. **Compare Baselines**: Always include `c_bald` or `variance` as reference
5. **Check Significance**: Use at least 5 trials for reliable statistics

## Troubleshooting

**Slow execution?**
- Reduce `--trials` or `--budget`
- Test fewer methods at once
- Use `--initial-samples 25` for faster base model training

**Out of memory?**
- Reduce `--budget` or `--prefilter-size`
- Script automatically uses CPU if CUDA unavailable

**Unexpected results?**
- Try different `--seed` values to check variance
- Increase `--trials` for more reliable statistics
- Check that methods are spelled correctly (use `--list-methods`)

## Advanced Configuration

For more control, you can:

1. Edit `run_experiment.py` directly
2. Modify acquisition function parameters in `ACQUISITION_FUNCTIONS` dictionary
3. Adjust model hyperparameters in the `config` namespace
4. Change data preprocessing or train/test split ratios

## Contributing

Found a bug or want to add a feature? The script is designed to be easily extensible:

1. Add new acquisition functions to `Model_stuff/acquisition.py`
2. Register them in `ACQUISITION_FUNCTIONS` dictionary
3. Run with `--methods your_new_method`

---

**Happy experimenting!** 🚀
