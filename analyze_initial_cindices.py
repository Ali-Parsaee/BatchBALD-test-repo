"""Extract initial and final C-indices for each method across trials."""

import re
import numpy as np

with open('budget20_results.txt', 'r') as f:
    content = f.read()

methods = ['BatchBALD', 'BatchBALD_optimal (λ₁=0.0)', 'BatchBALD_optimal (λ₁=0.3)',
           'Variance', 'Entropy', 'Random']

print("="*100)
print("INITIAL vs FINAL C-INDEX BY TRIAL")
print("="*100)
print()

# Extract data for each trial
for trial_num in range(1, 6):
    print(f"Trial {trial_num}:")
    print("-" * 100)
    print(f"{'Method':<40} {'Initial':<12} {'Final':<12} {'Improvement':<12}")
    print("-" * 100)

    # Find trial section
    trial_pattern = rf'Trial {trial_num}/5.*?(?=Trial {trial_num+1}/5|RESULTS|$)'
    trial_match = re.search(trial_pattern, content, re.DOTALL)

    if trial_match:
        trial_text = trial_match.group(0)

        for method in methods:
            # Find initial and final for this method
            init_pattern = rf'\[{re.escape(method)}\] Initial C-index: ([\d.]+)'
            final_pattern = rf'\[{re.escape(method)}\] Final C-index: ([\d.]+) \(Δ=([+-][\d.]+)\)'

            init_match = re.search(init_pattern, trial_text)
            final_match = re.search(final_pattern, trial_text)

            if init_match and final_match:
                initial = float(init_match.group(1))
                final = float(final_match.group(1))
                improvement = float(final_match.group(2))

                print(f"{method:<40} {initial:.4f}       {final:.4f}       {improvement:+.4f}")

    print()

# Calculate average initial C-index per method
print("="*100)
print("AVERAGE INITIAL C-INDEX BY METHOD (Across 5 trials)")
print("="*100)
print()

initial_by_method = {method: [] for method in methods}

for trial_num in range(1, 6):
    trial_pattern = rf'Trial {trial_num}/5.*?(?=Trial {trial_num+1}/5|RESULTS|$)'
    trial_match = re.search(trial_pattern, content, re.DOTALL)

    if trial_match:
        trial_text = trial_match.group(0)

        for method in methods:
            init_pattern = rf'\[{re.escape(method)}\] Initial C-index: ([\d.]+)'
            init_match = re.search(init_pattern, trial_text)

            if init_match:
                initial_by_method[method].append(float(init_match.group(1)))

print(f"{'Method':<40} {'Mean Initial':<15} {'Std Dev':<12} {'Range':<20}")
print("-" * 100)

for method in methods:
    if len(initial_by_method[method]) > 0:
        initials = np.array(initial_by_method[method])
        mean_init = np.mean(initials)
        std_init = np.std(initials, ddof=1)
        min_init = np.min(initials)
        max_init = np.max(initials)

        print(f"{method:<40} {mean_init:.4f}          {std_init:.4f}       [{min_init:.4f}, {max_init:.4f}]")

print()
print("="*100)
print("KEY OBSERVATION")
print("="*100)
print()
print("Notice how different methods have DIFFERENT initial C-indices!")
print("This is because each method trains its own base model with random initialization.")
print()
print("For a fair comparison, all methods should start from the SAME initial model.")
