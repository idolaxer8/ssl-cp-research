"""Plot MA-CS lambda sweep results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Results from macs_experiment.py (geodesic_topk_mean, 3 trials, CIFAR-100)
lambdas = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]

data = {
    300: {"cov": [0.879, 0.877, 0.879, 0.877, 0.877, 0.871],
          "sz":  [11.29, 10.33, 9.51,  8.73,  7.67,  6.42]},
    400: {"cov": [0.880, 0.876, 0.874, 0.876, 0.870, 0.880],
          "sz":  [5.53,  4.90,  4.56,  4.37,  4.10,  4.24]},
    600: {"cov": [0.901, 0.908, 0.902, 0.903, 0.903, 0.908],
          "sz":  [3.27,  3.01,  2.71,  2.65,  2.56,  2.92]},
    800: {"cov": [0.898, 0.898, 0.899, 0.902, 0.897, 0.896],
          "sz":  [2.20,  2.08,  2.01,  1.99,  1.98,  2.25]},
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

colors = {300: '#e74c3c', 400: '#e67e22', 600: '#27ae60', 800: '#2980b9'}
markers = {300: 's', 400: '^', 600: 'o', 800: 'D'}

# Left: set size vs lambda
ax = axes[0]
for cal in [300, 400, 600, 800]:
    ax.plot(lambdas, data[cal]["sz"], marker=markers[cal], color=colors[cal],
            label=f'cal={cal}', linewidth=2, markersize=7)
ax.set_xlabel('λ (MA-CS penalty)', fontsize=12)
ax.set_ylabel('Avg prediction set size', fontsize=12)
ax.set_title('MA-CS: Set Size vs λ', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(lambdas)

# Right: coverage vs lambda
ax = axes[1]
ax.axhline(y=0.90, color='gray', linestyle='--', linewidth=1.5, label='target (1-α=0.9)')
for cal in [300, 400, 600, 800]:
    ax.plot(lambdas, data[cal]["cov"], marker=markers[cal], color=colors[cal],
            label=f'cal={cal}', linewidth=2, markersize=7)
ax.set_xlabel('λ (MA-CS penalty)', fontsize=12)
ax.set_ylabel('Coverage', fontsize=12)
ax.set_title('MA-CS: Coverage vs λ', fontsize=13)
ax.legend(fontsize=10, loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_xticks(lambdas)
ax.set_ylim(0.85, 0.92)

plt.suptitle('MA-CS Binary Superclass Penalty — geodesic_topk_mean FCP on CIFAR-100',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

out_dir = "output/macs_sweep"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "macs_lambda_sweep.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
