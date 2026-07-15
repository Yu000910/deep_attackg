"""
Generate fig_robustness.pdf — Long-Tail Robustness analysis.
Reads data from fairness_results.json (cached output of run_main_evaluation.py --compute-fairness).
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

DATA_FILE = "fairness_results.json"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found. Run run_main_evaluation.py --compute-fairness first, "
                            f"or ensure the cached results file is present in the repository.")

with open(DATA_FILE, 'r') as f:
    data = json.load(f)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

fig, ax = plt.subplots(figsize=(9, 5.5))

groups = ['Head\n(>50 samples)', 'Middle\n(5–50 samples)', 'Tail\n(<5 samples)']
x = np.arange(len(groups))
width = 0.28

deep_attackg_f1 = [data['results']['Deep-AttacKG (Ours)'][g] for g in data['groups']]
llama3_f1       = [data['results']['FT-LLaMA-3-8B (Generative)'][g] for g in data['groups']]
acrcnn_f1       = [data['results']['ACRCNN (Supervised)'][g] for g in data['groups']]

bars1 = ax.bar(x - width, deep_attackg_f1, width, label='Deep-AttacKG (Ours)',
               color='#FA7F6F', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x, llama3_f1, width, label='FT-LLaMA-3-8B (Generative)',
               color='#82B0D2', edgecolor='black', linewidth=0.8)
bars3 = ax.bar(x + width, acrcnn_f1, width, label='ACRCNN (Supervised)',
               color='#8ECFC9', edgecolor='black', linewidth=0.8)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax.set_ylabel('Micro-F1 Score', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=11)
ax.set_title('Long-Tail Robustness Across Frequency Groups', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='black')
ax.set_ylim(0, 0.85)
ax.grid(axis='y', linestyle='--', alpha=0.4)

ax.annotate('Generative\nPerformance Cliff',
            xy=(2, 0.082), xytext=(0.2, 0.32),
            arrowprops=dict(facecolor='#B22222', shrink=0.05, width=1.5, headwidth=8),
            fontsize=10, color='#B22222', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.annotate('Stable Geometric Priors\n(F1 > 0.50 even on tail)',
            xy=(2 - width, 0.512), xytext=(0.2, 0.78),
            arrowprops=dict(facecolor='#228B22', shrink=0.05, width=1.5, headwidth=8),
            fontsize=10, color='#228B22', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout(pad=1.2)
plt.savefig('fig_robustness.pdf', format='pdf', dpi=300)
plt.savefig('fig_robustness.png', format='png', dpi=300)
print("Generated: fig_robustness.pdf (data source: fairness_results.json)")
