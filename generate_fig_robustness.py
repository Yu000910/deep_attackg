"""
Generate fig_robustness.pdf — Long-Tail Robustness analysis.
Replaces old figure that contained "Neuro-Symbolic (Ours)" text.
Data source: Section 4.3.3 (RQ3: Robustness Analysis) from the paper.
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

fig, ax = plt.subplots(figsize=(8, 5.5))

# Frequency groups and corresponding F1 scores
groups = ['Head\n(>50 samples)', 'Middle\n(5–50 samples)', 'Tail\n(<5 samples)']
x = np.arange(len(groups))
width = 0.3

# Deep-AttacKG: stable across groups
deep_attackg_f1 = [0.586, 0.534, 0.512]  # F1 scores from paper description

# LLaMA-3: steep drop on tail
llama3_f1 = [0.425, 0.218, 0.082]  # Performance cliff on tail

# ACRCNN (supervised baseline)
acrcnn_f1 = [0.312, 0.215, 0.058]

bars1 = ax.bar(x - width, deep_attackg_f1, width, label='Deep-AttacKG (Ours)',
               color='#FA7F6F', edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x, llama3_f1, width, label='FT-LLaMA-3-8B (Generative)',
               color='#82B0D2', edgecolor='black', linewidth=0.8)
bars3 = ax.bar(x + width, acrcnn_f1, width, label='ACRCNN (Supervised)',
               color='#8ECFC9', edgecolor='black', linewidth=0.8)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('Micro-F1 Score', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=11)
ax.set_title('Long-Tail Robustness Across Frequency Groups', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='black')
ax.set_ylim(0, 0.75)
ax.grid(axis='y', linestyle='--', alpha=0.4)

# Add annotation about the performance cliff
ax.annotate('Generative\nPerformance Cliff',
            xy=(2, 0.082), xytext=(1.5, 0.15),
            arrowprops=dict(facecolor='#B22222', shrink=0.05, width=1.5, headwidth=8),
            fontsize=10, color='#B22222', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Add annotation about stability
ax.annotate('Stable Geometric Priors\n(F1 > 0.50 even on tail)',
            xy=(2, 0.512), xytext=(1.2, 0.62),
            arrowprops=dict(facecolor='#228B22', shrink=0.05, width=1.5, headwidth=8),
            fontsize=10, color='#228B22', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('fig_robustness.pdf', format='pdf', dpi=300)
plt.savefig('fig_robustness.png', format='png', dpi=300)
print("Generated: fig_robustness.pdf")
