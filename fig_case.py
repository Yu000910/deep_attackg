"""
Generate fig_case_study.pdf — Case Study: Impact of Semantic Ambiguity.
Reads data from case_study_results.json (cached output of run_case_study.py).
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DATA_FILE = "case_study_results.json"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found. Run run_case_study.py first, "
                            f"or ensure the cached results file is present in the repository.")

with open(DATA_FILE, 'r') as f:
    data = json.load(f)

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300

def plot_final_case_study():
    print("Generating case study figure from case_study_results.json...")

    labels = [r['report'] for r in data['results']]
    precision = [r['precision'] for r in data['results']]
    recall    = [r['recall']    for r in data['results']]
    f1        = [r['micro_f1']  for r in data['results']]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))

    r1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.9, edgecolor='white')
    r2 = ax.bar(x, recall, width, label='Recall', color='#f1c40f', alpha=0.9, edgecolor='white')
    r3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c', alpha=0.9, edgecolor='black', linewidth=1.5)

    for bars in [r1, r2, r3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.annotate('Perfect Recall\n(Explicit verbs)',
                xy=(x[0], 105), xytext=(x[0], 115),
                ha='center', color='#27ae60', fontweight='bold', fontsize=11,
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))

    ax.annotate('Sibling Confusion\n(Ambiguous Intent)',
                xy=(x[1], 55), xytext=(x[1], 65),
                ha='center', color='#c0392b', fontweight='bold', fontsize=11,
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))

    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_title('Case Study: Impact of Semantic Ambiguity', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 130)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('fig_case_study.pdf', bbox_inches='tight')
    plt.savefig('fig_case_study.png', bbox_inches='tight')
    print("Generated: fig_case_study.pdf (data source: case_study_results.json)")

if __name__ == "__main__":
    plot_final_case_study()
