"""
Generate fig_performance_ablation.pdf — Performance comparison and ablation study.
Reads data from evaluation_results.json (cached output of run_main_evaluation.py).
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

DATA_FILE = "evaluation_results.json"
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found. Run run_main_evaluation.py first, "
                            f"or ensure the cached results file is present in the repository.")

with open(DATA_FILE, 'r') as f:
    data = json.load(f)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ==================== Subplot (a): Performance Comparison ====================
# Select methods for visual comparison (Tier 1 supervised + key baselines + ours)
figure_methods = [
    "TF-IDF + SVM", "TTPDrill", "AttacKG", "ATHRNN", "HM-ACNN",
    "FT-SecureBERT", "ACRCNN", "FT-LLaMA-3-8B (LoRA)",
    "Naive RAG (Cosine)", "Deep-AttacKG (Protocol B)"
]
display_names = [
    'TF-IDF\n+SVM', 'TTPDrill', 'AttacKG', 'ATHRNN', 'HM-ACNN',
    'FT-\nSecureBERT', 'ACRCNN\n(Sup.)', 'FT-LLaMA\n-3-8B',
    'Naive RAG\n(Retrieval)', 'Deep-AttacKG\n(Ours)'
]

main_results = {m['name']: m for m in data['main_results']['methods']}

precision = [main_results[name]['precision'] for name in figure_methods]
recall    = [main_results[name]['recall']    for name in figure_methods]
f1        = [main_results[name]['micro_f1']  for name in figure_methods]

x = np.arange(len(figure_methods))
width = 0.25

bars1 = ax1.bar(x - width, precision, width, label='Precision', color='#8ECFC9', edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x, recall, width, label='Recall', color='#FFBE7A', edgecolor='black', linewidth=0.5)
bars3 = ax1.bar(x + width, f1, width, label='Micro-F1', color='#FA7F6F', edgecolor='black', linewidth=0.5)

ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(display_names, fontsize=8, rotation=30, ha='right')
ax1.set_title('(a) Overall Performance Comparison', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.set_ylim(0, 0.85)
ax1.grid(axis='y', linestyle='--', alpha=0.4)

for i, bar in enumerate(bars3):
    if i == len(figure_methods) - 1:
        bar.set_edgecolor('#B22222')
        bar.set_linewidth(2)

# ==================== Subplot (b): Ablation Study ====================
stages      = [s['stage'] for s in data['ablation']]
prec_abl    = [s['precision'] for s in data['ablation']]
rec_abl     = [s['recall']    for s in data['ablation']]
f1_abl      = [s['micro_f1']  for s in data['ablation']]

display_stages = ['M1: Hybrid\nRetrieval Only', 'M2: + Cross-Encoder\nReranking', 'M3: + Logic\nReasoning']

x2 = np.arange(len(stages))
width2 = 0.25

ax2.bar(x2 - width2, [p * 100 for p in prec_abl], width2, label='Precision', color='#8ECFC9', edgecolor='black', linewidth=0.5)
ax2.bar(x2, [r * 100 for r in rec_abl], width2, label='Recall', color='#FFBE7A', edgecolor='black', linewidth=0.5)
ax2.bar(x2 + width2, [f * 100 for f in f1_abl], width2, label='Micro-F1', color='#FA7F6F', edgecolor='black', linewidth=0.5)

ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(display_stages, fontsize=10)
ax2.set_title('(b) Incremental Ablation Study', fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.set_ylim(0, 105)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

for bar_group in [ax2.containers[0], ax2.containers[1], ax2.containers[2]]:
    for bar in bar_group:
        height = bar.get_height()
        if height > 5:
            ax2.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_performance_ablation.pdf', format='pdf', dpi=300)
plt.savefig('fig_performance_ablation.png', format='png', dpi=300)
print("Generated: fig_performance_ablation.pdf (data source: evaluation_results.json)")
