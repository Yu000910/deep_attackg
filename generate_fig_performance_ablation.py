"""
Generate fig_performance_ablation.pdf — Performance comparison and ablation study.
Replaces old figure that contained "Ours (Neuro-Symbolic)" text.
Data source: Table 3 (Main Results) and Table 5 (Ablation Study) from the paper.
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ==================== Subplot (a): Performance Comparison ====================
methods = [
    'TF-IDF\n+SVM', 'TTPDrill', 'AttacKG', 'ATHRNN', 'HM-ACNN',
    'FT-\nSecureBERT', 'ACRCNN\n(Sup.)', 'FT-LLaMA\n-3-8B',
    'Naive RAG\n(Retrieval)', 'Deep-AttacKG\n(Ours)'
]

precision = [0.185, 0.212, 0.235, 0.224, 0.268, 0.312, 0.342, 0.245, 0.452, 0.513]
recall =    [0.132, 0.145, 0.152, 0.158, 0.172, 0.184, 0.195, 0.191, 0.548, 0.763]
f1 =        [0.154, 0.168, 0.179, 0.185, 0.209, 0.231, 0.248, 0.215, 0.495, 0.613]

x = np.arange(len(methods))
width = 0.25

bars1 = ax1.bar(x - width, precision, width, label='Precision', color='#8ECFC9', edgecolor='black', linewidth=0.5)
bars2 = ax1.bar(x, recall, width, label='Recall', color='#FFBE7A', edgecolor='black', linewidth=0.5)
bars3 = ax1.bar(x + width, f1, width, label='Micro-F1', color='#FA7F6F', edgecolor='black', linewidth=0.5)

ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=8, rotation=30, ha='right')
ax1.set_title('(a) Overall Performance Comparison', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.set_ylim(0, 0.85)
ax1.grid(axis='y', linestyle='--', alpha=0.4)

# Highlight Deep-AttacKG
for i, bar in enumerate(bars3):
    if i == len(methods) - 1:
        bar.set_edgecolor('#B22222')
        bar.set_linewidth(2)

# ==================== Subplot (b): Ablation Study ====================
stages = ['M1: Hybrid\nRetrieval Only', 'M2: + Cross-Encoder\nReranking', 'M3: + Logic\nReasoning']
prec_abl = [0.154, 0.356, 0.5125]
rec_abl  = [0.895, 0.821, 0.7632]
f1_abl   = [0.262, 0.496, 0.6132]

x2 = np.arange(len(stages))
width2 = 0.25

ax2.bar(x2 - width2, [p * 100 for p in prec_abl], width2, label='Precision', color='#8ECFC9', edgecolor='black', linewidth=0.5)
ax2.bar(x2, [r * 100 for r in rec_abl], width2, label='Recall', color='#FFBE7A', edgecolor='black', linewidth=0.5)
ax2.bar(x2 + width2, [f * 100 for f in f1_abl], width2, label='Micro-F1', color='#FA7F6F', edgecolor='black', linewidth=0.5)

ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(stages, fontsize=10)
ax2.set_title('(b) Incremental Ablation Study', fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', fontsize=9)
ax2.set_ylim(0, 105)
ax2.grid(axis='y', linestyle='--', alpha=0.4)

# Add value labels on bars
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
print("Generated: fig_performance_ablation.pdf")
