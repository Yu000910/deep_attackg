"""
参数敏感性分析 — 优先使用真实实验数据
数据来源: run_sensitivity_sweep.py 输出的 sensitivity_real_data.json
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import os

DATA_FILE = "sensitivity_real_data.json"

# ================= 1. 数据加载 =================
def load_real_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        k1 = data["k1_sweep"]
        k2 = data["k2_sweep"]
        k1_values = [r["k1"] for r in k1]
        recall_k1 = [r["mean_recall"] for r in k1]
        latency_k1 = [r["mean_latency_ms"] for r in k1]
        k2_values = [r["k2"] for r in k2]
        precision_k2 = [r["precision"] for r in k2]
        tokens_k2 = [r["avg_tokens_per_report"] for r in k2]
        print(f"Loaded real sensitivity data: K1={len(k1_values)} pts, K2={len(k2_values)} pts")
        return k1_values, recall_k1, latency_k1, k2_values, precision_k2, tokens_k2, True
    else:
        print(f"WARNING: {DATA_FILE} not found. Using illustrative data.")
        return None, None, None, None, None, None, False

def load_illustrative_data():
    k1_values = [10, 20, 30, 40, 50, 60, 80, 100]
    recall_k1 = [72.1, 81.5, 86.8, 88.5, 89.2, 89.4, 89.6, 89.7]
    latency_k1 = [15, 28, 42, 58, 72, 85, 115, 145]
    k2_values = [3, 5, 8, 10, 12, 15, 20]
    precision_k2 = [48.5, 55.2, 60.1, 62.8, 62.5, 61.2, 58.5]
    tokens_k2 = [1200, 1800, 2600, 3100, 3600, 4400, 5800]
    return k1_values, recall_k1, latency_k1, k2_values, precision_k2, tokens_k2, False

# ================= 2. 绘制 =================
k1_values, recall_k1, latency_k1, k2_values, precision_k2, tokens_k2, is_real = load_real_data()
if not is_real:
    k1_values, recall_k1, latency_k1, k2_values, precision_k2, tokens_k2, is_real = load_illustrative_data()

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# ---- 子图1: K1 ----
color1 = '#1f77b4'
color2 = '#d62728'

ax1.set_xlabel('Stage 1 Candidate Pool Size ($K_1$)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Boundary Recall (%)', color=color1, fontsize=12, fontweight='bold')
line1 = ax1.plot(k1_values, recall_k1, marker='o', color=color1, linewidth=2, label='Recall')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(min(recall_k1) * 0.95, max(recall_k1) * 1.05)

ax1_twin = ax1.twinx()
ax1_twin.set_ylabel('Stage 1 & 2 Latency (ms)', color=color2, fontsize=12, fontweight='bold')
line2 = ax1_twin.plot(k1_values, latency_k1, marker='s', color=color2, linewidth=2, linestyle='--', label='Latency')
ax1_twin.tick_params(axis='y', labelcolor=color2)

ax1.axvline(x=50, color='gray', linestyle=':', linewidth=1.5)
k1_idx = k1_values.index(50) if 50 in k1_values else 4
ax1.annotate('Pareto Optimal\n($K_1=50$)', xy=(50, recall_k1[k1_idx]),
             xytext=(60, recall_k1[k1_idx] - 7),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
             fontsize=10, fontweight='bold')

lines_1 = line1 + line2
labels_1 = [l.get_label() for l in lines_1]
ax1.legend(lines_1, labels_1, loc='lower right')
data_tag = "Real Data" if is_real else "Illustrative"
ax1.set_title(f'(a) Sensitivity of Hybrid Retrieval ($K_1$) [{data_tag}]', fontsize=13, fontweight='bold')

# ---- 子图2: K2 ----
color3 = '#2ca02c'
color4 = '#ff7f0e'

ax2.set_xlabel('Stage 2 Reranked Pool Size ($K_2$)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Micro-Precision (%)', color=color3, fontsize=12, fontweight='bold')
line3 = ax2.plot(k2_values, precision_k2, marker='^', color=color3, linewidth=2, label='Precision')
ax2.tick_params(axis='y', labelcolor=color3)

ax2_twin = ax2.twinx()
ax2_twin.set_ylabel('LLM Prompt Tokens', color=color4, fontsize=12, fontweight='bold')
line4 = ax2_twin.plot(k2_values, tokens_k2, marker='D', color=color4, linewidth=2, linestyle='--', label='Prompt Tokens')
ax2_twin.tick_params(axis='y', labelcolor=color4)

ax2.axvline(x=10, color='gray', linestyle=':', linewidth=1.5)
k2_idx = k2_values.index(10) if 10 in k2_values else 3
ax2.annotate('Optimal Context\n($K_2=10$)', xy=(10, precision_k2[k2_idx]),
             xytext=(12, precision_k2[k2_idx] - 8),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
             fontsize=10, fontweight='bold')

lines_2 = line3 + line4
labels_2 = [l.get_label() for l in lines_2]
ax2.legend(lines_2, labels_2, loc='lower right')
ax2.set_title(f'(b) Sensitivity of LLM Verification ($K_2$) [{data_tag}]', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('parameter_sensitivity.pdf', format='pdf', dpi=300)
plt.savefig('parameter_sensitivity.png', format='png', dpi=300)
print(f"Chart saved: parameter_sensitivity.pdf (source: {data_tag})")
