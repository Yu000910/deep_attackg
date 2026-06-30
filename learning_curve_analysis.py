"""
ACRNN Learning Curve Analysis — 使用真实训练日志
数据来源: deep_learning_train_with_logging.py 输出的 learning_curve_real_data.json
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import os

LOG_FILE = "learning_curve_real_data.json"

# ================= 1. 加载数据 =================
def load_real_data():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            data = json.load(f)
        records = data["records"]
        epochs = np.array([r["epoch"] for r in records])
        train_loss = np.array([r["train_loss"] for r in records])
        val_recall_all = np.array([r["val_recall"] for r in records])
        val_recall_head = np.array([r["val_recall_head"] for r in records])
        val_recall_tail = np.array([r["val_recall_tail"] for r in records])
        print(f"Loaded real training data: {len(records)} epochs, "
              f"head={data['head_classes']}, tail={data['tail_classes']}")
        return epochs, train_loss, val_recall_head, val_recall_tail, val_recall_all, True
    else:
        print(f"WARNING: {LOG_FILE} not found. Using simulated data as placeholder.")
        return None, None, None, None, None, False

def load_simulated_data():
    epochs = np.arange(1, 21)
    train_loss = 0.8 * np.exp(-0.3 * epochs) + 0.05 + np.random.normal(0, 0.01, 20)
    val_recall_head = 0.55 - 0.45 * np.exp(-0.25 * epochs) + np.random.normal(0, 0.01, 20)
    val_recall_head = np.clip(val_recall_head, 0, 1)
    val_recall_tail = 0.02 - 0.02 * np.exp(-0.5 * epochs) + np.random.normal(0, 0.005, 20)
    val_recall_tail = np.clip(val_recall_tail, 0, 0.05)
    val_recall_all = 0.195 - 0.18 * np.exp(-0.3 * epochs) + np.random.normal(0, 0.005, 20)
    val_recall_all = np.clip(val_recall_all, 0, 1)
    return epochs, train_loss, val_recall_head, val_recall_tail, val_recall_all, False

# ================= 2. 绘制 =================
def plot_learning_curve(epochs, train_loss, val_recall_head, val_recall_tail, val_recall_all, is_real):
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    color_loss = '#1f77b4'
    ax1.set_xlabel('Training Epochs', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Binary Cross Entropy (BCE) Loss', color=color_loss, fontsize=14, fontweight='bold')
    ax1.plot(epochs, train_loss, color=color_loss, linestyle='-', marker='o', markersize=5, linewidth=2, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss, labelsize=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color_head = '#d62728'
    color_all = '#2ca02c'
    color_tail = '#7f7f7f'

    ax2.set_ylabel('Validation Recall', color='black', fontsize=14, fontweight='bold')
    ax2.plot(epochs, val_recall_head, color=color_head, linestyle='--', linewidth=2.5, label='Recall (Head Classes)')
    ax2.plot(epochs, val_recall_all, color=color_all, linestyle='-', linewidth=3, label='Global Micro-Recall')
    ax2.plot(epochs, val_recall_tail, color=color_tail, linestyle=':', linewidth=2.5, label='Recall (Tail Classes — Representation Collapse)')

    ax2.tick_params(axis='y', labelcolor='black', labelsize=12)
    ax2.set_ylim(0, max(0.6, np.max(val_recall_head) * 1.15))

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', fontsize=11, framealpha=0.9, edgecolor='black')

    data_label = "Real Training Data" if is_real else "Simulated Data (Placeholder)"
    plt.title(f'Learning Dynamics of ACRCNN Baseline\n(Demonstrating the BCE Loss Trap) [{data_label}]',
              fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('learning_curve_analysis.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('learning_curve_analysis.png', format='png', dpi=300, bbox_inches='tight')
    print(f"Chart saved: learning_curve_analysis.pdf (source: {data_label})")

# ================= 3. Main =================
epochs, train_loss, val_recall_head, val_recall_tail, val_recall_all, is_real = load_real_data()
if not is_real:
    epochs, train_loss, val_recall_head, val_recall_tail, val_recall_all, is_real = load_simulated_data()
plot_learning_curve(epochs, train_loss, val_recall_head, val_recall_tail, val_recall_all, is_real)
