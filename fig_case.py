import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ================= SCI 风格 =================
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300

def plot_final_case_study():
    print("正在生成最终 Case Study 图表...")
    
    # 真实数据
    # Report 352: F1=88.89, P=80.00, R=100.00
    # Report 509: F1=43.24, P=38.10, R=50.00
    
    labels = ['Report 352\n(Explicit Patterns)', 'Report 509\n(Implicit/Ambiguous)']
    precision = [80.00, 38.10]
    recall =    [100.00, 50.00]
    f1 =        [88.89, 43.24]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 柱状图
    r1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.9, edgecolor='white')
    r2 = ax.bar(x, recall, width, label='Recall', color='#f1c40f', alpha=0.9, edgecolor='white')
    r3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c', alpha=0.9, edgecolor='black', linewidth=1.5)
    
    # 标注数值
    for bars in [r1, r2, r3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
    # 添加注释箭头
    # 1. Perfect Recall
    ax.annotate('Perfect Recall\n(Explicit verbs)', 
                xy=(x[0], 105), xytext=(x[0], 115),
                ha='center', color='#27ae60', fontweight='bold', fontsize=11,
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
    
    # 2. Sibling Confusion
    ax.annotate('Sibling Confusion\n(Ambiguous Intent)', 
                xy=(x[1], 55), xytext=(x[1], 65),
                ha='center', color='#c0392b', fontweight='bold', fontsize=11,
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=2))

    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_title('Case Study: Impact of Semantic Ambiguity', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 130) # 留出顶部空间给注释
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('fig_case_study.pdf', bbox_inches='tight')
    print("✅ 已保存: fig_case_study.pdf")
    plt.show()

if __name__ == "__main__":
    plot_final_case_study()