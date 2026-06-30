# Deep-AttacKG

**Deep-AttacKG: A Logic-Based Framework for Zero-Shot CTI Identification via Semantic Manifold Alignment**

ASOC-D-26-00148R1 | Applied Soft Computing

---

## 环境配置

```bash
conda create -n deep_attackg python=3.10
conda activate deep_attackg
pip install -r requirements.txt
```

## 文件结构

```
├── requirements.txt                          # Python依赖
├── README.md                                 # 本文件
│
├── cti_model_20k_finetuned/   (symlink)      # Bi-Encoder权重 (~437MB)
├── cti_reranker_final/        (symlink)      # Cross-Encoder权重 (~90MB)
├── CTI_reports/               (symlink)      # CTI-1002数据集 (1,002 reports)
├── TRAM/                      (symlink)      # MITRE TRAM数据集
├── attack-pattern/            (symlink)      # MITRE ATT&CK知识库
├── BEDR_resampled_dataset.csv                # BEDR重采样数据集
├── D_BEDR.npz                 (symlink)      # BEDR原始NPZ文件
├── test_split.json                           # CTI-1002评估分割
│
├── run_main_evaluation.py                    # ★ 主评估 (Table 3, Table 4)
├── run_tram_hierarchy_eval.py                # ★ TRAM双指标评估 (Table 6)
├── latency_profiling.py                      # ★ 延迟剖析 (Figure 5b)
├── learning_curve_analysis.py                # ★ 学习曲线 (Figure 4)
├── plot_sensitivity.py                       # ★ 参数敏感性 (Figure 6)
├── run_case_study.py                         # ★ 案例研究 (Figure 7)
│
├── train_with_dataset.py                     # Bi-Encoder训练
├── train_cross_encoder.py                    # Cross-Encoder训练
├── deep-learning-test.py                     # ACRNN Baseline
├── deep_learning_train_with_logging.py        # ACRNN Baseline (带逐epoch日志)
│
├── run_sensitivity_sweep.py                  # 敏感性参数扫描
├── fig_case.py                               # 案例研究辅助
├── utils_kb_filter.py                        # 知识库过滤工具
├── inspect_dataset.py                        # 数据集检查工具
└── debug.py                                  # 调试脚本
```

## 模型权重

Bi-Encoder和Cross-Encoder权重上传至HuggingFace Model Hub：

- **Bi-Encoder:** `https://huggingface.co/Yu000910/deep-attackg-bi-encoder`
- **Cross-Encoder:** `https://huggingface.co/Yu000910/deep-attackg-cross-encoder`

下载后放置于项目根目录即可（或创建软链接）。

## 复现实验

### API配置

Stage 3使用DeepSeek API进行LLM推理。需要在脚本中配置：

```python
LLM_API_KEY = "your-deepseek-api-key"
LLM_BASE_URL = "https://api.deepseek.com"
```

API模型: `deepseek-chat`, temperature=0.0, response_format=json_object.
生成日期: 2025年12月–2026年4月.

### Table 3 & Table 4 (主实验结果 & 消融实验)

```bash
python run_main_evaluation.py
```

- 运行时间: ~1-2小时 (201条测试报告)
- 输出: M1/M2/M3各阶段的Micro-Precision, Recall, F1, TP, FP, FN
- QUICK_TEST=True可快速测试(5条报告)

### Table 6 (TRAM外部验证)

```bash
python run_tram_hierarchy_eval.py
```

- 运行时间: ~2.5小时 (50条TRAM报告)
- 输出: Strict Exact-Match 和 Hierarchy-Aware 双指标对比

### Figure 4 (学习曲线)

```bash
# Step 1: 训练ACRNN Baseline并记录逐epoch日志
python deep_learning_train_with_logging.py

# Step 2: 绘制学习曲线
python learning_curve_analysis.py
```

### Figure 5b (延迟剖析)

```bash
python latency_profiling.py
```

### Figure 6 (参数敏感性)

```bash
# Step 1: 运行参数扫描 (K1 sweep无需LLM, K2 sweep需要LLM)
python run_sensitivity_sweep.py

# Step 2: 绘制敏感性曲线
python plot_sensitivity.py
```

### Figure 7 (案例研究)

```bash
python run_case_study.py
```

## 实验结果

### 主实验 (CTI-1002, N=201)

| Stage | Precision | Recall | Micro-F1 |
|-------|-----------|--------|----------|
| M1: Hybrid Retrieval | 5.06% | 98.98% | 9.63% |
| M2: + Cross-Encoder | 15.50% | 86.47% | 26.28% |
| M3: + LLM Reasoning | 45.78% | 77.41% | 57.53% |

### TRAM外部验证 (N=50)

| Metric | Strict | Hierarchy-Aware |
|--------|--------|-----------------|
| Precision | 13.12% | 24.83% |
| Recall | 59.25% | 75.37% |
| Micro-F1 | 21.48% | 37.35% |

## 数据来源

- **CTI-1002:** 1,002份CTI报告及ATT&CK标注
- **BEDR:** Boundary Entropy-Driven Resampling数据集 (21,453条, 679类)
- **TRAM:** MITRE TRAM公开基准 (multi_label.json)
- **ATT&CK KB:** MITRE ATT&CK v15 Enterprise (attack-pattern/*.json)

## 引用

If you use this code or data, please cite:
```

## License

MIT License

## DOI/Archive

[待添加]
