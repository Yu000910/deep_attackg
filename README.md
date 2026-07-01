# Deep-AttacKG

**Deep-AttacKG: A Logic-Based Framework for Zero-Shot CTI Identification via Semantic Manifold Alignment**

ASOC-D-26-00148R1 | Applied Soft Computing

---

## Environment Setup

```bash
conda create -n deep_attackg python=3.10
conda activate deep_attackg
pip install -r requirements.txt
```

## File Structure

```
├── requirements.txt                          # Python dependencies
├── README.md                                 # This file
│
├── cti_model_20k_finetuned/   (symlink)      # Bi-Encoder weights (~437MB)
├── cti_reranker_final/        (symlink)      # Cross-Encoder weights (~90MB)
├── CTI_reports/               (symlink)      # CTI-1002 dataset (1,002 reports)
├── TRAM/                      (symlink)      # MITRE TRAM dataset
├── attack-pattern/            (symlink)      # MITRE ATT&CK knowledge base (v15)
├── BEDR_resampled_dataset.csv                # BEDR resampled dataset
├── D_BEDR.npz                 (symlink)      # BEDR vectorized training data
├── test_split.json                           # CTI-1002 evaluation split
│
├── run_main_evaluation.py                    # Main evaluation (Table 3, Table 4)
├── run_tram_hierarchy_eval.py                # TRAM dual-metric evaluation (Table 6)
├── latency_profiling.py                      # Latency profiling (Figure 5b)
├── learning_curve_analysis.py                # Learning curve (Figure 4)
├── plot_sensitivity.py                       # Parameter sensitivity (Figure 6)
├── run_case_study.py                         # Case study (Figure 7)
│
├── train_with_dataset.py                     # Bi-Encoder training (InfoNCE)
├── train_cross_encoder.py                    # Cross-Encoder training
├── deep-learning-test.py                     # ACRNN baseline evaluation
├── deep_learning_train_with_logging.py       # ACRNN baseline training (per-epoch logging)
│
├── run_sensitivity_sweep.py                  # Sensitivity parameter sweep
├── fig_case.py                               # Case study visualization helper
├── utils_kb_filter.py                        # Knowledge base filter utility
├── inspect_dataset.py                        # Dataset statistics (Figure 1)
└── debug.py                                  # Debugging script
```

## Model Weights

Trained Bi-Encoder and Cross-Encoder checkpoints are hosted on HuggingFace Model Hub:

- **Bi-Encoder:** `https://huggingface.co/Andou2yu/deep-attackg-bi-encoder`
- **Cross-Encoder:** `https://huggingface.co/Andou2yu/deep-attackg-cross-encoder`

Download via HuggingFace CLI or place manually in the project root:

```bash
huggingface-cli download Andou2yu/deep-attackg-bi-encoder --local-dir ./cti_model_20k_finetuned
huggingface-cli download Andou2yu/deep-attackg-cross-encoder --local-dir ./cti_reranker_final
```

## Reproducing Experiments

### API Configuration

Stage 3 (LLM Logic-Constrained Verification) requires a DeepSeek API key. Edit the relevant scripts to set:

```python
LLM_API_KEY = "your-deepseek-api-key"
LLM_BASE_URL = "https://api.deepseek.com"
```

See [API Reproducibility Notes](#api-reproducibility-notes) below for full details.

### Table & Figure Mapping

| Paper Element | Script | Description |
|---------------|--------|-------------|
| Table 3 (Main Results) | `run_main_evaluation.py` | M1/M2/M3 Precision, Recall, F1, TP, FP, FN |
| Table 4 (Ablation Study) | `run_main_evaluation.py` | Ablation variants |
| Table 5 (BEDR Augmentation) | `run_main_evaluation.py` | ACRCNN$_{aug}$ comparison |
| Table 6 (TRAM External Validation) | `run_tram_hierarchy_eval.py` | Strict and Hierarchy-Aware dual metrics |
| Figure 1 (Dataset Statistics) | `inspect_dataset.py` | Class distribution statistics from BEDR CSV |
| Figure 4 (Learning Curve) | `deep_learning_train_with_logging.py` → `learning_curve_analysis.py` | ACRCNN per-epoch training log + curve plot |
| Figure 5b (Latency Breakdown) | `latency_profiling.py` | Per-stage latency with mean ± std |
| Figure 6 (Parameter Sensitivity) | `run_sensitivity_sweep.py` → `plot_sensitivity.py` | K1/K2 sweep + sensitivity curves |
| Figure 7 (Case Study) | `run_case_study.py` → `fig_case.py` | Qualitative error analysis on Reports 352 and 509 |

### Commands

```bash
# Table 3 & Table 4 (Main Results & Ablation)
python run_main_evaluation.py
# Set QUICK_TEST=True for a fast 5-report sanity check

# Table 6 (TRAM External Validation)
python run_tram_hierarchy_eval.py

# Figure 4 (Learning Curve)
python deep_learning_train_with_logging.py   # Step 1: train + log
python learning_curve_analysis.py            # Step 2: plot

# Figure 5b (Latency Breakdown)
python latency_profiling.py

# Figure 6 (Parameter Sensitivity)
python run_sensitivity_sweep.py              # Step 1: sweep K1, K2
python plot_sensitivity.py                   # Step 2: plot

# Figure 7 (Case Study)
python run_case_study.py
```

### Training Scripts

```bash
# Bi-Encoder (InfoNCE contrastive loss, BEDR dataset)
python train_with_dataset.py

# Cross-Encoder (pairwise relevance, BEDR dataset)
python train_cross_encoder.py
```

### Baseline Scripts

```bash
# ACRCNN supervised baseline (training + evaluation)
python deep_learning_train_with_logging.py
python deep-learning-test.py
```

## API Reproducibility Notes

### LLM (Stage 3 — Deep-AttacKG; All stages — BEDR)

| Parameter | Value |
|-----------|-------|
| Provider | DeepSeek |
| Model | `deepseek-chat` (DeepSeek-V3) |
| Base URL | `https://api.deepseek.com` |
| Temperature | 0.0 (deterministic decoding) |
| Response Format | `json_object` |
| Generation Period | December 2025 – April 2026 |

### Embedding (BEDR Pipeline)

| Parameter | Value |
|-----------|-------|
| Provider | Aliyun DashScope |
| Model | `text-embedding-v4` |
| Base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Dimensions | 768 |
| Generation Period | December 2025 – April 2026 |

### Random Seeds

All scripts use fixed random seeds (`random_state=42`, `seed=42`) for reproducible data splits and training initialization.

### On API-Based Reproducibility

Stages 1–2 (Bi-Encoder + Cross-Encoder) use locally deployed fixed-weight models and are fully deterministic. Stage 3 depends on the DeepSeek API, whose underlying model weights may be updated by the provider over time. The BEDR pipeline uses both DeepSeek (LLM) and DashScope (embedding) APIs. The methodology — including all prompts, decoding parameters, random seeds, pinned package versions, and intermediate outputs — is fully documented, ensuring that the pipeline is verifiable and the datasets are fully available. Intermediate augmentation outputs (CSV files) are included in the BEDR repository, allowing downstream steps to be executed without re-calling the LLM API.

## Data Sources

- **CTI-1002:** 1,002 CTI reports with ATT&CK annotations (801 train / 201 test)
- **BEDR:** Boundary Entropy-Driven Resampling dataset (21,453 samples, 679 classes)
- **TRAM:** MITRE TRAM public benchmark (multi_label.json)
- **ATT&CK KB:** MITRE ATT&CK v15 Enterprise (attack-pattern/*.json)

## License

MIT License

## DOI/Archive

To be registered with Zenodo upon manuscript acceptance.
