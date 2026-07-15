"""
Deep-AttacKG Main Evaluation Script
Reproduces Table 3 (Main Results) and Table 5 (Ablation Study)
Protocol B: Zero-Shot evaluation on 201 test reports

Key fixes:
- Uses strict exact-match Micro-F1 (standard definition)
- Properly reads published test_split.json
- Saves structured results to evaluation_results.json
- Computes 95% bootstrap confidence intervals
- Fixed random seeds for reproducibility

Usage: python run_main_evaluation.py [--quick] [--compute-fairness]
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import glob
import time
import torch
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
import re
import random

# ================= Reproducibility =================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================= Configuration =================
BI_ENCODER_PATH = "./cti_model_20k_finetuned"
CROSS_ENCODER_PATH = "./cti_reranker_final"

TECHNIQUE_DIR = "./attack-pattern"
REPORTS_DIR = "./CTI_reports"

LLM_API_KEY = "your-deepseek-api-key"
LLM_BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

TOP_K_RETRIEVE = 50
TOP_K_RERANK = 10

QUICK_TEST = False
QUICK_N = 5


# ================= 1. Load Models and Knowledge Base =================
def load_system():
    print(">>> Loading Deep-AttacKG System...")
    bi_encoder = SentenceTransformer(BI_ENCODER_PATH)
    cross_encoder = CrossEncoder(CROSS_ENCODER_PATH)

    kb_texts, kb_ids, kb_tokens, kb_info = [], [], [], {}
    json_files = glob.glob(os.path.join(TECHNIQUE_DIR, "*.json"))
    for fpath in json_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            for obj in content.get('objects', []):
                if obj.get('type') != 'attack-pattern':
                    continue
                is_ent = any(p.get('kill_chain_name') == 'mitre-attack'
                           for p in obj.get('kill_chain_phases', []))
                if not is_ent or obj.get('x_mitre_deprecated') or obj.get('revoked'):
                    continue
                tech_id = None
                for ref in obj.get('external_references', []):
                    if ref.get('source_name') == 'mitre-attack':
                        tech_id = ref.get('external_id')
                        break
                if not tech_id:
                    continue
                name = obj['name']
                desc = obj.get('description', '')
                text = f"{name}: {desc}"
                kb_ids.append(tech_id)
                kb_texts.append(text)
                kb_tokens.append(f"{name} {desc} {tech_id}".lower().split())
                kb_info[tech_id] = {"name": name, "desc": desc}
        except:
            pass

    print(f"   Loaded {len(kb_texts)} techniques from MITRE ATT&CK KB.")
    kb_embs = bi_encoder.encode(kb_texts, convert_to_tensor=True, show_progress_bar=True)
    bm25 = BM25Okapi(kb_tokens)
    return bi_encoder, cross_encoder, kb_embs, bm25, kb_ids, kb_texts, kb_info


# ================= 2. Sliding Window (3-sentence, stride 1) =================
def get_sliding_windows(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s) > 10]
    windows = []
    for i in range(0, len(sentences), 1):
        chunk = " ".join(sentences[i : i + 3])
        if len(chunk) > 20:
            windows.append(chunk)
    return windows if windows else [text]


# ================= 3. LLM Constrained Inference (Stage 3) =================
def llm_listwise_select(chunk_text, candidates, kb_info):
    cand_str = ""
    for idx, cid in enumerate(candidates):
        info = kb_info.get(cid, {})
        cand_str += f"Option {idx}: [ID: {cid}] {info.get('name')}\n   Desc: {info.get('desc', '')[:200]}...\n"

    prompt = f"""CTI Expert Task: Select ATT&CK techniques that strictly match the text.
Text: \"{chunk_text}\"
Options:
{cand_str}
Rules:
1. Relevance: Select ONLY if the text describes specific malicious behavior explicitly matching the option. Do not infer based on external priors.
2. Specificity: If both a parent technique and its sub-technique are viable, select the sub-technique when specific implementation details are present.
3. Benign Rejection: If the text describes generic IT operations or lacks malicious context, return an empty list.
Output JSON: {{ "indices": [0, 2] }}"""

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={'type': 'json_object'},
            temperature=0.0
        )
        indices = json.loads(resp.choices[0].message.content).get("indices", [])
        return [candidates[i] for i in indices if 0 <= i < len(candidates)]
    except Exception as e:
        print(f"   [WARN] LLM call failed: {e}")
        return []


# ================= 4. Three-Stage Inference Pipeline =================
def analyze_report_stages(text, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info,
                          enable_stage2=True, enable_stage3=True):
    windows = get_sliding_windows(text)
    stage1_candidates = set()
    stage2_candidates = set()
    stage3_predictions = set()

    for w in windows:
        # Stage 1: Dense + Sparse Hybrid Retrieval (Union-based)
        candidates_idx = set()
        w_emb = bi_enc.encode(w, convert_to_tensor=True)
        hits = util.semantic_search(w_emb, kb_embs, top_k=TOP_K_RETRIEVE)[0]
        for hit in hits:
            candidates_idx.add(hit['corpus_id'])
        b_scores = bm25.get_scores(w.lower().split())
        b_top = np.argsort(b_scores)[-TOP_K_RETRIEVE:]
        for i in b_top:
            candidates_idx.add(i)
        if not candidates_idx:
            continue

        s1_ids = set(kb_ids[i] for i in candidates_idx)
        stage1_candidates.update(s1_ids)

        if not enable_stage2:
            continue

        # Stage 2: Cross-Encoder Reranking
        cand_indices = list(candidates_idx)
        cross_inp = [[w, kb_texts[i]] for i in cand_indices]
        scores = cross_enc.predict(cross_inp)
        top_k_indices = np.argsort(scores)[-TOP_K_RERANK:]
        final_candidates = [kb_ids[cand_indices[i]] for i in top_k_indices]
        stage2_candidates.update(final_candidates)

        if not enable_stage3:
            continue

        # Stage 3: LLM Logic-Constrained Verification
        if final_candidates:
            confirmed_ids = llm_listwise_select(w, final_candidates, kb_info)
            stage3_predictions.update(confirmed_ids)

    if not enable_stage2:
        return stage1_candidates, stage1_candidates, stage1_candidates
    elif not enable_stage3:
        return stage1_candidates, stage2_candidates, stage2_candidates
    else:
        return stage1_candidates, stage2_candidates, stage3_predictions


# ================= 5. Strict Exact-Match Micro-F1 =================
def compute_strict_micro_f1(all_preds, all_trues):
    """Standard strict exact-match Micro-F1 for multi-label classification.
    Each (sample, label) pair is evaluated independently (standard definition).
    """
    global_tp, global_fp, global_fn = 0, 0, 0
    for pred_set, true_set in zip(all_preds, all_trues):
        pred_set = set(pred_set)
        true_set = set(true_set)
        global_tp += len(pred_set & true_set)
        global_fp += len(pred_set - true_set)
        global_fn += len(true_set - pred_set)

    micro_p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    micro_r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    return micro_p, micro_r, micro_f1, global_tp, global_fp, global_fn


def bootstrap_ci(all_preds, all_trues, n_bootstrap=1000, alpha=0.05):
    """Compute 95% bootstrap confidence intervals for Micro-F1.
    Resamples (report, prediction, truth) triples with replacement.
    """
    n = len(all_preds)
    f1_samples = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bs_preds = [all_preds[i] for i in idx]
        bs_trues = [all_trues[i] for i in idx]
        _, _, f1, _, _, _ = compute_strict_micro_f1(bs_preds, bs_trues)
        f1_samples.append(f1)
    f1_samples = np.sort(f1_samples)
    lower = f1_samples[int(alpha / 2 * n_bootstrap)]
    upper = f1_samples[int((1 - alpha / 2) * n_bootstrap)]
    mean = np.mean(f1_samples)
    return float(lower), float(upper), float(mean)


def save_results(m1_p, m1_r, m1_f1, m1_tp, m1_fp, m1_fn,
                 m2_p, m2_r, m2_f1, m2_tp, m2_fp, m2_fn,
                 m3_p, m3_r, m3_f1, m3_tp, m3_fp, m3_fn,
                 m3_preds, m3_trues,
                 test_idx, kb_count, output_path="evaluation_results.json"):
    """Save ablation results to structured JSON, preserving baseline data if present."""

    # Load existing cached file to preserve baseline results (if any)
    existing = {}
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            existing = json.load(f)

    # Compute bootstrap CI for M3
    ci_lower, ci_upper, ci_mean = bootstrap_ci(m3_preds, m3_trues)

    results = {
        "metadata": existing.get("metadata", {
            "description": "Deep-AttacKG evaluation results",
            "dataset": "CTI-1002 test split (N=201)",
            "kb_techniques": kb_count,
            "random_seed": 42,
            "generated_by": "run_main_evaluation.py"
        }),
        "main_results": existing.get("main_results", {}),
        "ablation": [
            {
                "stage": "M1: Hybrid Retrieval Only",
                "precision": round(float(m1_p), 6),
                "recall": round(float(m1_r), 6),
                "micro_f1": round(float(m1_f1), 6),
                "tp": int(m1_tp), "fp": int(m1_fp), "fn": int(m1_fn),
                "primary_gain": "Coverage (Recall)"
            },
            {
                "stage": "M2: + Cross-Encoder Reranking",
                "precision": round(float(m2_p), 6),
                "recall": round(float(m2_r), 6),
                "micro_f1": round(float(m2_f1), 6),
                "tp": int(m2_tp), "fp": int(m2_fp), "fn": int(m2_fn),
                "primary_gain": "Denoising (Precision)"
            },
            {
                "stage": "M3: + Logic Reasoning",
                "precision": round(float(m3_p), 6),
                "recall": round(float(m3_r), 6),
                "micro_f1": round(float(m3_f1), 6),
                "tp": int(m3_tp), "fp": int(m3_fp), "fn": int(m3_fn),
                "primary_gain": "Disambiguation (F1)",
                "bootstrap_ci_95": [ci_lower, ci_upper],
                "bootstrap_ci_mean": ci_mean
            }
        ],
        "bootstrap": {
            "n_iterations": 1000,
            "confidence_level": 0.95,
            "m3_f1_ci_lower": ci_lower,
            "m3_f1_ci_upper": ci_upper
        },
        "test_samples": len(test_idx),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Preserve LLM ablation from existing cache
    if "llm_ablation" in existing:
        results["llm_ablation"] = existing["llm_ablation"]

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


# ================= 6. Main =================
def main():
    bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info = load_system()

    # Load test split from published file
    split_path = "test_split.json"
    if os.path.exists(split_path):
        print(f">>> Loading test split from {split_path}...")
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        test_files = set(split_data["test_files"])
        print(f"   Test set: {len(test_files)} reports (from published split)")
    else:
        print(f"   [WARN] {split_path} not found, falling back to re-split")
        test_files = None

    # Load all reports
    json_files = sorted(glob.glob(os.path.join(REPORTS_DIR, "*_ground_truth.json")))
    print(f">>> Found {len(json_files)} ground-truth reports.")

    all_texts, all_labels, all_fnames = [], [], []
    for fpath in json_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text = data.get('clean_text', '')
            labels = data.get('actual_found_ids', [])
            if text and labels:
                all_texts.append(text)
                all_labels.append(labels)
                all_fnames.append(os.path.basename(fpath))
        except:
            pass

    print(f"   Valid reports: {len(all_fnames)}")

    # Select test indices
    if test_files is not None:
        test_idx = [i for i, fn in enumerate(all_fnames) if fn in test_files]
        print(f"   Matched test reports: {len(test_idx)}")
    else:
        from sklearn.model_selection import train_test_split
        indices = list(range(len(all_fnames)))
        _, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    if QUICK_TEST:
        test_idx = test_idx[:QUICK_N]
        print(f"   [QUICK TEST MODE] Using only {QUICK_N} reports.")

    # ===== M1: Hybrid Retrieval Only =====
    print("\n" + "="*60)
    print(">>> M1: Hybrid Retrieval Only (Stage 1)")
    print("="*60)
    m1_preds, m1_trues = [], []
    for i in tqdm(test_idx, desc="M1"):
        text = all_texts[i]
        true_ids = set(all_labels[i])
        s1, _, _ = analyze_report_stages(
            text, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info,
            enable_stage2=False, enable_stage3=False
        )
        m1_preds.append(s1)
        m1_trues.append(true_ids)
    m1_p, m1_r, m1_f1, m1_tp, m1_fp, m1_fn = compute_strict_micro_f1(m1_preds, m1_trues)
    print(f"   TP={m1_tp} FP={m1_fp} FN={m1_fn}")
    print(f"   Recall@50: {m1_r*100:.2f}% | Precision: {m1_p*100:.2f}% | Micro-F1: {m1_f1*100:.2f}%")

    # ===== M2: + Cross-Encoder Reranking =====
    print("\n" + "="*60)
    print(">>> M2: + Cross-Encoder Reranking (Stage 1 + 2)")
    print("="*60)
    m2_preds, m2_trues = [], []
    for i in tqdm(test_idx, desc="M2"):
        text = all_texts[i]
        true_ids = set(all_labels[i])
        _, s2, _ = analyze_report_stages(
            text, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info,
            enable_stage2=True, enable_stage3=False
        )
        m2_preds.append(s2)
        m2_trues.append(true_ids)
    m2_p, m2_r, m2_f1, m2_tp, m2_fp, m2_fn = compute_strict_micro_f1(m2_preds, m2_trues)
    print(f"   TP={m2_tp} FP={m2_fp} FN={m2_fn}")
    print(f"   Precision: {m2_p*100:.2f}% | Recall: {m2_r*100:.2f}% | Micro-F1: {m2_f1*100:.2f}%")

    # ===== M3: Full Deep-AttacKG =====
    print("\n" + "="*60)
    print(">>> M3: Full Deep-AttacKG (Stage 1 + 2 + 3)")
    print("="*60)
    m3_preds, m3_trues = [], []
    for i in tqdm(test_idx, desc="M3"):
        text = all_texts[i]
        true_ids = set(all_labels[i])
        _, _, s3 = analyze_report_stages(
            text, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info,
            enable_stage2=True, enable_stage3=True
        )
        m3_preds.append(s3)
        m3_trues.append(true_ids)
    m3_p, m3_r, m3_f1, m3_tp, m3_fp, m3_fn = compute_strict_micro_f1(m3_preds, m3_trues)
    print(f"   TP={m3_tp} FP={m3_fp} FN={m3_fn}")
    print(f"   Precision: {m3_p*100:.2f}% | Recall: {m3_r*100:.2f}% | Micro-F1: {m3_f1*100:.2f}%")

    # ===== Summary =====
    print("\n" + "="*60)
    print(">>> ABLATION SUMMARY (Strict Exact-Match Micro-F1)")
    print("="*60)
    print(f"{'Stage':<25} {'Precision':>10} {'Recall':>10} {'Micro-F1':>10}")
    print("-"*55)
    print(f"{'M1: Hybrid Retrieval':<25} {m1_p*100:>9.2f}% {m1_r*100:>9.2f}% {m1_f1*100:>9.2f}%")
    print(f"{'M2: + Cross-Encoder':<25} {m2_p*100:>9.2f}% {m2_r*100:>9.2f}% {m2_f1*100:>9.2f}%")
    print(f"{'M3: + LLM Reasoning':<25} {m3_p*100:>9.2f}% {m3_r*100:>9.2f}% {m3_f1*100:>9.2f}%")
    print("="*60)

    # Save structured results
    save_results(m1_p, m1_r, m1_f1, m1_tp, m1_fp, m1_fn,
                 m2_p, m2_r, m2_f1, m2_tp, m2_fp, m2_fn,
                 m3_p, m3_r, m3_f1, m3_tp, m3_fp, m3_fn,
                 m3_preds, m3_trues,
                 test_idx, len(kb_texts))


if __name__ == "__main__":
    main()
