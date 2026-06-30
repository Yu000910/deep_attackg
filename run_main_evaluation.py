"""
Deep-AttacKG 主评估脚本
复现论文 Table 3 (Main Results) 和 Table 4 (Ablation Study)
Protocol B: 201 条测试集上 Zero-Shot 评估

用法: python run_main_evaluation.py
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import glob
import torch
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
import re
from sklearn.model_selection import train_test_split

# ================= 配置 =================
BI_ENCODER_PATH = "./cti_model_20k_finetuned"
CROSS_ENCODER_PATH = "./cti_reranker_final"

TECHNIQUE_DIR = "./attack-pattern"
REPORTS_DIR = "./CTI_reports"

LLM_API_KEY = "your-deepseek-api-key"
LLM_BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

TOP_K_RETRIEVE = 50   # Stage 1: 混合检索召回数
TOP_K_RERANK = 10     # Stage 2: Cross-Encoder精选数

# 测试模式: True=只跑少量报告快速验证, False=完整评估
QUICK_TEST = False
QUICK_N = 5  # 快速测试时跑几条

# ================= 1. 加载模型和知识库 =================
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


# ================= 2. 文本滑动窗口 =================
def get_sliding_windows(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s) > 10]
    windows = []
    for i in range(0, len(sentences), 1):
        chunk = " ".join(sentences[i : i + 3])
        if len(chunk) > 20:
            windows.append(chunk)
    return windows if windows else [text]  # 兜底


# ================= 3. LLM 约束推理 (Stage 3) =================
def llm_listwise_select(chunk_text, candidates, kb_info):
    cand_str = ""
    for idx, cid in enumerate(candidates):
        info = kb_info.get(cid, {})
        cand_str += f"Option {idx}: [ID: {cid}] {info.get('name')}\n   Desc: {info.get('desc', '')[:200]}...\n"

    prompt = f"""CTI Expert Task: Select ATT&CK techniques that strictly match the text.
Text: "{chunk_text}"
Options:
{cand_str}
Rules:
1. Select ONLY if the text describes specific malicious behavior matching the option.
2. If benign/generic, return empty.
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


# ================= 4. 三阶段推理 (核心管线) =================
def analyze_report_stages(text, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info,
                          enable_stage2=True, enable_stage3=True):
    """
    执行Deep-AttacKG推理。
    enable_stage2=False → 仅Stage1 (Hybrid Retrieval, M1)
    enable_stage3=False → Stage1+2 (M2, 用Cross-Encoder直接取Top-K)
    enable_stage3=True  → 完整三阶段 (M3)
    """
    windows = get_sliding_windows(text)

    # 各阶段累计结果
    stage1_candidates = set()   # M1: 所有候选 (Recall Pool)
    stage2_candidates = set()   # M2: Cross-Encoder精选
    stage3_predictions = set()  # M3: LLM最终选择

    for w in windows:
        # --- Stage 1: Dense + Sparse Hybrid Retrieval ---
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

        # --- Stage 2: Cross-Encoder Reranking ---
        cand_indices = list(candidates_idx)
        cross_inp = [[w, kb_texts[i]] for i in cand_indices]
        scores = cross_enc.predict(cross_inp)
        top_k_indices = np.argsort(scores)[-TOP_K_RERANK:]

        final_candidates = [kb_ids[cand_indices[i]] for i in top_k_indices]
        stage2_candidates.update(final_candidates)

        if not enable_stage3:
            continue

        # --- Stage 3: LLM Logic-Constrained Verification ---
        if final_candidates:
            confirmed_ids = llm_listwise_select(w, final_candidates, kb_info)
            stage3_predictions.update(confirmed_ids)

    if not enable_stage2:
        return stage1_candidates, stage1_candidates, stage1_candidates
    elif not enable_stage3:
        return stage1_candidates, stage2_candidates, stage2_candidates
    else:
        return stage1_candidates, stage2_candidates, stage3_predictions


# ================= 5. 匹配规则 (Parent-Child tolerant) =================
def get_parent_id(tid):
    return tid.split(".")[0] if "." in tid else tid

def check_match(pred, truth):
    """父子宽容匹配"""
    if pred == truth:
        return True
    if get_parent_id(pred) == truth:
        return True
    if pred == get_parent_id(truth):
        return True
    return False


# ================= 6. 全局 Micro-F1 计算 =================
def compute_global_metrics(all_preds, all_trues):
    """基于全局 TP/FP/FN 计算 Micro-F1"""
    global_tp, global_fp, global_fn = 0, 0, 0

    for pred_set, true_set in zip(all_preds, all_trues):
        pred_list = list(pred_set)
        true_list = list(true_set)

        # TP & FP
        for p in pred_list:
            matched = False
            for t in true_list:
                if check_match(p, t):
                    global_tp += 1
                    matched = True
                    break
            if not matched:
                global_fp += 1

        # FN
        for t in true_list:
            covered = False
            for p in pred_list:
                if check_match(p, t):
                    covered = True
                    break
            if not covered:
                global_fn += 1

    micro_p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    micro_r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    return micro_p, micro_r, micro_f1, global_tp, global_fp, global_fn


# ================= 7. 主程序 =================
def main():
    # 加载系统
    bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info = load_system()

    # 加载报告 (使用与deep-learning-test.py相同的80/20分割)
    json_files = sorted(glob.glob(os.path.join(REPORTS_DIR, "*_ground_truth.json")))
    print(f"\n>>> Found {len(json_files)} ground-truth reports.")

    # 提取所有报告ID用于分割
    all_texts = []
    all_labels = []
    valid_files = []
    for fpath in json_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text = data.get('clean_text', '')
            labels = data.get('actual_found_ids', [])
            if text and labels:
                all_texts.append(text)
                all_labels.append(labels)
                valid_files.append(fpath)
        except:
            pass

    print(f"   Valid reports: {len(valid_files)}")

    # 使用完全相同的分割 (random_state=42, test_size=0.2)
    indices = list(range(len(valid_files)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    print(f"   Train set: {len(train_idx)}, Test set: {len(test_idx)}")

    if QUICK_TEST:
        test_idx = test_idx[:QUICK_N]
        print(f"   [QUICK TEST MODE] Using only {QUICK_N} reports.")

    # ===== 消融实验: M1 (仅Hybrid Retrieval) =====
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

    m1_p, m1_r, m1_f1, m1_tp, m1_fp, m1_fn = compute_global_metrics(m1_preds, m1_trues)
    print(f"   TP={m1_tp} FP={m1_fp} FN={m1_fn}")
    print(f"   Recall@50: {m1_r*100:.2f}% | Precision: {m1_p*100:.2f}% | Micro-F1: {m1_f1*100:.2f}%")

    # ===== 消融实验: M2 (Stage 1 + 2, Cross-Encoder Reranking) =====
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

    m2_p, m2_r, m2_f1, m2_tp, m2_fp, m2_fn = compute_global_metrics(m2_preds, m2_trues)
    print(f"   TP={m2_tp} FP={m2_fp} FN={m2_fn}")
    print(f"   Precision: {m2_p*100:.2f}% | Recall: {m2_r*100:.2f}% | Micro-F1: {m2_f1*100:.2f}%")

    # ===== 完整三阶段: M3 (Stage 1 + 2 + 3) =====
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

    m3_p, m3_r, m3_f1, m3_tp, m3_fp, m3_fn = compute_global_metrics(m3_preds, m3_trues)
    print(f"   TP={m3_tp} FP={m3_fp} FN={m3_fn}")
    print(f"   Precision: {m3_p*100:.2f}% | Recall: {m3_r*100:.2f}% | Micro-F1: {m3_f1*100:.2f}%")

    # ===== 汇总 =====
    print("\n" + "="*60)
    print(">>> ABLATION SUMMARY")
    print("="*60)
    print(f"{'Stage':<25} {'Precision':>10} {'Recall':>10} {'Micro-F1':>10}")
    print("-"*55)
    print(f"{'M1: Hybrid Retrieval':<25} {m1_p*100:>9.2f}% {m1_r*100:>9.2f}% {m1_f1*100:>9.2f}%")
    print(f"{'M2: + Cross-Encoder':<25} {m2_p*100:>9.2f}% {m2_r*100:>9.2f}% {m2_f1*100:>9.2f}%")
    print(f"{'M3: + LLM Reasoning':<25} {m3_p*100:>9.2f}% {m3_r*100:>9.2f}% {m3_f1*100:>9.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
