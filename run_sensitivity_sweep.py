"""
参数敏感性分析 — 真实数据版本
K1 sweep: Stage1 retrieval recall + latency (无需LLM, 全测试集)
K2 sweep: Full pipeline precision + token cost (需要LLM, 子集)
输出: sensitivity_real_data.json → plot_sensitivity.py读取
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
from sklearn.model_selection import train_test_split

# ================= 配置 =================
BI_ENCODER_PATH = "./cti_model_20k_finetuned"
CROSS_ENCODER_PATH = "./cti_reranker_final"
TECHNIQUE_DIR = "./attack-pattern"
REPORTS_DIR = "./CTI_reports"

LLM_API_KEY = "your-deepseek-api-key"
LLM_BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

K1_VALUES = [10, 20, 30, 40, 50, 60, 80, 100]
K2_VALUES = [3, 5, 8, 10, 12, 15, 20]
K2_SUBSET_SIZE = 30  # K2 sweep用子集以节省API费用
BASE_K2 = 10
BASE_K1 = 50

# ================= 1. 加载系统 =================
def load_system():
    print(">>> Loading System...")
    bi_encoder = SentenceTransformer(BI_ENCODER_PATH)
    cross_encoder = CrossEncoder(CROSS_ENCODER_PATH)

    kb_texts, kb_ids, kb_tokens, kb_info = [], [], [], {}
    json_files = glob.glob(os.path.join(TECHNIQUE_DIR, "*.json"))
    for fpath in json_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            for obj in content.get('objects', []):
                if obj.get('type') != 'attack-pattern': continue
                is_ent = any(p.get('kill_chain_name')=='mitre-attack' for p in obj.get('kill_chain_phases',[]))
                if not is_ent or obj.get('x_mitre_deprecated') or obj.get('revoked'): continue
                tech_id = None
                for ref in obj.get('external_references', []):
                    if ref.get('source_name') == 'mitre-attack':
                        tech_id = ref.get('external_id')
                        break
                if not tech_id: continue
                name = obj['name']
                desc = obj.get('description', '')
                text = f"{name}: {desc}"
                kb_ids.append(tech_id)
                kb_texts.append(text)
                kb_tokens.append(f"{name} {desc} {tech_id}".lower().split())
                kb_info[tech_id] = {"name": name, "desc": desc}
        except: pass

    print(f"   Loaded {len(kb_texts)} techniques.")
    kb_embs = bi_encoder.encode(kb_texts, convert_to_tensor=True, show_progress_bar=True)
    bm25 = BM25Okapi(kb_tokens)
    return bi_encoder, cross_encoder, kb_embs, bm25, kb_ids, kb_texts, kb_info

def get_sliding_windows(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s) > 10]
    windows = []
    for i in range(0, len(sentences), 1):
        chunk = " ".join(sentences[i : i + 3])
        if len(chunk) > 20: windows.append(chunk)
    return windows if windows else [text]

def get_parent_id(tid):
    return tid.split(".")[0] if "." in tid else tid

def check_match(pred, truth):
    if pred == truth: return True
    if get_parent_id(pred) == truth: return True
    if pred == get_parent_id(truth): return True
    return False

# ================= 2. K1 Sweep (仅Stage1, 无需LLM) =================
def compute_recall_at_k1(text, true_ids, bi_enc, kb_embs, bm25, kb_ids, k1):
    """计算Recall@K1: 真值中有多少出现在Top-K1候选中"""
    windows = get_sliding_windows(text)
    all_candidates = set()

    t0 = time.perf_counter()
    for w in windows:
        candidates_idx = set()
        w_emb = bi_enc.encode(w, convert_to_tensor=True)
        hits = util.semantic_search(w_emb, kb_embs, top_k=k1)[0]
        for hit in hits: candidates_idx.add(hit['corpus_id'])
        b_scores = bm25.get_scores(w.lower().split())
        b_top = np.argsort(b_scores)[-k1:]
        for i in b_top: candidates_idx.add(i)
        for ci in candidates_idx:
            all_candidates.add(kb_ids[ci])
    elapsed = (time.perf_counter() - t0) * 1000  # ms

    # Recall: fraction of true_ids covered
    covered = 0
    for t in true_ids:
        if any(check_match(c, t) for c in all_candidates):
            covered += 1
    recall = covered / len(true_ids) if true_ids else 0
    return recall, elapsed

def run_k1_sweep(test_texts, test_labels, bi_enc, kb_embs, bm25, kb_ids):
    print(f"\n>>> K1 Sweep: {len(K1_VALUES)} values × {len(test_texts)} reports...")
    results = []
    for k1 in K1_VALUES:
        recalls, latencies = [], []
        for text, labels in tqdm(zip(test_texts, test_labels), total=len(test_texts), desc=f"K1={k1}"):
            r, lat = compute_recall_at_k1(text, labels, bi_enc, kb_embs, bm25, kb_ids, k1)
            recalls.append(r)
            latencies.append(lat)
        results.append({
            "k1": k1,
            "mean_recall": round(np.mean(recalls) * 100, 2),
            "std_recall": round(np.std(recalls) * 100, 2),
            "mean_latency_ms": round(np.mean(latencies), 2),
            "std_latency_ms": round(np.std(latencies), 2)
        })
        print(f"  K1={k1:3d}: Recall={results[-1]['mean_recall']:.1f}% ± {results[-1]['std_recall']:.1f}%, "
              f"Latency={results[-1]['mean_latency_ms']:.1f} ± {results[-1]['std_latency_ms']:.1f} ms")
    return results

# ================= 3. K2 Sweep (完整管线, 需要LLM) =================
def llm_listwise_select(chunk_text, candidates, kb_info):
    cand_str = ""
    for idx, cid in enumerate(candidates):
        info = kb_info.get(cid, {})
        cand_str += f"Option {idx}: [ID: {cid}] {info.get('name')}\n"
    # 估算token数 (粗略: 每个候选~30 tokens + prompt overhead ~150 tokens)
    est_tokens = 150 + len(candidates) * 30

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
        usage = resp.usage
        actual_tokens = usage.prompt_tokens if usage else est_tokens
        return [candidates[i] for i in indices if 0 <= i < len(candidates)], actual_tokens
    except:
        return [], est_tokens

def analyze_report_k2(text, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info, k2):
    """完整三阶段, 使用指定K2"""
    windows = get_sliding_windows(text)
    all_preds = set()
    total_tokens = 0

    for w in windows:
        candidates_idx = set()
        w_emb = bi_enc.encode(w, convert_to_tensor=True)
        hits = util.semantic_search(w_emb, kb_embs, top_k=BASE_K1)[0]
        for hit in hits: candidates_idx.add(hit['corpus_id'])
        b_scores = bm25.get_scores(w.lower().split())
        b_top = np.argsort(b_scores)[-BASE_K1:]
        for i in b_top: candidates_idx.add(i)
        if not candidates_idx: continue

        cand_indices = list(candidates_idx)
        cross_inp = [[w, kb_texts[i]] for i in cand_indices]
        scores = cross_enc.predict(cross_inp)
        top_k_indices = np.argsort(scores)[-k2:]
        final_candidates = [kb_ids[cand_indices[i]] for i in top_k_indices]

        if final_candidates:
            confirmed, tokens = llm_listwise_select(w, final_candidates, kb_info)
            all_preds.update(confirmed)
            total_tokens += tokens

    return all_preds, total_tokens

def run_k2_sweep(test_texts, test_labels, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info):
    print(f"\n>>> K2 Sweep: {len(K2_VALUES)} values × {len(test_texts)} reports...")
    results = []
    for k2 in K2_VALUES:
        all_preds, all_trues = [], []
        total_tokens = 0

        for text, labels in tqdm(zip(test_texts, test_labels), total=len(test_texts), desc=f"K2={k2}"):
            preds, tokens = analyze_report_k2(
                text, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info, k2)
            all_preds.append(preds)
            all_trues.append(labels)
            total_tokens += tokens

        # 计算Micro-Precision
        tp, fp = 0, 0
        for pred_set, true_set in zip(all_preds, all_trues):
            pred_list = list(pred_set)
            true_list = list(true_set)
            for p in pred_list:
                if any(check_match(p, t) for t in true_list): tp += 1
                else: fp += 1

        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        avg_tokens = total_tokens / len(test_texts)

        results.append({
            "k2": k2,
            "precision": round(precision, 2),
            "avg_tokens_per_report": round(avg_tokens, 0),
            "tp": tp, "fp": fp
        })
        print(f"  K2={k2:2d}: Precision={precision:.1f}%, Avg Tokens/Report={avg_tokens:.0f}")

    return results

# ================= 4. Main =================
def main():
    bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info = load_system()

    # 加载测试集 (与主实验一致)
    json_files = sorted(glob.glob(os.path.join(REPORTS_DIR, "*_ground_truth.json")))
    all_texts, all_labels = [], []
    for fpath in json_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text = data.get('clean_text', '')
            labels = data.get('actual_found_ids', [])
            if text and labels:
                all_texts.append(text)
                all_labels.append(labels)
        except: pass

    indices = list(range(len(all_texts)))
    # Use published test split
    if os.path.exists("test_split.json"):
        with open("test_split.json", 'r') as f:
            split_data = json.load(f)
        test_files = set(split_data["test_files"])
        # Map filenames to indices
        all_fnames_sweep = [os.path.basename(fp) for fp in json_files]
        test_idx = [i for i, fn in enumerate(all_fnames_sweep) if fn in test_files]
        print(f"   Test set (from split): {len(test_idx)} reports")
    else:
        print("   [WARN] test_split.json not found, falling back to re-split")
        _, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    test_texts = [all_texts[i] for i in test_idx]
    test_labels = [all_labels[i] for i in test_idx]
    print(f"Test set: {len(test_texts)} reports")

    # ==== K1 Sweep (全测试集, 无LLM) ====
    k1_results = run_k1_sweep(test_texts, test_labels, bi_enc, kb_embs, bm25, kb_ids)

    # ==== K2 Sweep (子集, 有LLM) ====
    import random
    random.seed(42)
    subset_idx = random.sample(range(len(test_texts)), min(K2_SUBSET_SIZE, len(test_texts)))
    k2_texts = [test_texts[i] for i in subset_idx]
    k2_labels = [test_labels[i] for i in subset_idx]
    print(f"\nK2 Sweep subset: {len(k2_texts)} reports")

    k2_results = run_k2_sweep(k2_texts, k2_labels, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info)

    # 保存
    output = {"k1_sweep": k1_results, "k2_sweep": k2_results,
              "k1_values": K1_VALUES, "k2_values": K2_VALUES,
              "k1_test_size": len(test_texts), "k2_subset_size": len(k2_texts)}
    with open("sensitivity_real_data.json", 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to sensitivity_real_data.json")

if __name__ == "__main__":
    main()
