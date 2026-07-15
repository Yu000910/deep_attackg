"""
TRAM Evaluation Script — Strict-F1 and Hierarchy-Aware-F1 with proper one-to-one matching.

Key fixes:
- Uses greedy one-to-one matching for hierarchy-aware evaluation
  to prevent multiple predictions from matching the same ground truth
- Fixed random seeds

Usage: python run_tram_hierarchy_eval.py
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import glob
import random
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
import re
from collections import defaultdict

# ================= Reproducibility =================
random.seed(42)
np.random.seed(42)

# ================= Configuration =================
BI_ENCODER_PATH = "./cti_model_20k_finetuned"
CROSS_ENCODER_PATH = "./cti_reranker_final"
TECHNIQUE_DIR = "./attack-pattern"
TRAM_FILE_PATH = "./TRAM/multi_label.json"

LLM_API_KEY = "your-deepseek-api-key"
LLM_BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

TOP_K_RETRIEVE = 50
TOP_K_RERANK = 10
SAMPLE_SIZE = 50

# ================= 1. System Loading =================
def load_system():
    print(">>> Loading System for TRAM Hierarchy-Aware Eval...")
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
                tech_id = obj['external_references'][0]['external_id']
                name = obj['name']
                desc = obj['description']
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

# ================= 2. TRAM Data Loading =================
def load_and_sample_tram_data(filepath, sample_size=50):
    print(f">>> Loading TRAM data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    reports = defaultdict(lambda: {"text": "", "labels": set()})
    for item in data:
        title = item.get("doc_title", "Unknown_Doc")
        sentence = item.get("sentence", "").strip()
        if sentence:
            reports[title]["text"] += sentence + " "
        for label in item.get("labels", []):
            reports[title]["labels"].add(label)

    valid_reports = []
    for title, info in reports.items():
        if len(info["labels"]) > 0:
            valid_reports.append({
                "title": title,
                "text": info["text"].strip(),
                "true_ids": info["labels"]
            })

    random.seed(42)
    sampled = random.sample(valid_reports, min(sample_size, len(valid_reports)))
    print(f"   Sampled {len(sampled)} reports.")
    return sampled

# ================= 3. Inference Pipeline =================
def get_sliding_windows(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s) > 10]
    windows = []
    for i in range(0, len(sentences), 1):
        chunk = " ".join(sentences[i : i + 3])
        if len(chunk) > 20: windows.append(chunk)
    return windows if windows else [text]

def llm_listwise_select(chunk_text, candidates, kb_info):
    cand_str = ""
    for idx, cid in enumerate(candidates):
        info = kb_info.get(cid, {})
        cand_str += f"Option {idx}: [ID: {cid}] {info.get('name')}\n"
    prompt = f"""CTI Expert Task: Select ATT&CK techniques that strictly match the text.
Text: \"{chunk_text}\"
Options:
{cand_str}
Rules:
1. Relevance: Select ONLY if the text describes specific malicious behavior matching the option.
2. Specificity: Prefer sub-techniques over parent techniques when specific implementation details are present.
3. Benign Rejection: If benign/generic, return empty.
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
    except:
        return []

def analyze_report(text, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info):
    windows = get_sliding_windows(text)
    all_preds = set()
    for w in windows:
        candidates_idx = set()
        w_emb = bi_enc.encode(w, convert_to_tensor=True)
        hits = util.semantic_search(w_emb, kb_embs, top_k=TOP_K_RETRIEVE)[0]
        for hit in hits: candidates_idx.add(hit['corpus_id'])
        b_scores = bm25.get_scores(w.lower().split())
        b_top = np.argsort(b_scores)[-TOP_K_RETRIEVE:]
        for i in b_top: candidates_idx.add(i)
        if not candidates_idx: continue
        cand_indices = list(candidates_idx)
        cross_inp = [[w, kb_texts[i]] for i in cand_indices]
        scores = cross_enc.predict(cross_inp)
        top_k_indices = np.argsort(scores)[-TOP_K_RERANK:]
        final_candidates = [kb_ids[cand_indices[i]] for i in top_k_indices]
        confirmed_ids = llm_listwise_select(w, final_candidates, kb_info)
        all_preds.update(confirmed_ids)
    return all_preds

# ================= 4. Matching Functions =================
def get_parent_id(tid):
    return tid.split(".")[0] if "." in tid else tid

def hierarchy_matches(pred, truth):
    """Check if pred and truth match under hierarchy-aware rules."""
    if pred == truth:
        return True
    if get_parent_id(pred) == truth:
        return True
    if pred == get_parent_id(truth):
        return True
    return False

def compute_strict_metrics(all_preds, all_trues):
    """Standard strict exact-match Micro-F1 (set intersection)."""
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

def compute_hierarchy_metrics_one_to_one(all_preds, all_trues):
    """Hierarchy-aware Micro-F1 with maximum bipartite matching.
    Each prediction matches at most one ground truth, and vice versa.
    Uses greedy matching: exact matches first, then parent-child.
    """
    global_tp, global_fp, global_fn = 0, 0, 0
    for pred_set, true_set in zip(all_preds, all_trues):
        pred_list = list(pred_set)
        true_list = list(true_set)

        # Build bipartite match matrix
        match_matrix = {}
        for pi, p in enumerate(pred_list):
            for ti, t in enumerate(true_list):
                if hierarchy_matches(p, t):
                    match_matrix[(pi, ti)] = 1 if p == t else 0  # prefer exact

        # Prioritize exact matches, then parent-child
        matched_preds = set()
        matched_trues = set()

        # Pass 1: exact matches
        for (pi, ti), priority in match_matrix.items():
            if priority == 1 and pi not in matched_preds and ti not in matched_trues:
                matched_preds.add(pi)
                matched_trues.add(ti)

        # Pass 2: parent-child matches
        for (pi, ti), priority in match_matrix.items():
            if priority == 0 and pi not in matched_preds and ti not in matched_trues:
                matched_preds.add(pi)
                matched_trues.add(ti)

        global_tp += len(matched_preds)
        global_fp += len(pred_list) - len(matched_preds)
        global_fn += len(true_list) - len(matched_trues)

    micro_p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    micro_r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0
    return micro_p, micro_r, micro_f1, global_tp, global_fp, global_fn

# ================= 5. Main =================
def main():
    bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info = load_system()
    test_reports = load_and_sample_tram_data(TRAM_FILE_PATH, SAMPLE_SIZE)

    all_preds, all_trues = [], []
    print(f"\n>>> Running Zero-Shot Evaluation on {len(test_reports)} TRAM reports...")
    for report in tqdm(test_reports, desc="Evaluating"):
        preds = analyze_report(report["text"], bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info)
        all_preds.append(preds)
        all_trues.append(report["true_ids"])

    # Strict match
    sp, sr, sf1, stp, sfp, sfn = compute_strict_metrics(all_preds, all_trues)
    # Hierarchy-aware (one-to-one bipartite matching)
    hp, hr, hf1, htp, hfp, hfn = compute_hierarchy_metrics_one_to_one(all_preds, all_trues)

    print("\n" + "="*60)
    print("TRAM External Validation — Dual Metric Comparison (N=50)")
    print("="*60)
    print(f"{'Metric':<30} {'Strict':>12} {'Hierarchy-Aware':>15}")
    print("-"*57)
    print(f"{'TP':<30} {stp:>12} {htp:>15}")
    print(f"{'FP':<30} {sfp:>12} {hfp:>15}")
    print(f"{'FN':<30} {sfn:>12} {hfn:>15}")
    print(f"{'Micro-Precision':<30} {sp*100:>11.2f}% {hp*100:>14.2f}%")
    print(f"{'Micro-Recall':<30} {sr*100:>11.2f}% {hr*100:>14.2f}%")
    print(f"{'Micro-F1':<30} {sf1*100:>11.2f}% {hf1*100:>14.2f}%")
    print("="*60)
    print(f"\n  F1 improvement with hierarchy-aware: +{(hf1-sf1)*100:.2f}% absolute")
    print(f"  FN reduction: {sfn} -> {hfn} ({sfn-hfn} fewer false negatives)")


if __name__ == "__main__":
    main()
