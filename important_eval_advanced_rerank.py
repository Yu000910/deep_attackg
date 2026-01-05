import os
import json
import glob
import torch
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
import re
import numpy as np

# ================= 配置 =================
# 1. 两个模型路径
BI_ENCODER_PATH = "./cti_model_20k_finetuned"
CROSS_ENCODER_PATH = "new_experiment_4/cti_reranker_final" # 刚才训练的新模型

# 2. 知识库与报告
TECHNIQUE_DIR = "/Users/nnn/Desktop/temp/博士毕业/第五篇/cti-master/enterprise-attack/attack-pattern/"
REPORTS_DIR = "generated_reports"

# 3. LLM
LLM_API_KEY = "" 
LLM_BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# 参数
TOP_K_RETRIEVE = 50   # 第一步召回 50 个
TOP_K_RERANK = 10     # 第二步 Cross-Encoder 选 10 个给 LLM

# ================= 1. 加载所有资源 =================
def load_system():
    print(">>> 🚀 Loading Advanced System...")
    
    # 1. Bi-Encoder
    bi_encoder = SentenceTransformer(BI_ENCODER_PATH)
    
    # 2. Cross-Encoder
    cross_encoder = CrossEncoder(CROSS_ENCODER_PATH)
    
    # 3. Knowledge Base (Filtering included)
    kb_texts = []
    kb_ids = []
    kb_info = {}
    kb_tokens = []
    
    json_files = glob.glob(os.path.join(TECHNIQUE_DIR, "*.json"))
    for fpath in json_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            for obj in content.get('objects', []):
                if obj.get('type') != 'attack-pattern': continue
                # 简单过滤: 只保留 Enterprise & 非废弃
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
        
    print(f"   ⚡ Encoding {len(kb_texts)} KB items with Bi-Encoder...")
    kb_embs = bi_encoder.encode(kb_texts, convert_to_tensor=True)
    bm25 = BM25Okapi(kb_tokens)
    
    return bi_encoder, cross_encoder, kb_embs, bm25, kb_ids, kb_texts, kb_info

# ================= 2. 三级推理逻辑 =================

def analyze_chunk_advanced(text, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info):
    # --- Stage 1: Bi-Encoder + BM25 Retrieval (Broad Search) ---
    candidates_idx = set()
    
    # Vector
    w_emb = bi_enc.encode(text, convert_to_tensor=True)
    hits = util.semantic_search(w_emb, kb_embs, top_k=TOP_K_RETRIEVE)[0]
    for hit in hits: candidates_idx.add(hit['corpus_id'])
        
    # BM25
    b_scores = bm25.get_scores(text.lower().split())
    b_top = np.argsort(b_scores)[-TOP_K_RETRIEVE:]
    for i in b_top: candidates_idx.add(i)
        
    if not candidates_idx: return []
    
    # --- Stage 2: Cross-Encoder Reranking (Precision Filtering) ---
    # 准备 Pairs: [[Query, Doc1], [Query, Doc2]...]
    cand_indices = list(candidates_idx)
    cross_inp = [[text, kb_texts[i]] for i in cand_indices]
    
    scores = cross_enc.predict(cross_inp)
    
    # 获取分数最高的 Top-K
    # argsort 默认是从小到大，所以要取最后面的
    top_k_indices = np.argsort(scores)[-TOP_K_RERANK:]
    
    final_candidates = []
    for i in top_k_indices:
        original_idx = cand_indices[i]
        final_candidates.append(kb_ids[original_idx])
        
    # --- Stage 3: LLM Selection (Final Judge) ---
    # 复用之前的 Listwise 逻辑
    return llm_listwise_select(text, final_candidates, kb_info)

# ... (复用之前的 llm_listwise_select, get_sliding_windows, evaluate 等辅助函数) ...
# 为了代码完整性，我这里简写，请将上一段代码的辅助函数复制过来

# ----------------- 必须复制的辅助函数 -----------------
def get_sliding_windows(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s) > 10]
    windows = []
    for i in range(0, len(sentences), 1): 
        chunk = " ".join(sentences[i : i + 3])
        if len(chunk) > 20: windows.append(chunk)
    return windows

def llm_listwise_select(chunk_text, candidates, kb_info):
    cand_str = ""
    for idx, cid in enumerate(candidates):
        info = kb_info.get(cid, {})
        cand_str += f"Option {idx}: [ID: {cid}] {info.get('name')}\n   Desc: {info.get('desc')[:200]}...\n"
    prompt = f"""
CTI Expert Task: Select ATT&CK techniques that strictly match the text.
Text: "{chunk_text}"
Options:
{cand_str}
Rules:
1. Select ONLY if the text describes specific malicious behavior matching the option.
2. If benign/generic, return empty.
Output JSON: {{ "indices": [0, 2] }}
"""
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={'type': 'json_object'},
            temperature=0.0
        )
        indices = json.loads(resp.choices[0].message.content).get("indices", [])
        return [candidates[i] for i in indices if 0 <= i < len(candidates)]
    except: return []
    
def get_parent_id(tid): return tid.split(".")[0] if "." in tid else tid
def check_match(pred, truth):
    if pred == truth: return True
    if get_parent_id(pred) == truth: return True
    if pred == get_parent_id(truth): return True
    return False

def detailed_report(name, preds, true_ids, kb_info):
    # ... (复用之前的代码) ...
    # 简写版：
    pred_list, true_list = list(preds), list(true_ids)
    tp = 0
    for p in pred_list:
        for t in true_list:
            if check_match(p, t): tp += 1; break
    
    covered = 0
    for t in true_list:
        for p in pred_list:
            if check_match(p, t): covered += 1; break
            
    p = tp/len(pred_list) if pred_list else 0
    r = covered/len(true_list) if true_list else 0
    f1 = 2*p*r/(p+r) if (p+r)>0 else 0
    print(f"{name}: F1={f1:.2%} (P={p:.2%}, R={r:.2%})")
    return p,r,f1
# ----------------------------------------------------

def run_advanced_test():
    bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info = load_system()
    
    report_files = glob.glob(os.path.join(REPORTS_DIR, "*.json"))[:200]
    avg_f1 = 0
    
    print(f"\n>>> 🚀 Starting Advanced Evaluation (Bi-Enc -> Cross-Enc -> LLM)...\n")
    
    for r_path in report_files:
        with open(r_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        text = data.get('clean_text', '')
        true_ids = set(data.get('actual_found_ids', []))
        if not text: continue
        
        windows = get_sliding_windows(text)
        all_preds = set()
        for w in tqdm(windows, desc="Scanning", leave=False):
            ids = analyze_chunk_advanced(w, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info)
            all_preds.update(ids)
            
        _,_,f1 = detailed_report(os.path.basename(r_path), all_preds, true_ids, kb_info)
        avg_f1 += f1
        
    print(f"🏆 Avg F1: {avg_f1/len(report_files):.2%}")

if __name__ == "__main__":
    run_advanced_test()