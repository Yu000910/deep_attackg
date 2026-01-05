import os
import json
import glob
import torch
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import re

# ================= 配置 =================
MODEL_PATH = "./cti_model_20k_finetuned" 
TECHNIQUE_DIR = "/Users/nnn/Desktop/temp/博士毕业/第五篇/cti-master/enterprise-attack/attack-pattern/"
REPORTS_DIR = "generated_reports"

LLM_API_KEY = "" 
LLM_BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# 召回参数：宁滥勿缺，后续靠 LLM 过滤
TOP_K_VECTOR = 15
TOP_K_BM25 = 15

# ================= 1. 资源加载 =================

def load_resources():
    print(f">>> 🧠 Loading Global Hybrid System...")
    embedder = SentenceTransformer(MODEL_PATH)
    
    kb_texts = [] # for vector
    kb_tokens = [] # for bm25
    kb_ids = []
    kb_info = {} # id -> details
    
    json_files = glob.glob(os.path.join(TECHNIQUE_DIR, "*.json"))
    for fpath in json_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            for obj in content.get('objects', []):
                if obj.get('type') != 'attack-pattern': continue
                tech_id = None
                for ref in obj.get('external_references', []):
                    if ref.get('source_name') == 'mitre-attack':
                        tech_id = ref.get('external_id'); break
                if not tech_id: continue
                
                name = obj.get('name', '')
                desc = obj.get('description', '')
                
                # 向量化文本：侧重语义
                vec_text = f"{name}: {desc}"
                # BM25文本：侧重关键词
                bm25_text = f"{name} {desc} {tech_id}" # 把ID也加进去
                
                kb_ids.append(tech_id)
                kb_texts.append(vec_text)
                kb_tokens.append(bm25_text.lower().split())
                
                kb_info[tech_id] = {"name": name, "desc": desc[:250]} # 截断描述
        except: pass
        
    print(f"   ⚡ Encoding {len(kb_texts)} techniques...")
    kb_embs = embedder.encode(kb_texts, convert_to_tensor=True)
    
    print(f"   📚 Building BM25 Index...")
    bm25 = BM25Okapi(kb_tokens)
    
    return embedder, kb_embs, bm25, kb_ids, kb_info

# ================= 2. 核心逻辑: Listwise Reranking =================

def get_sliding_windows(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s) > 10]
    windows = []
    for i in range(0, len(sentences), 1): 
        chunk = " ".join(sentences[i : i + 3]) # 窗口大小3
        if len(chunk) > 20: windows.append(chunk)
    return windows

def llm_listwise_select(chunk_text, candidates, kb_info):
    """
    Listwise Reranking: 一次性把所有候选给 LLM，让它挑最好的。
    """
    if not candidates: return []
    
    # 构造候选清单
    cand_str = ""
    for idx, cid in enumerate(candidates):
        info = kb_info.get(cid, {})
        cand_str += f"Option {idx}: [ID: {cid}] {info.get('name')}\n   Desc: {info.get('desc')}...\n"
        
    prompt = f"""
You are a Cyber Threat Intelligence Expert.
Your Task: Identify if the text describes any of the listed ATT&CK techniques.

Text: "{chunk_text}"

Candidate Options:
{cand_str}

Instructions:
1. Compare the text against the options.
2. Select the Option IDs (e.g., Option 0, Option 2) that strictly match the MALICIOUS behavior in the text.
3. If the text is generic, legitimate, or matches none, return empty list.
4. Be precise: Do not select broad techniques if a specific one fits.

Output JSON: {{ "selected_indices": [0, 2] }}
"""
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={'type': 'json_object'},
            temperature=0.0
        )
        data = json.loads(resp.choices[0].message.content)
        indices = data.get("selected_indices", [])
        
        # 映射回 Tech ID
        selected_ids = []
        candidate_list = list(candidates)
        for i in indices:
            if 0 <= i < len(candidate_list):
                selected_ids.append(candidate_list[i])
        return selected_ids
    except: return []

def analyze_report_global(text, embedder, kb_embs, bm25, kb_ids, kb_info):
    windows = get_sliding_windows(text)
    final_findings = set()
    
    if not windows: return set()
    
    # 批量向量计算
    win_embs = embedder.encode(windows, convert_to_tensor=True)
    cos_scores = util.cos_sim(win_embs, kb_embs)
    
    for idx, window_text in enumerate(tqdm(windows, desc="   Scanning", leave=False)):
        # --- Stage 1: Global Recall (Union of Vector + BM25) ---
        candidates = set()
        
        # Vector Top-K
        v_scores, v_indices = torch.topk(cos_scores[idx], k=TOP_K_VECTOR)
        for i, s in zip(v_indices, v_scores):
            if s > 0.25: # 宽松阈值，全放进来
                candidates.add(kb_ids[i])
                
        # BM25 Top-K
        tokens = window_text.lower().split()
        b_scores = bm25.get_scores(tokens)
        b_indices = np.argsort(b_scores)[-TOP_K_BM25:]
        for i in b_indices:
            candidates.add(kb_ids[i])
            
        if not candidates: continue
        
        # --- Stage 2: Listwise Verification ---
        # 限制候选数量，防止 Context Window 爆炸 (最多取25个最相关的)
        # 这里简单处理，直接转list
        candidate_list = list(candidates)[:25] 
        
        confirmed_ids = llm_listwise_select(window_text, candidate_list, kb_info)
        final_findings.update(confirmed_ids)
            
    return final_findings

# ================= 3. 评估逻辑 =================

def get_parent_id(tid): return tid.split(".")[0] if "." in tid else tid
def check_match(pred, truth):
    if pred == truth: return True
    if get_parent_id(pred) == truth: return True
    if pred == get_parent_id(truth): return True
    return False

def detailed_report(report_name, pred_ids, true_ids, kb_info):
    pred_list = list(pred_ids)
    true_list = list(true_ids)
    
    tp_set = set()
    fp_set = set()
    fn_set = set()
    
    # TP & FP
    for p in pred_list:
        matched = False
        for t in true_list:
            if check_match(p, t):
                tp_set.add(p); matched = True; break
        if not matched: fp_set.add(p)
    # FN
    for t in true_list:
        covered = False
        for p in pred_list:
            if check_match(p, t): covered = True; break
        if not covered: fn_set.add(t)
        
    p = len(tp_set)/len(pred_list) if pred_list else 0.0
    r = (len(true_list)-len(fn_set))/len(true_list) if true_list else 0.0
    f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
    
    print(f"\n📝 {report_name}")
    print("   ✅ TP: {}".format([f'{i} {kb_info.get(i,{}).get("name")}' for i in tp_set]))
    print("   ❌ FP: {}".format([f'{i} {kb_info.get(i,{}).get("name")}' for i in fp_set]))
    print("   🔻 FN: {}".format([f'{i} {kb_info.get(i,{}).get("name")}' for i in fn_set]))
    print(f"   📊 F1: {f1:.2%} (P={p:.2%}, R={r:.2%})")
    
    return p, r, f1

# ================= 4. 主程序 =================

def run_global_test():
    embedder, kb_embs, bm25, kb_ids, kb_info = load_resources()
    
    report_files = glob.glob(os.path.join(REPORTS_DIR, "*.json"))
    report_files.sort()
    test_files = report_files[:5]
    
    avg_p, avg_r, avg_f1 = 0, 0, 0
    
    print(f"\n>>> 🚀 Starting Global Listwise Evaluation...\n")
    
    for r_idx, r_path in enumerate(test_files):
        with open(r_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            text = data.get('clean_text', '')
            true_ids = set(data.get('actual_found_ids', []))
            
        if not text: continue
        
        preds = analyze_report_global(text, embedder, kb_embs, bm25, kb_ids, kb_info)
        p, r, f1 = detailed_report(os.path.basename(r_path), preds, true_ids, kb_info)
        avg_p += p; avg_r += r; avg_f1 += f1

    print("\n" + "="*50)
    print(f"🏆 Avg F1: {avg_f1/len(test_files):.2%}")
    print(f"   Avg P:  {avg_p/len(test_files):.2%}")
    print(f"   Avg R:  {avg_r/len(test_files):.2%}")
    print("="*50)

if __name__ == "__main__":
    run_global_test()