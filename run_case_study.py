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

# ================= 🔧 专用配置 =================
# 1. 基础路径 (使用绝对路径以防出错)
BASE_DIR = "."
BI_ENCODER_PATH = "./cti_model_20k_finetuned"  # 请确保运行目录下有此文件夹
CROSS_ENCODER_PATH = "./cti_reranker_final"

# 2. 知识库与报告
TECHNIQUE_DIR = "./attack-pattern"
REPORTS_DIR = "./CTI_reports"

# 3. 目标报告 (只测这两个)
TARGET_REPORTS = [
    "report_352_20251222_151010_ground_truth.json", # Good Case
    "report_509_20251222_161855_ground_truth.json"  # Bad Case
]

# 4. LLM
LLM_API_KEY = "" 
LLM_BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# 参数
TOP_K_RETRIEVE = 50   
TOP_K_RERANK = 10     

# ================= 1. 系统加载 =================
def load_system():
    print(">>> 🚀 Loading Case Study System...")
    
    bi_encoder = SentenceTransformer(BI_ENCODER_PATH)
    cross_encoder = CrossEncoder(CROSS_ENCODER_PATH)
    
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
        
    print(f"   ⚡ Encoded {len(kb_texts)} techniques.")
    kb_embs = bi_encoder.encode(kb_texts, convert_to_tensor=True)
    bm25 = BM25Okapi(kb_tokens)
    
    return bi_encoder, cross_encoder, kb_embs, bm25, kb_ids, kb_texts, kb_info

# ================= 2. 推理逻辑 =================

def analyze_chunk_advanced(text, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info):
    candidates_idx = set()
    
    # 1. Broad Retrieval
    w_emb = bi_enc.encode(text, convert_to_tensor=True)
    hits = util.semantic_search(w_emb, kb_embs, top_k=TOP_K_RETRIEVE)[0]
    for hit in hits: candidates_idx.add(hit['corpus_id'])
        
    b_scores = bm25.get_scores(text.lower().split())
    b_top = np.argsort(b_scores)[-TOP_K_RETRIEVE:]
    for i in b_top: candidates_idx.add(i)
        
    if not candidates_idx: return []
    
    # 2. Reranking
    cand_indices = list(candidates_idx)
    cross_inp = [[text, kb_texts[i]] for i in cand_indices]
    
    scores = cross_enc.predict(cross_inp)
    top_k_indices = np.argsort(scores)[-TOP_K_RERANK:]
    
    final_candidates = []
    for i in top_k_indices:
        final_candidates.append(kb_ids[cand_indices[i]])
        
    # 3. LLM Reasoning
    return llm_listwise_select(text, final_candidates, kb_info)

def llm_listwise_select(chunk_text, candidates, kb_info):
    cand_str = ""
    for idx, cid in enumerate(candidates):
        info = kb_info.get(cid, {})
        cand_str += f"Option {idx}: [ID: {cid}] {info.get('name')}\n"
        
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

# ================= 3. 辅助工具 =================
def get_sliding_windows(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s) > 10]
    windows = []
    for i in range(0, len(sentences), 1): 
        chunk = " ".join(sentences[i : i + 3])
        if len(chunk) > 20: windows.append(chunk)
    return windows

def get_parent_id(tid): return tid.split(".")[0] if "." in tid else tid

def check_match(pred, truth):
    if pred == truth: return True
    if get_parent_id(pred) == truth: return True
    if pred == get_parent_id(truth): return True
    return False

# ================= 4. 详细诊断打印 (核心) =================
def get_case_metrics(preds, true_ids):
    """Compute P/R/F1 with hierarchy-aware matching for case study diagnosis."""
    pred_list = list(preds)
    true_list = list(true_ids)

    tp_set = set()
    covered_truths = set()

    for p in pred_list:
        matched = False
        for t in true_list:
            if check_match(p, t):
                tp_set.add(p)
                covered_truths.add(t)
                matched = True
                break

    p_val = len(tp_set)/len(pred_list) if pred_list else 0.0
    r_val = len(covered_truths)/len(true_list) if true_list else 0.0
    f1_val = 2*p_val*r_val/(p_val+r_val) if (p_val+r_val)>0 else 0.0
    return p_val, r_val, f1_val

def save_case_study_results(all_case_data, output_path="case_study_results.json"):
    """Save case study results to structured JSON."""
    import time
    results = {
        "metadata": {
            "description": "Case study results — cached output from Deep-AttacKG qualitative analysis",
            "generated_by": "run_case_study.py",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "results": all_case_data
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nCase study results saved to {output_path}")
    pred_list = list(preds)
    true_list = list(true_ids)
    
    tp_set = set()
    fp_set = set()
    fn_set = set()
    covered_truths = set()
    
    # 找 TP (Match) 和 FP
    for p in pred_list:
        matched = False
        for t in true_list:
            if check_match(p, t):
                tp_set.add(p)
                covered_truths.add(t)
                matched = True
                break
        if not matched:
            fp_set.add(p)
            
    # 找 FN (Missed)
    for t in true_list:
        if t not in covered_truths:
            # 双重检查
            is_covered = False
            for p in pred_list:
                if check_match(p, t):
                    is_covered = True; break
            if not is_covered:
                fn_set.add(t)
            else:
                covered_truths.add(t)

    # 计算指标
    p_val = len(tp_set)/len(pred_list) if pred_list else 0.0
    r_val = len(covered_truths)/len(true_list) if true_list else 0.0
    f1_val = 2*p_val*r_val/(p_val+r_val) if (p_val+r_val)>0 else 0.0
    
    print(f"\n{'='*20} 📝 诊断报告: {name} {'='*20}")
    print(f"📊 F1: {f1_val:.2%} | Precision: {p_val:.2%} | Recall: {r_val:.2%}")
    print("-" * 60)
    
    print("✅ TP (成功捕获):")
    if tp_set:
        for i in tp_set:
            info = kb_info.get(i, {'name': 'Unknown'})
            print(f"   [Prediction] {i:<12} -> {info['name']}")
    else: print("   (None)")

    print("\n❌ FP (误报 - 重点分析这里):")
    if fp_set:
        for i in fp_set:
            info = kb_info.get(i, {'name': 'Unknown'})
            print(f"   [Prediction] {i:<12} -> {info['name']}")
    else: print("   (None)")
        
    print("\n🔻 FN (漏报 - 重点分析这里):")
    if fn_set:
        for i in fn_set:
            info = kb_info.get(i, {'name': 'Unknown'})
            print(f"   [Truth]      {i:<12} -> {info['name']}")
    else: print("   (None)")
        
    print("=" * 60 + "\n")

# ================= 主程序 =================
def run_case_study():
    # 检查文件
    if not os.path.exists(REPORTS_DIR):
        print(f"❌ 错误: 找不到报告目录 {REPORTS_DIR}")
        return

    bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info = load_system()
    
    print(f"\n>>> 🚀 开始对 {len(TARGET_REPORTS)} 篇目标报告进行深度诊断...\n")

    all_case_data = []

    for target_name in TARGET_REPORTS:
        r_path = os.path.join(REPORTS_DIR, target_name)
        if not os.path.exists(r_path):
            print(f"⚠️ 跳过: 找不到文件 {target_name}")
            continue

        with open(r_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        text = data.get('clean_text', '')
        true_ids = set(data.get('actual_found_ids', []))

        if not text:
            print(f"⚠️ 跳过: {target_name} 没有文本内容")
            continue

        windows = get_sliding_windows(text)
        all_preds = set()

        # 逐个窗口扫描
        for w in tqdm(windows, desc=f"Scanning {target_name}", leave=False):
            ids = analyze_chunk_advanced(w, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info)
            all_preds.update(ids)

        # 计算指标并打印诊断
        p_val, r_val, f1_val = get_case_metrics(all_preds, true_ids)
        all_case_data.append({
            "report": target_name,
            "precision": round(p_val * 100, 2),
            "recall": round(r_val * 100, 2),
            "micro_f1": round(f1_val * 100, 2)
        })

        # 打印详细诊断
        print_case_diagnosis(target_name, all_preds, true_ids, kb_info)

    # Save structured results
    save_case_study_results(all_case_data)

if __name__ == "__main__":
    run_case_study()