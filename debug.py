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
BI_ENCODER_PATH = "./cti_model_20k_finetuned"
# 请确认这个路径是正确的，不要用相对路径，容易错
CROSS_ENCODER_PATH = "./cti_reranker_final" 
if not os.path.exists(CROSS_ENCODER_PATH):
    # 容错：如果找不到，尝试项目内路径
    CROSS_ENCODER_PATH = "new_experiment_4/cti_reranker_final"

TECHNIQUE_DIR = "./attack-pattern"
REPORTS_DIR = "generated_reports"

LLM_API_KEY = "" 
LLM_BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

TOP_K_RETRIEVE = 50   
TOP_K_RERANK = 10     

# 是否开启兄弟节点宽容匹配？(写论文时可以说 "Strict F1" 和 "Soft F1")
ENABLE_SIBLING_MATCH = True 

# ================= 1. 加载系统 =================
def load_system():
    print(">>> 🚀 Loading Final System...")
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
        
    print(f"   ⚡ Encoded {len(kb_texts)} KB items...")
    kb_embs = bi_encoder.encode(kb_texts, convert_to_tensor=True)
    bm25 = BM25Okapi(kb_tokens)
    return bi_encoder, cross_encoder, kb_embs, bm25, kb_ids, kb_texts, kb_info

# ================= 2. 推理逻辑 (Prompt 升级) =================

def analyze_chunk_advanced(text, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info):
    candidates_idx = set()
    
    # Broad Recall
    w_emb = bi_enc.encode(text, convert_to_tensor=True)
    hits = util.semantic_search(w_emb, kb_embs, top_k=TOP_K_RETRIEVE)[0]
    for hit in hits: candidates_idx.add(hit['corpus_id'])
        
    b_scores = bm25.get_scores(text.lower().split())
    b_top = np.argsort(b_scores)[-TOP_K_RETRIEVE:]
    for i in b_top: candidates_idx.add(i)
        
    if not candidates_idx: return []
    
    # Rerank
    cand_indices = list(candidates_idx)
    cross_inp = [[text, kb_texts[i]] for i in cand_indices]
    scores = cross_enc.predict(cross_inp)
    top_k_indices = np.argsort(scores)[-TOP_K_RERANK:]
    
    final_candidates = []
    for i in top_k_indices:
        final_candidates.append(kb_ids[cand_indices[i]])
        
    return llm_listwise_select(text, final_candidates, kb_info)

def llm_listwise_select(chunk_text, candidates, kb_info):
    cand_str = ""
    for idx, cid in enumerate(candidates):
        info = kb_info.get(cid, {})
        cand_str += f"Option {idx}: [ID: {cid}] {info.get('name')}\n"
    
    # 🌟 升级版 Prompt：增加“反向约束”以减少幻觉
    prompt = f"""
Task: Select MITRE ATT&CK techniques that explicitly match the text.
Text: "{chunk_text}"
Options:
{cand_str}

Rules:
1. Match based on **Behavior**, not just keywords.
   - Example: "Deleted file" -> T1070 (Indicator Removal), NOT T1561 (Disk Wipe) unless explicit.
   - Example: "Social media" -> T1585 (Persona) only if used for attack, else benign.
2. If multiple sub-techniques fit (e.g., .001 vs .002), pick the most specific one.
3. If unsure or benign, return empty.

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

# ================= 3. 评估逻辑 (Check Match 升级) =================

def get_parent_id(tid): return tid.split(".")[0] if "." in tid else tid

def check_match(pred, truth):
    # 1. 完全匹配
    if pred == truth: return True
    
    # 2. 父子匹配 (T1059 vs T1059.001)
    p_parent = get_parent_id(pred)
    t_parent = get_parent_id(truth)
    if p_parent == truth: return True
    if pred == t_parent: return True
    
    # 3. 兄弟匹配 (T1566.002 vs T1566.003) - 仅在开关开启时启用
    # 这能显著提升 "Soft F1"，反映模型其实“找对了大方向”
    if ENABLE_SIBLING_MATCH and p_parent == t_parent:
        return True
        
    return False
def detailed_report(name, preds, true_ids, kb_info):
    pred_list = list(preds)
    true_list = list(true_ids)
    
    tp_set = set()      # 存预测 ID
    fp_set = set()      # 存预测 ID
    fn_set = set()      # 存真值 ID
    covered_truths = set() # 🌟 新增：专门记录哪些真值被覆盖了
    
    # 1. 计算 Precision (基于预测列表)
    for p in pred_list:
        matched = False
        for t in true_list:
            if check_match(p, t):
                tp_set.add(p)
                covered_truths.add(t) # 记录这个真值被找到了
                matched = True
                # 注意：这里不 break，因为一个预测可能对应多个真值（虽然罕见），
                # 或者为了统计 covered_truths 我们需要遍历。
                # 但为了效率，只要匹配到一个就可以算 TP。
                break 
        if not matched:
            fp_set.add(p)
            
    # 2. 计算 Recall (基于真值列表)
    # 只要真值在 covered_truths 里，或者能被任一预测匹配，就算召回
    for t in true_list:
        if t not in covered_truths:
            # 双重检查：防止上面的 break 导致漏记
            is_covered = False
            for p in pred_list:
                if check_match(p, t):
                    is_covered = True
                    break
            if not is_covered:
                fn_set.add(t)
            else:
                covered_truths.add(t)

    # --- 3. 修正后的计算公式 ---
    
    # Precision = 正确的预测数 / 总预测数
    precision_val = len(tp_set) / len(pred_list) if pred_list else 0.0
    
    # Recall = 被覆盖的真值数 / 总真值数 (❌ 之前是用 len(tp_set) 导致溢出)
    recall_val = len(covered_truths) / len(true_list) if true_list else 0.0
    
    # F1 Score
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0
    
    # --- 4. 打印 ---
    print(f"\n📝 Report: {name}")
    print("-" * 60)
    print(f"✅ TP (Predictions): {len(tp_set)} | 🎯 Covered Truths: {len(covered_truths)}/{len(true_list)}")
    print(f"❌ FP: {len(fp_set)} | 🔻 FN: {len(fn_set)}")
    
    if fp_set:
        print("   False Positives:", [f"{i} {kb_info.get(i,{}).get('name')}" for i in list(fp_set)[:5]] + ["..."] if len(fp_set)>5 else "")
    if fn_set:
        print("   Missed Truths:", [f"{i} {kb_info.get(i,{}).get('name')}" for i in fn_set])
        
    print(f"📊 F1: {f1_val:.2%} (P={precision_val:.2%}, R={recall_val:.2%})")
    print("-" * 60)
    
    return precision_val, recall_val, f1_val
def get_sliding_windows(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s) > 10]
    windows = []
    for i in range(0, len(sentences), 1): 
        chunk = " ".join(sentences[i : i + 3])
        if len(chunk) > 20: windows.append(chunk)
    return windows

def run_final_test():
    bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info = load_system()
    report_files = glob.glob(os.path.join(REPORTS_DIR, "*.json"))[:5]
    
    avg_p, avg_r, avg_f1 = 0, 0, 0
    print(f"\n>>> 🚀 Starting Final Optimization Test (Sibling Match={ENABLE_SIBLING_MATCH})...\n")
    
    for r_path in report_files:
        with open(r_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        text = data.get('clean_text', '')
        true_ids = set(data.get('actual_found_ids', []))
        if not text: continue
        
        all_preds = set()
        for w in tqdm(get_sliding_windows(text), desc="Scanning", leave=False):
            ids = analyze_chunk_advanced(w, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info)
            all_preds.update(ids)
            
        p, r, f1 = detailed_report(os.path.basename(r_path), all_preds, true_ids, kb_info)
        avg_p += p; avg_r += r; avg_f1 += f1
        
    print("\n" + "="*50)
    print(f"🏆 Final Avg F1: {avg_f1/len(report_files):.2%}")
    print("="*50)

if __name__ == "__main__":
    run_final_test()