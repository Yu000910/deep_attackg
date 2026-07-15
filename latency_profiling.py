import os
import json
import glob
import torch
import time
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from rank_bm25 import BM25Okapi
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 消除 tokenizers 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================= 🔧 专用配置 (请替换为你的路径) =================
BI_ENCODER_PATH = "./cti_model_20k_finetuned"
CROSS_ENCODER_PATH = "./cti_reranker_final"
TECHNIQUE_DIR = "./attack-pattern"
TRAM_FILE_PATH = "./TRAM/multi_label.json"
REPORTS_DIR = "./CTI_reports"

LLM_API_KEY = "your-deepseek-api-key"  # ⚠️ 请填入你的 API KEY
LLM_BASE_URL = "https://api.deepseek.com"
client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

TOP_K_RETRIEVE = 50   
TOP_K_RERANK = 10     
SAMPLE_SIZE = 50    # 测试样本量 (足够计算均值±标准差)

# ================= 1. 加载模型与知识库 =================
def load_system():
    print(">>> 🚀 Loading Models for Latency Profiling...")
    # 根据你的环境，如果是Mac请保持不变，如果是Linux服务器请确保使用 cuda
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    bi_encoder = SentenceTransformer(BI_ENCODER_PATH, device=device)
    cross_encoder = CrossEncoder(CROSS_ENCODER_PATH, device=device)
    
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
        
    kb_embs = bi_encoder.encode(kb_texts, convert_to_tensor=True)
    bm25 = BM25Okapi(kb_tokens)
    return bi_encoder, cross_encoder, kb_embs, bm25, kb_ids, kb_texts, kb_info

def load_and_sample_tram_data(filepath, sample_size):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    reports = defaultdict(lambda: {"text": "", "labels": set()})
    for item in data:
        title = item.get("doc_title", "Unknown_Doc")
        sentence = item.get("sentence", "").strip()
        if sentence:
            reports[title]["text"] += sentence + " "
            
    valid_reports = [{"title": k, "text": v["text"].strip()} for k, v in reports.items() if len(v["text"]) > 50]
    
    import random
    random.seed(42)
    sampled = random.sample(valid_reports, min(sample_size, len(valid_reports)))
    return sampled

def load_and_sample_cti_data(reports_dir, sample_size, split_path="test_split.json"):
    """Load CTI reports for latency profiling using the published test split."""
    json_files = sorted(glob.glob(os.path.join(reports_dir, "*_ground_truth.json")))
    reports = []
    all_fnames = []
    for fpath in json_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text = data.get('clean_text', '')
            if text and len(text) > 50:
                reports.append({"title": os.path.basename(fpath), "text": text})
                all_fnames.append(os.path.basename(fpath))
        except:
            pass

    # Use published test split
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        test_files = set(split_data["test_files"])
        test_reports = [r for r, fn in zip(reports, all_fnames) if fn in test_files]
        print(f"   Test reports (from split): {len(test_reports)}")
    else:
        print(f"   [WARN] {split_path} not found, falling back to re-split")
        from sklearn.model_selection import train_test_split
        indices = list(range(len(reports)))
        _, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        test_reports = [reports[i] for i in test_idx]

    import random as rnd
    rnd.seed(42)
    return rnd.sample(test_reports, min(sample_size, len(test_reports)))

def get_sliding_windows(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s) > 10]
    windows = []
    for i in range(0, len(sentences), 1): 
        chunk = " ".join(sentences[i : i + 3])
        if len(chunk) > 20: windows.append(chunk)
    return windows

# LLM API 调用 (Stage 3)
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
        return True # 我们只需要测时间，不关心具体的返回结果
    except: 
        return False

# ================= 2. 核心：带时间性能监控的分析函数 =================
def analyze_report_with_profiling(text, bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info):
    windows = get_sliding_windows(text)
    
    # 记录该报告在每个 Stage 的累计耗时 (秒)
    stage1_time = 0.0
    stage2_time = 0.0
    stage3_time = 0.0
    
    llm_tasks = []
    
    for w in windows:
        # ---------------- Stage 1 ----------------
        t0 = time.perf_counter()
        candidates_idx = set()
        w_emb = bi_enc.encode(w, convert_to_tensor=True)
        hits = util.semantic_search(w_emb, kb_embs, top_k=TOP_K_RETRIEVE)[0]
        for hit in hits: candidates_idx.add(hit['corpus_id'])
            
        b_scores = bm25.get_scores(w.lower().split())
        b_top = np.argsort(b_scores)[-TOP_K_RETRIEVE:]
        for i in b_top: candidates_idx.add(i)
        t1 = time.perf_counter()
        stage1_time += (t1 - t0)
        
        if not candidates_idx: continue
        
        # ---------------- Stage 2 ----------------
        t2 = time.perf_counter()
        cand_indices = list(candidates_idx)
        cross_inp = [[w, kb_texts[i]] for i in cand_indices]
        scores = cross_enc.predict(cross_inp)
        
        if np.max(scores) >= -2.0:  
            top_k_indices = np.argsort(scores)[-TOP_K_RERANK:]
            final_candidates = [kb_ids[cand_indices[i]] for i in top_k_indices]
            if final_candidates:
                llm_tasks.append((w, final_candidates))
        t3 = time.perf_counter()
        stage2_time += (t3 - t2)
            
    # ---------------- Stage 3 ----------------
    if llm_tasks:
        t4 = time.perf_counter()
        # 模拟真实的并发API请求
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(llm_listwise_select, task_w, task_cands, kb_info) for task_w, task_cands in llm_tasks]
            for future in as_completed(futures):
                pass 
        t5 = time.perf_counter()
        stage3_time += (t5 - t4)
        
    return stage1_time * 1000, stage2_time * 1000, stage3_time * 1000 # 转换为毫秒

# ================= 3. 执行评估与统计 =================
def run_latency_profiling():
    bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info = load_system()
    test_reports = load_and_sample_cti_data(REPORTS_DIR, SAMPLE_SIZE)
    
    stage1_records = []
    stage2_records = []
    stage3_records = []
    total_records = []
    
    print(f"\n>>> ⏳ Starting Latency Profiling for {len(test_reports)} reports...")
    
    for report in tqdm(test_reports, desc="Profiling"):
        s1, s2, s3 = analyze_report_with_profiling(report["text"], bi_enc, cross_enc, kb_embs, bm25, kb_ids, kb_texts, kb_info)
        
        # 只记录触发了完整三阶段的报告，以保证统计有效性
        if s3 > 0:
            stage1_records.append(s1)
            stage2_records.append(s2)
            stage3_records.append(s3)
            total_records.append(s1 + s2 + s3)
            
    print("\n" + "="*50)
    print("📊 延迟性能剖析报告 (Latency Profiling Report)")
    print(f"有效测试样本量: {len(total_records)}")
    print("="*50)
    
    print(f"Stage 1 (混合检索 - 本地): {np.mean(stage1_records):.2f} ± {np.std(stage1_records):.2f} ms")
    print(f"Stage 2 (交叉重排 - 本地): {np.mean(stage2_records):.2f} ± {np.std(stage2_records):.2f} ms")
    print(f"Stage 3 (LLM推理 - API):  {np.mean(stage3_records):.2f} ± {np.std(stage3_records):.2f} ms")
    print("-" * 50)
    print(f"总计端到端延迟:          {np.mean(total_records):.2f} ± {np.std(total_records):.2f} ms")
    print("="*50)

if __name__ == "__main__":
    run_latency_profiling()