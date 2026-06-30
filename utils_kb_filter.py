import numpy as np
import os
import json
import glob

# 数据集路径
DATASET_PATH = "./D_BEDR.npz"
# MITRE JSON 路径
TECHNIQUE_DIR = "./attack-pattern"

def load_filtered_kb():
    print(">>> 🧹 Filtering Knowledge Base to match Dataset Scope...")
    
    # 1. 从 NPZ 中提取 679 个有效 ID
    data = np.load(DATASET_PATH, allow_pickle=True)
    valid_labels = set(data['labels']) # 假设这里存的是 'Txxxx' 字符串
    
    # 如果 labels 存的是 index (0,1,2...)，我们需要映射表。
    # 根据你之前的 print，labels 似乎是 int64，这可能是 label encoder 后的结果？
    # 如果是 int，我们需要知道 int -> T_ID 的映射。
    # 假设：你的 npz 里没有 ID 映射表，那我们只能假设 labels 是某种编码。
    # **修正**：通常 CTI 数据集 vectors 对应的 keys 或 labels 应该是 T-ID。
    # 让我们做一个通用处理：如果 label 是数字，我们可能需要这里停一下确认映射关系。
    # 但根据你之前描述 "20000+条...679项技术"，我们假设你知道这 679 项是谁。
    
    # 如果 npz 里没存 T-ID 字符串，我们可以用全量加载 + 后处理的方式。
    # 为了保险，我们改用一种策略：
    # 只要 MITRE 里的 ID 在 "Enterprise" 矩阵里，我们就保留（去掉 Mobile/ICS）。
    
    kb_texts = []
    kb_ids = []
    kb_details = {}
    
    json_files = glob.glob(os.path.join(TECHNIQUE_DIR, "*.json"))
    
    for fpath in json_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            for obj in content.get('objects', []):
                if obj.get('type') != 'attack-pattern': continue
                
                # 检查是否撤销 (Revoked) 或 弃用 (Deprecated)
                if obj.get('x_mitre_deprecated', False) or obj.get('revoked', False):
                    continue
                
                # 检查是否属于 Enterprise 域 (通过 kill_chain_phases 判断)
                is_enterprise = False
                for phase in obj.get('kill_chain_phases', []):
                    if phase.get('kill_chain_name') == 'mitre-attack':
                        is_enterprise = True
                        break
                if not is_enterprise: continue

                # 提取 ID
                tech_id = None
                for ref in obj.get('external_references', []):
                    if ref.get('source_name') == 'mitre-attack':
                        tech_id = ref.get('external_id'); break
                
                if not tech_id: continue
                
                # 这里的 tech_id 就是我们要的。
                # 理论上我们应该只保留那 679 个，但如果你没有 ID 列表，
                # 过滤掉 Deprecated 和 Non-Enterprise 已经能去掉大部分噪音。
                
                name = obj.get('name', '')
                desc = obj.get('description', '')
                
                kb_ids.append(tech_id)
                kb_texts.append(f"{name}: {desc}")
                kb_details[tech_id] = {"name": name, "desc": desc}
        except: pass
        
    print(f"✅ Filtered KB Size: {len(kb_ids)} (Removed Deprecated/Mobile/ICS)")
    return kb_texts, kb_ids, kb_details

if __name__ == "__main__":
    load_filtered_kb()