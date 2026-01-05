import numpy as np
import torch
import random
import os
import shutil
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, CrossEncoder
from collections import defaultdict

# ================= 🚨 关键修改配置 🚨 =================
# 1. 强制保存到桌面最外层，使用纯英文路径
# 这样可以绝对避免路径过深或中文字符导致的问题
user_home = os.path.expanduser("~")
OUTPUT_PATH = os.path.join(user_home, "Desktop", "cti_reranker_final")

# 2. 数据集路径 (保持你的原始路径)
DATASET_PATH = "/Users/nnn/Desktop/temp/博士毕业/第五篇/verb-tool-project/datasets/D_BEDR.npz"

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2" 
BATCH_SIZE = 16
EPOCHS = 3

def prepare_cross_data(path):
    print(f">>> 📦 Loading Data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据集未找到: {path}")

    data = np.load(path, allow_pickle=True)
    texts = [str(t) if not isinstance(t, bytes) else t.decode('utf-8') for t in data['texts']]
    labels = data['labels']
    
    groups = defaultdict(list)
    for t, l in zip(texts, labels):
        groups[l].append(t)
        
    train_samples = []
    keys = list(groups.keys())
    
    print(">>> ⚙️ Generating Positive/Negative Pairs...")
    for label, samples in groups.items():
        for text in samples:
            if len(samples) > 1:
                pos_text = random.choice(samples)
                while pos_text == text and len(samples) > 1:
                    pos_text = random.choice(samples)
                train_samples.append(InputExample(texts=[text, pos_text], label=1.0))
            
            neg_label = random.choice(keys)
            while neg_label == label: neg_label = random.choice(keys)
            neg_text = random.choice(groups[neg_label])
            
            train_samples.append(InputExample(texts=[text, neg_text], label=0.0))
            
    print(f"✅ Generated {len(train_samples)} training pairs.")
    return train_samples

def train_reranker():
    # --- 1. 权限与路径测试 (Write Test) ---
    print(f">>> 📂 目标路径: {OUTPUT_PATH}")
    if os.path.exists(OUTPUT_PATH):
        print("    (文件夹已存在，将覆盖)")
    else:
        os.makedirs(OUTPUT_PATH)
        print("    (文件夹已创建)")
    
    # 📝 写入一个测试文件，确保有写入权限
    test_file = os.path.join(OUTPUT_PATH, "write_test.txt")
    try:
        with open(test_file, 'w') as f:
            f.write("Test write permission OK.")
        print("✅ 写入权限测试通过！文件系统正常。")
    except Exception as e:
        print(f"❌ 严重错误: 无法写入目标文件夹！请检查权限。\n{e}")
        return # 直接退出，不浪费时间

    # --- 2. 准备模型 ---
    print(f">>> 🧠 Loading Model: {MODEL_NAME}...")
    model = CrossEncoder(MODEL_NAME, num_labels=1)
    
    # --- 3. 准备数据 ---
    train_samples = prepare_cross_data(DATASET_PATH)
    random.shuffle(train_samples)
    
    train_dataloader = DataLoader(
        train_samples, 
        shuffle=True, 
        batch_size=BATCH_SIZE,
        collate_fn=model.smart_batching_collate
    )
    
    # --- 4. 训练 ---
    print(f">>> 🏋️ Starting Training...")
    
    try:
        model.fit(
            train_dataloader=train_dataloader,
            epochs=EPOCHS,
            warmup_steps=int(len(train_dataloader) * 0.1),
            output_path=OUTPUT_PATH,  # 自动保存尝试 1
            show_progress_bar=True
        )
        print("\n✅ Training Finished Loop.")
        
        # --- 5. 强制手动保存 (双重保险) ---
        print(">>> 💾 Forcing Manual Save...")
        model.save(OUTPUT_PATH) # 显式调用保存
        
        # 再次确认文件是否真的在
        if os.path.exists(os.path.join(OUTPUT_PATH, "config.json")):
             print(f"\n🎉 成功！模型已确认保存在桌面文件夹: cti_reranker_final")
             print(f"路径: {OUTPUT_PATH}")
        else:
             print(f"\n⚠️ 警告: 训练完成但未检测到 config.json，请手动检查路径！")

    except Exception as e:
        print(f"\n❌ Training Crashed: {e}")

if __name__ == "__main__":
    train_reranker()