import numpy as np
import torch
import os
import random
from collections import defaultdict
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# ================= Reproducibility (Fixed Seeds) =================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================= 配置区域 =================
DATASET_PATH = "./D_BEDR.npz"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
OUTPUT_PATH = "cti_model_20k_finetuned"

# 训练参数 (M1 Pro 优化版)
BATCH_SIZE = 16    # 20k 数据，16 比较稳
EPOCHS = 4         # 训练 4 轮，足够收敛
LR = 2e-5

def load_and_balance_data(path):
    print(f">>> 📂 Loading and Balancing Data from {path}...")
    data = np.load(path, allow_pickle=True)
    
    # 1. 提取文本和标签
    # 注意：这里假设 text 存储在 'texts' key 中，如果报错请改为 keys列表中实际的文本key
    texts = data['texts']
    labels = data['labels']
    
    # 解码 (如果是 bytes)
    decoded_texts = []
    for t in texts:
        if isinstance(t, bytes):
            decoded_texts.append(t.decode('utf-8'))
        else:
            decoded_texts.append(str(t))
            
    # 2. 按类别分组
    groups = defaultdict(list)
    for text, label in zip(decoded_texts, labels):
        groups[label].append(text)
        
    print(f"   Original: {len(decoded_texts)} samples, {len(groups)} classes.")
    
    # 3. 过采样 (Oversampling) - 关键步骤
    # 目标：让每个类别的样本数都达到最大类的数量 (或者是中位数，这里用 80 也就是最大值)
    TARGET_COUNT = 80 
    balanced_pairs = []
    
    print(f"   ⚖️ Balancing classes to target count: {TARGET_COUNT}...")
    
    for label, samples in groups.items():
        # 如果样本不够，随机重复采样直到填满 TARGET_COUNT
        curr_samples = samples.copy()
        while len(curr_samples) < TARGET_COUNT:
            curr_samples.append(random.choice(samples)) # 随机回采
            
        # 如果样本本来就很多(比如80)，就截断或保持 (这里保持)
        # 现在构建训练对 (Anchor, Positive)
        # 从同一个类里随机选两个不同的句子组成一对
        for _ in range(TARGET_COUNT): 
            # 随机抽两个
            a = random.choice(curr_samples)
            b = random.choice(curr_samples)
            # 尽量不要自己和自己配对，除非只有一条数据
            if a == b and len(set(curr_samples)) > 1:
                while b == a:
                    b = random.choice(curr_samples)
            
            balanced_pairs.append(InputExample(texts=[a, b]))
            
    print(f"✅ Data Prepared. Total Training Pairs: {len(balanced_pairs)}")
    return balanced_pairs

def train():
    # 1. 设备检查
    if torch.backends.mps.is_available():
        device = "mps"
        print(">>> 🚀 MPS Acceleration Enabled")
    else:
        device = "cpu"
        print(">>> ⚠️ Using CPU")

    # 2. 准备数据
    train_examples = load_and_balance_data(DATASET_PATH)
    
    # 切分一小部分做验证 (可选，这里为了最大化训练数据，全量训练)
    random.shuffle(train_examples)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE, num_workers=0)

    # 3. 加载基座模型
    print(f">>> Loading base model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = 512

    # 4. 损失函数
    # MultipleNegativesRankingLoss 是无监督/自监督训练的神器
    # 它会把 batch 里其他对的句子作为负样本
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 5. 开始训练
    print(f">>> 🏋️ Starting Fine-Tuning ({EPOCHS} epochs)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=int(len(train_dataloader) * 0.1),
        output_path=OUTPUT_PATH,
        optimizer_params={'lr': LR},
        show_progress_bar=True,
        use_amp=False 
    )
    
    print(f"✅ Model saved to: {OUTPUT_PATH}")
    print("👉 Now you can use this model in your evaluation script!")

if __name__ == "__main__":
    train()