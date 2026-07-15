import os
import json
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
import numpy as np

# ================= 🔧 配置参数 =================
CTI_DATA_DIR = "./CTI_reports"                  # CTI报告文件夹路径
BEDR_CSV_PATH = "./BEDR_resampled_dataset.csv"  # BEDR数据集路径
MAX_SEQ_LEN = 512                               # 最大序列长度
MAX_VOCAB_SIZE = 25000                          # 词表大小 (由于加入了BEDR，词表适当扩大)
BATCH_SIZE = 32                                 # 批次大小
EMBEDDING_DIM = 128                             # 词向量维度
HIDDEN_DIM = 128                                # LSTM和CNN的隐藏层维度
EPOCHS = 15                                     # 训练轮数
LEARNING_RATE = 1e-3                            # 学习率
PREDICT_THRESHOLD = 0.5                         # 多标签分类阈值

# ================= 1. 数据加载与融合 =================
def load_cti_data(data_dir, return_fnames=False):
    """Load CTI JSON reports and return texts with multi-hot labels."""
    texts = []
    labels = []
    fnames = []
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if "clean_text" in data and "actual_found_ids" in data:
                    texts.append(data["clean_text"])
                    labels.append(data["actual_found_ids"])
                    fnames.append(os.path.basename(file_path))
            except Exception as e:
                print(f"解析 {file_path} 时出错: {e}")

    if return_fnames:
        return texts, labels, fnames
    return texts, labels

def load_bedr_data(csv_path):
    """加载 BEDR CSV 数据集，转换为与 CTI 兼容的多标签格式"""
    texts = []
    labels = []
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            if pd.notna(row['text']) and pd.notna(row['tech_id']):
                texts.append(str(row['text']))
                # 即使是单标签，也封装成 list 以统一格式
                labels.append([str(row['tech_id']).strip()])
    except Exception as e:
        print(f"加载 BEDR 数据集出错: {e}")
        
    return texts, labels

# ================= 2. 文本预处理与 Dataset 构建 =================
def build_vocab(texts, max_size):
    word_counts = Counter()
    for text in texts:
        word_counts.update(str(text).lower().split())
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in word_counts.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab

def text_to_sequence(text, vocab, max_len):
    tokens = str(text).lower().split()
    seq = [vocab.get(word, vocab["<UNK>"]) for word in tokens]
    # 截断或填充
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [vocab["<PAD>"]] * (max_len - len(seq))
    return seq

class HybridDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ================= 3. 深度学习模型架构 (CNN-LSTM) =================
class CNN_LSTM_MultiLabel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(CNN_LSTM_MultiLabel, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # BiLSTM 分支：捕捉长距离序列上下文
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, 
                            bidirectional=True, batch_first=True)
        
        # CNN 分支：捕捉局部 n-gram 特征 (如特定的命令、API名称)
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # 全连接分类层
        # BiLSTM 输出维度为 hidden_dim * 2, CNN 输出维度为 hidden_dim
        self.fc = nn.Linear(hidden_dim * 2 + hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x) # (batch, seq_len, embed_dim)
        
        # --- BiLSTM 路径 ---
        _, (hidden, _) = self.lstm(embedded)
        # 拼接正向和反向的最后隐状态
        lstm_features = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) 
        
        # --- CNN 路径 ---
        embedded_permuted = embedded.permute(0, 2, 1) # (batch, channels, seq_len)
        conv_out = self.relu(self.conv1d(embedded_permuted))
        # 全局最大池化
        cnn_features, _ = torch.max(conv_out, dim=2) 
        
        # --- 特征融合 ---
        combined = torch.cat((lstm_features, cnn_features), dim=1)
        combined = self.dropout(combined)
        
        logits = self.fc(combined)
        return logits

# ================= 4. 主干训练与测试流程 =================
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 使用计算设备: {device}")

    # 1. 加载 CTI 和 BEDR 数据
    print("📂 正在加载数据...")
    cti_texts, cti_labels, cti_fnames = load_cti_data(CTI_DATA_DIR, return_fnames=True)
    bedr_texts, bedr_labels = load_bedr_data(BEDR_CSV_PATH)

    print(f"   - 成功加载 {len(cti_texts)} 篇 CTI 报告")
    print(f"   - 成功加载 {len(bedr_texts)} 条 BEDR 记录")

    # 2. 统一标签编码 (MultiLabelBinarizer 需要拟合所有可能出现的标签)
    all_labels = cti_labels + bedr_labels
    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)
    num_classes = len(mlb.classes_)
    print(f"🎯 统一标签空间: 共 {num_classes} 种独特的 ATT&CK 技术")

    # 3. Use published test split for consistent evaluation
    split_path = "test_split.json"
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            split_data = json.load(f)
        test_files = set(split_data["test_files"])
        train_idx = [i for i, fn in enumerate(cti_fnames) if fn not in test_files]
        test_idx  = [i for i, fn in enumerate(cti_fnames) if fn in test_files]
        print(f"   Using published split: {len(train_idx)} train / {len(test_idx)} test")
    else:
        print(f"   [WARN] {split_path} not found, falling back to re-split")
        indices = list(range(len(cti_texts)))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    X_cti_train = [cti_texts[i] for i in train_idx]
    y_cti_train = [cti_labels[i] for i in train_idx]
    X_cti_test  = [cti_texts[i] for i in test_idx]
    y_cti_test  = [cti_labels[i] for i in test_idx]

    # 训练集 = 80% 的 CTI + 100% 的 BEDR
    train_texts = X_cti_train + bedr_texts
    train_labels = y_cti_train + bedr_labels

    # 测试集 = 20% 的 CTI (完全隔离真实领域测试)
    test_texts = X_cti_test
    test_labels = y_cti_test

    print(f"📊 训练集大小: {len(train_texts)} (CTI 80% + BEDR 100%)")
    print(f"📊 测试集大小: {len(test_texts)} (仅 CTI 20%)")

    # 4. 构建词表与数字化
    vocab = build_vocab(train_texts, MAX_VOCAB_SIZE)
    vocab_size = len(vocab)
    
    X_train_seq = [text_to_sequence(t, vocab, MAX_SEQ_LEN) for t in train_texts]
    X_test_seq = [text_to_sequence(t, vocab, MAX_SEQ_LEN) for t in test_texts]
    
    y_train_bin = mlb.transform(train_labels)
    y_test_bin = mlb.transform(test_labels)

    # 5. 构建 DataLoader
    train_dataset = HybridDataset(X_train_seq, y_train_bin)
    test_dataset = HybridDataset(X_test_seq, y_test_bin)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6. 初始化模型、损失函数与优化器
    model = CNN_LSTM_MultiLabel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, num_classes).to(device)
    # 使用 BCEWithLogitsLoss 处理多标签分类问题
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 7. 训练循环
    print("\n🔥 开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}] | 训练损失 (Loss): {total_loss/len(train_loader):.4f}")

    # 8. 评估循环
    print("\n🔍 开始在 20% CTI 报告上进行 Zero/Few-shot 测试评估...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            
            # 使用 Sigmoid 将 logits 转换为 0-1 的概率
            probs = torch.sigmoid(logits)
            
            # 采用 0.5 的严格阈值判断类别存在与否
            preds = (probs > PREDICT_THRESHOLD).float()
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # 9. 计算 Micro 指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='micro', zero_division=0
    )
    
    print("\n" + "="*40)
    print("🏆 CTI-1002 测试集最终评估结果 (Data-Symmetric Baseline)")
    print("="*40)
    print(f"🎯 Micro-Precision: {precision:.4f}  ({precision*100:.2f}%)")
    print(f"🎯 Micro-Recall   : {recall:.4f}  ({recall*100:.2f}%)")
    print(f"🔥 Micro-F1 Score : {f1:.4f}  ({f1*100:.2f}%)")
    print("="*40)

if __name__ == "__main__":
    run_experiment()