"""
ACRNN Baseline — 带逐Epoch日志记录的版本
用于生成 learning_curve_analysis.py 所需的真实训练曲线数据
"""
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

# ================= 配置 =================
CTI_DATA_DIR = "./CTI_reports"
BEDR_CSV_PATH = "./BEDR_resampled_dataset.csv"
MAX_SEQ_LEN = 512
MAX_VOCAB_SIZE = 25000
BATCH_SIZE = 32
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
EPOCHS = 20         # 20 epochs for clearer curve
LEARNING_RATE = 1e-3
PREDICT_THRESHOLD = 0.5
OUTPUT_LOG = "learning_curve_real_data.json"

# ================= 1. 数据加载 =================
def load_cti_data(data_dir):
    texts, labels = [], []
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if "clean_text" in data and "actual_found_ids" in data:
                    texts.append(data["clean_text"])
                    labels.append(data["actual_found_ids"])
            except:
                pass
    return texts, labels

def load_bedr_data(csv_path):
    texts, labels = [], []
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            if pd.notna(row['text']) and pd.notna(row['tech_id']):
                texts.append(str(row['text']))
                labels.append([str(row['tech_id']).strip()])
    except Exception as e:
        print(f"加载 BEDR 数据集出错: {e}")
    return texts, labels

# ================= 2. 文本预处理 =================
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

# ================= 3. 模型 =================
class CNN_LSTM_MultiLabel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(CNN_LSTM_MultiLabel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim * 2 + hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        lstm_features = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        embedded_permuted = embedded.permute(0, 2, 1)
        conv_out = self.relu(self.conv1d(embedded_permuted))
        cnn_features, _ = torch.max(conv_out, dim=2)
        combined = torch.cat((lstm_features, cnn_features), dim=1)
        combined = self.dropout(combined)
        logits = self.fc(combined)
        return logits

# ================= 4. 验证集评估 =================
@torch.no_grad()
def evaluate(model, data_loader, device, mlb, head_class_indices, tail_class_indices):
    model.eval()
    all_preds, all_targets = [], []
    for batch_x, batch_y in data_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        probs = torch.sigmoid(logits)
        preds = (probs > PREDICT_THRESHOLD).float()
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Global micro metrics
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='micro', zero_division=0)

    # Head-class recall
    if head_class_indices:
        head_p, head_r, head_f1, _ = precision_recall_fscore_support(
            all_targets[:, head_class_indices], all_preds[:, head_class_indices],
            average='micro', zero_division=0)
    else:
        head_r = 0.0

    # Tail-class recall
    if tail_class_indices:
        tail_p, tail_r, tail_f1, _ = precision_recall_fscore_support(
            all_targets[:, tail_class_indices], all_preds[:, tail_class_indices],
            average='micro', zero_division=0)
    else:
        tail_r = 0.0

    model.train()
    return float(micro_p), float(micro_r), float(micro_f1), float(head_r), float(tail_r)

# ================= 5. 主程序 =================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据
    print("Loading data...")
    cti_texts, cti_labels = load_cti_data(CTI_DATA_DIR)
    bedr_texts, bedr_labels = load_bedr_data(BEDR_CSV_PATH)
    print(f"  CTI reports: {len(cti_texts)}, BEDR records: {len(bedr_texts)}")

    # 统一标签
    all_labels = cti_labels + bedr_labels
    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)
    num_classes = len(mlb.classes_)
    print(f"  Label space: {num_classes} classes")

    # 三层分割: CTI → train(64%) / val(16%) / test(20%)
    X_cti_trainval, X_cti_test, y_cti_trainval, y_cti_test = train_test_split(
        cti_texts, cti_labels, test_size=0.2, random_state=42)
    X_cti_train, X_cti_val, y_cti_train, y_cti_val = train_test_split(
        X_cti_trainval, y_cti_trainval, test_size=0.2, random_state=42)

    # 训练集 = CTI train + BEDR
    train_texts = X_cti_train + bedr_texts
    train_labels = y_cti_train + bedr_labels
    val_texts, val_labels = X_cti_val, y_cti_val
    test_texts, test_labels = X_cti_test, y_cti_test

    print(f"  Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # 计算 Head/Tail 划分 (基于训练集中各类别的样本数)
    train_bin = mlb.transform(train_labels)
    class_sample_counts = train_bin.sum(axis=0)
    median_count = np.median(class_sample_counts[class_sample_counts > 0])
    head_indices = [i for i, c in enumerate(class_sample_counts) if c >= median_count]
    tail_indices = [i for i, c in enumerate(class_sample_counts) if 0 < c < median_count]
    print(f"  Head classes: {len(head_indices)}, Tail classes: {len(tail_indices)} (median={median_count:.0f})")

    # 词表
    vocab = build_vocab(train_texts, MAX_VOCAB_SIZE)
    vocab_size = len(vocab)

    X_train_seq = [text_to_sequence(t, vocab, MAX_SEQ_LEN) for t in train_texts]
    X_val_seq = [text_to_sequence(t, vocab, MAX_SEQ_LEN) for t in val_texts]
    X_test_seq = [text_to_sequence(t, vocab, MAX_SEQ_LEN) for t in test_texts]

    y_train_bin = mlb.transform(train_labels)
    y_val_bin = mlb.transform(val_labels)
    y_test_bin = mlb.transform(test_labels)

    train_dataset = HybridDataset(X_train_seq, y_train_bin)
    val_dataset = HybridDataset(X_val_seq, y_val_bin)
    test_dataset = HybridDataset(X_test_seq, y_test_bin)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 模型
    model = CNN_LSTM_MultiLabel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ===== 训练 + 逐Epoch日志 =====
    log_records = []
    print("\nTraining...")
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

        avg_loss = total_loss / len(train_loader)
        val_p, val_r, val_f1, head_r, tail_r = evaluate(
            model, val_loader, device, mlb, head_indices, tail_indices)

        record = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6),
            "val_precision": round(val_p, 4),
            "val_recall": round(val_r, 4),
            "val_f1": round(val_f1, 4),
            "val_recall_head": round(head_r, 4),
            "val_recall_tail": round(tail_r, 4)
        }
        log_records.append(record)
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss={avg_loss:.4f} | "
              f"Global-R={val_r:.4f} | Head-R={head_r:.4f} | Tail-R={tail_r:.4f}")

    # 保存日志
    with open(OUTPUT_LOG, 'w') as f:
        json.dump({"epochs": EPOCHS, "head_classes": len(head_indices),
                    "tail_classes": len(tail_indices), "median_count": float(median_count),
                    "records": log_records}, f, indent=2)
    print(f"\nTraining log saved to {OUTPUT_LOG}")

    # 最终测试
    print("\n=== Final Test Evaluation ===")
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            preds = (probs > PREDICT_THRESHOLD).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='micro', zero_division=0)
    print(f"  Micro-P: {precision:.4f}  Micro-R: {recall:.4f}  Micro-F1: {f1:.4f}")
    print(f"  Test F1: {f1*100:.2f}%")

if __name__ == "__main__":
    main()
