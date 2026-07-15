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

# ================= Configuration =================
DATASET_PATH = "./D_BEDR.npz"
MODEL_NAME = "BAAI/bge-base-en-v1.5"
OUTPUT_PATH = "cti_model_20k_finetuned"

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5

def load_and_balance_data(path):
    print(f">>> 📂 Loading and Balancing Data from {path}...")
    data = np.load(path, allow_pickle=True)
    
    # 1. Extract texts and labels
    # Note: assumes texts are stored under the 'texts' key
    texts = data['texts']
    labels = data['labels']
    
    # Decode (if bytes)
    decoded_texts = []
    for t in texts:
        if isinstance(t, bytes):
            decoded_texts.append(t.decode('utf-8'))
        else:
            decoded_texts.append(str(t))
            
    # 2. Group by class
    groups = defaultdict(list)
    for text, label in zip(decoded_texts, labels):
        groups[label].append(text)
        
    print(f"   Original: {len(decoded_texts)} samples, {len(groups)} classes.")
    
    # 3. Oversampling — critical step
    # Target: upsample each class to TARGET_COUNT
    TARGET_COUNT = 80 
    balanced_pairs = []
    
    print(f"   ⚖️ Balancing classes to target count: {TARGET_COUNT}...")
    
    for label, samples in groups.items():
        # If not enough samples, randomly resample with replacement up to TARGET_COUNT
        curr_samples = samples.copy()
        while len(curr_samples) < TARGET_COUNT:
            curr_samples.append(random.choice(samples)) # random resampling
            
        # Build training pairs (Anchor, Positive) from the same class
        for _ in range(TARGET_COUNT): 
            # Randomly pick two
            a = random.choice(curr_samples)
            b = random.choice(curr_samples)
            # Avoid self-pairing unless only one sample exists
            if a == b and len(set(curr_samples)) > 1:
                while b == a:
                    b = random.choice(curr_samples)
            
            balanced_pairs.append(InputExample(texts=[a, b]))
            
    print(f"✅ Data Prepared. Total Training Pairs: {len(balanced_pairs)}")
    return balanced_pairs

def train():
    # 1. Device check
    if torch.backends.mps.is_available():
        device = "mps"
        print(">>> 🚀 MPS Acceleration Enabled")
    else:
        device = "cpu"
        print(">>> ⚠️ Using CPU")

    # 2. Prepare data
    train_examples = load_and_balance_data(DATASET_PATH)
    
    # Optional validation split; full training here to maximize data
    random.shuffle(train_examples)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE, num_workers=0)

    # 3. Load base model
    print(f">>> Loading base model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = 512

    # 4. Loss function — MultipleNegativesRankingLoss treats other in-batch pairs as negatives
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 5. Start training
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