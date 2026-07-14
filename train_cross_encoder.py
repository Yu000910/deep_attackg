"""
Cross-Encoder Fine-Tuning on BEDR dataset.
Uses InfoNCE (MultipleNegativesRankingLoss) with resampling to 80 samples/class.
Fixed random seeds for reproducibility.
"""
import numpy as np
import torch
import random
import os
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, CrossEncoder, losses
from collections import defaultdict

# ================= Reproducibility (Fixed Seeds) =================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================= Configuration =================
user_home = os.path.expanduser("~")
OUTPUT_PATH = os.path.join(user_home, "Desktop", "cti_reranker_final")

DATASET_PATH = "./D_BEDR.npz"
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE = 16
EPOCHS = 3
TARGET_COUNT = 80  # Resample to 80 per class (matching BiEncoder)


def prepare_infonce_data(path):
    """Prepare training pairs with resampling for InfoNCE loss.
    Each class is resampled to TARGET_COUNT, then positive pairs are formed.
    InfoNCE treats other pairs in the batch as negatives (no explicit negative labels needed).
    """
    print(f">>> Loading Data from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.load(path, allow_pickle=True)
    texts = [str(t) if not isinstance(t, bytes) else t.decode('utf-8') for t in data['texts']]
    labels = data['labels']

    # Group by class
    groups = defaultdict(list)
    for t, l in zip(texts, labels):
        groups[l].append(t)

    print(f"   Original: {len(texts)} samples, {len(groups)} classes.")

    # Resample each class to TARGET_COUNT
    print(f"   Resampling classes to target count: {TARGET_COUNT}...")
    balanced_pairs = []

    for label, samples in groups.items():
        curr_samples = samples.copy()
        while len(curr_samples) < TARGET_COUNT:
            curr_samples.append(random.choice(samples))

        for _ in range(TARGET_COUNT):
            a = random.choice(curr_samples)
            b = random.choice(curr_samples)
            if a == b and len(set(curr_samples)) > 1:
                while b == a:
                    b = random.choice(curr_samples)
            balanced_pairs.append(InputExample(texts=[a, b]))

    print(f"   Prepared {len(balanced_pairs)} training pairs (InfoNCE).")
    return balanced_pairs


def train_reranker():
    print(f">>> Output path: {OUTPUT_PATH}")
    if os.path.exists(OUTPUT_PATH):
        print("   (folder exists, will overwrite)")
    else:
        os.makedirs(OUTPUT_PATH)

    # Write test
    test_file = os.path.join(OUTPUT_PATH, "write_test.txt")
    try:
        with open(test_file, 'w') as f:
            f.write("Test write permission OK.")
        print("   Write permission test: OK.")
    except Exception as e:
        print(f"   ERROR: Cannot write to output folder!\n{e}")
        return

    # Load model
    print(f">>> Loading Model: {MODEL_NAME}...")
    model = CrossEncoder(MODEL_NAME, num_labels=1)

    # Prepare data with resampling
    train_samples = prepare_infonce_data(DATASET_PATH)
    random.shuffle(train_samples)

    train_dataloader = DataLoader(
        train_samples,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=model.smart_batching_collate
    )

    # InfoNCE loss (MultipleNegativesRankingLoss)
    # Works with CrossEncoder: text pairs in batch, other pairs serve as negatives
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Train
    print(f">>> Starting Fine-Tuning ({EPOCHS} epochs, InfoNCE loss)...")

    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=EPOCHS,
            warmup_steps=int(len(train_dataloader) * 0.1),
            output_path=OUTPUT_PATH,
            show_progress_bar=True
        )
        print("\n   Training Finished.")

        # Manual save (belt and suspenders)
        print("   Forcing Manual Save...")
        model.save(OUTPUT_PATH)

        if os.path.exists(os.path.join(OUTPUT_PATH, "config.json")):
            print(f"\n   SUCCESS: Model saved to: {OUTPUT_PATH}")
        else:
            print(f"\n   WARNING: Training completed but config.json not found. Check path.")

    except Exception as e:
        print(f"\n   ERROR: Training crashed: {e}")


if __name__ == "__main__":
    train_reranker()
