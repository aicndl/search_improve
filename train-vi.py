import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import torch
import os

print("📥 Loading dataset...")

# === Settings ===
model_name = "all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 10  # Increased for more vigorous training
batch_size = 8  # Smaller batch size for more gradient updates
output_dir = "search_model_combined_pairs"
learning_rate = 2e-5  # Lower learning rate for finer updates
patience = 5  # Early stopping patience

# === Load dataset ===
df = pd.read_csv(r"C:\Users\vunhl\Downloads\search-improve\train.csv")
print(f"Loaded {len(df)} pairs.")

# === Shuffle and split into train/validation (80/20 split) ===
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split = int(0.8 * len(df))
df_train = df.iloc[:split].reset_index(drop=True)
df_val = df.iloc[split:].reset_index(drop=True)
print(f"Train size: {len(df_train)}, Validation size: {len(df_val)}")

# === Clean Data ===
print("🧹 Cleaning data...")
df_train = df_train.dropna(subset=['word1', 'word2'])
df_val = df_val.dropna(subset=['word1', 'word2'])
df_train['word1'] = df_train['word1'].astype(str)
df_train['word2'] = df_train['word2'].astype(str)
df_val['word1'] = df_val['word1'].astype(str)
df_val['word2'] = df_val['word2'].astype(str)
print(f"After cleaning - Train size: {len(df_train)}, Validation size: {len(df_val)}")

# === Prepare InputExamples ===
train_samples = [
    InputExample(texts=[row['word1'], row['word2']], label=float(row['label']))
    for _, row in df_train.iterrows()
]
val_samples = [
    InputExample(texts=[row['word1'], row['word2']], label=float(row['label']))
    for _, row in df_val.iterrows()
]

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(val_samples, name="val")

# === Load or resume model ===
if os.path.exists(output_dir):
    print(f"🔄 Resuming from checkpoint: {output_dir}")
    model = SentenceTransformer(output_dir, device=device)
else:
    print(f"🆕 Loading base model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

train_loss = losses.CosineSimilarityLoss(model)

# === Fit model ===
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=epochs,
    warmup_steps=int(0.1 * len(train_dataloader)),
    evaluation_steps=max(1, len(train_dataloader) // 2),
    show_progress_bar=True,
    output_path=output_dir,
    save_best_model=True,
    optimizer_params={'lr': learning_rate}
)

print(f"✅ Training complete. Model saved to: {output_dir}")