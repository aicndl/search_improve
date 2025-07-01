from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from transformers import MarianMTModel, MarianTokenizer
import torch
import tqdm

# ✅ 1. Check GPU availability
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ PyTorch is not using GPU.")

# ✅ 2. Load dataset and convert to DataFrame
ds = load_dataset("humarin/chatgpt-paraphrases")
df = pd.DataFrame(ds['train']).head(5000)  # Use only first 5000 entries

# ✅ 3. Set up MarianMT translation model (English → Vietnamese)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Helsinki-NLP/opus-mt-en-vi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(
    model_name,
    use_safetensors=True,
    trust_remote_code=True
).to(device)

def batch_translate(texts, batch_size=16):
    results = []
    for i in tqdm.tqdm(range(0, len(texts), batch_size), desc="Batch translating"):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        translated = model.generate(**encoded, max_length=256)
        decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
        results.extend(decoded)
    return results

# ✅ 4. Translate columns using MarianMT
print("Translating 'text' column to Vietnamese...")
df['keyword_1'] = batch_translate(df['text'].tolist())

print("Translating 'paraphrases' column to Vietnamese...")
df['keyword_2'] = batch_translate(df['paraphrases'].tolist())

# ✅ 5. Prepare SentenceTransformer input examples
train_examples = [
    InputExample(texts=[row['keyword_1'], row['keyword_2']], label=1.0)
    for _, row in df.iterrows()
]

# ✅ 6. Load and fine-tune SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
train_loss = losses.CosineSimilarityLoss(model)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

print("Training model on translated Vietnamese paraphrases...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=int(len(train_dataloader) * 0.1),
    output_path="output/finetuned-vietnamese-model"
)

print("✅ Model fine-tuned and saved to: output/finetuned-vietnamese-model")
