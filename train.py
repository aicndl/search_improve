from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
import torch
import tqdm

# === STEP 1: Translate Dataset ===

def translate_batch(texts, model, tokenizer, device, src_lang="en", tgt_lang="vi", batch_size=16):
    results = []
    prefix = f"translate {src_lang} to {tgt_lang}: "
    for i in tqdm.tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = [prefix + text for text in texts[i:i+batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        outputs = model.generate(**inputs, max_length=256)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
    return results

print(torch.cuda.is_available())         # should be True
print(torch.cuda.get_device_name(0))     # should show your GPU model
print("🚀 Checking GPU...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

print("📥 Loading dataset...")
ds = load_dataset("DiligentPenguinn/vietnamese-author-styles-paraphrased")

print("🌐 Loading translation model (envit5)...")
translator_name = "VietAI/envit5-translation"
tokenizer = AutoTokenizer.from_pretrained(translator_name)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(
    translator_name,
    use_safetensors=True,            # ✅ this avoids unsafe .bin loading
    trust_remote_code=True           # required by some community models
).to(device)

print("🔄 Translating text column...")
df['keyword_1'] = translate_batch(df['text'].tolist(), translator_model, tokenizer, device)

print("🔄 Translating paraphrases column...")
df['keyword_2'] = translate_batch(df['paraphrases'].tolist(), translator_model, tokenizer, device)

df.to_csv("translated_envi_paraphrases.csv", index=False, encoding="utf-8-sig")
print("💾 Saved translated data to translated_envi_paraphrases.csv")

# === STEP 2: Train SentenceTransformer ===

print("🧠 Preparing training data...")
train_samples = [
    InputExample(texts=[row['keyword_1'], row['keyword_2']], label=1.0)
    for _, row in df.iterrows()
]
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=32)

print("📦 Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
train_loss = losses.MultipleNegativesRankingLoss(model)

print("🏋️ Starting training...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    show_progress_bar=True
)

model.save("search_model_vi")
print("✅ Model saved to folder: search_model_vi")