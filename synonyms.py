import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load the file and clean commas from lines
with open("viet74k.csv", "r", encoding="utf-8") as f:
    words = [line.strip().replace(",", "") for line in f if line.strip()]

# Remove duplicates and empty lines
words = list(set(filter(None, words)))

# Load multilingual embedding model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Encode all words
embeddings = model.encode(words, convert_to_tensor=True, show_progress_bar=True)

# Compute top 5 nearest words for each word
results = []
for idx, word in enumerate(words):
    scores = util.cos_sim(embeddings[idx], embeddings)[0]
    top_indices = scores.topk(6).indices.tolist()
    top_synonyms = [words[i] for i in top_indices if i != idx][:5]
    results.append({"word": word, "synonyms": ", ".join(top_synonyms)})

# Save to CSV
output_df = pd.DataFrame(results)
output_df.to_csv("viet74k_synonyms_cleaned.csv", index=False, encoding="utf-8-sig")
print("✅ Synonyms saved to viet74k_synonyms_cleaned.csv")
