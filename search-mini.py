import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import unidecode
import json
import numpy as np

# === ✅ Load Dictionary ===
with open("search_dict.json", "r", encoding="utf-8") as f:
    search_dict = json.load(f)

# === ✅ Load Model ===
model_name = "search_model_visim400"
model = SentenceTransformer(model_name, device="cuda")

# === ✅ Normalize Text ===
def normalize(text):
    return unidecode.unidecode(text.strip().lower())

# === ✅ Encoding Function ===
def encode_single(texts, normalize_vec=True):
    if isinstance(texts, str):
        texts = [texts]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    if normalize_vec:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-8, None)
    return torch.tensor(embeddings)

# === ✅ Load Test File ===
test_df = pd.read_csv("search-test.csv")

# === ✅ Build Normalized Corpus ===
base_keywords = test_df['base_keyword'].unique()
corpus = [normalize(text) for text in base_keywords]
corpus_ids = list(range(len(base_keywords)))

print("📦 Encoding corpus...")
corpus_embeddings = encode_single(corpus)

# === ✅ Search: Model first, then Dictionary if needed ===
results = []
for _, row in test_df.iterrows():
    query = normalize(row['keyword'])
    # 1. Try model search first
    query_embedding = encode_single(query)[0]
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    topk_ids = torch.topk(scores, k=5).indices.tolist()
    top5_bases = [base_keywords[i] for i in topk_ids]
    is_hit = row['base_keyword'] in top5_bases
    method = "model"
    # 2. If not satisfactory, try dictionary
    if not is_hit:
        dict_hits = search_dict.get(query, [])
        if dict_hits:
            top5_bases = dict_hits[:5] if len(dict_hits) >= 5 else dict_hits + ["" for _ in range(5 - len(dict_hits))]
            is_hit = row['base_keyword'] in top5_bases
            method = "dictionary"
    results.append({
        "keyword": row['keyword'],
        "expected": row['base_keyword'],
        "top5_actual": top5_bases,
        "is_hit": is_hit,
        "case_type": row['case_type'],
        "method": method
    })

# === ✅ Save Results ===
with open("search_results_single.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

results_df = pd.DataFrame(results)
results_df.to_csv("search_results_single.csv", index=False)
print("✅ Done. Results saved to search_results_single.csv and search_results_single.json")