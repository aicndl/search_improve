import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import unidecode
import json
from pyvi.ViTokenizer import tokenize
# Load your test file
test_df = pd.read_csv("search-test.csv")

# Normalize text (remove accents, lowercase)
def normalize(text):
    return unidecode.unidecode(text.strip().lower())

# Build corpus using all unique base_keywords
base_keywords = test_df['base_keyword'].unique()
corpus = [normalize(text) for text in base_keywords]
corpus_ids = list(range(len(base_keywords)))

# Load embedding model

model = SentenceTransformer('NghiemAbe/Vi-Legal-Bi-Encoder-v2')
print("Encoding base keyword corpus...")
corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)

# Search each keyword and compare to ground truth

# After collecting results
results = []
for _, row in test_df.iterrows():
    query = normalize(row['keyword'])
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_id = torch.argmax(scores).item()
    matched_base = base_keywords[top_id]
    results.append({
        "keyword": row['keyword'],
        "expected": row['base_keyword'],
        "actual": matched_base,
        "is_hit": matched_base == row['base_keyword'],
        "case_type": row['case_type']  # ✅ added case_type here
    })


# Save results as JSON
with open("search_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ JSON output saved as search_results.json")


# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("search_results_self_corpus.csv", index=False)
print("✅ Done. Results saved to search_results_self_corpus.csv")
