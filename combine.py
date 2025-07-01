import pandas as pd

# Load all three files
adj = pd.read_csv("600_adj_pairs.txt", sep="\t")
verb = pd.read_csv("400_verb_pairs.txt", sep="\t")
noun = pd.read_csv("400_noun_pairs.txt", sep="\t")

# Standardize column names
for df in [adj, verb, noun]:
    df.columns = ["word1", "word2", "relation"]

# Concatenate all pairs
all_pairs = pd.concat([adj, verb, noun], ignore_index=True)

# Save to a single file
all_pairs.to_csv("combined_pairs.csv", index=False, encoding="utf-8")
print(f"✅ Combined {len(all_pairs)} pairs into combined_pairs.csv")