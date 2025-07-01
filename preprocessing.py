import pandas as pd
print("Script started")
print("Loading combined pairs CSV file...")
df = pd.read_csv("C:\\Users\\vunhl\\Downloads\\search-improve\\combined_pairs.csv")

print(f"Loaded {len(df)} rows.")

processed = []
for i, row in enumerate(df.iterrows()):
    if i % 50 == 0:
        print(f"Processing row {i}...")
    _, row = row
    word1, word2, relation = row['word1'], row['word2'], row['relation']
    if relation == "SYN":
        label = 1.0
    elif relation == "ANT":
        label = 0.0
    else:
        continue  # skip if relation is not SYN or ANT
    processed.append({'word1': word1, 'word2': word2, 'label': label})

print(f"Finished processing. Total processed pairs: {len(processed)}")
# Save to CSV for later use
processed_df = pd.DataFrame(processed)
processed_df.to_csv("combined_pairs_processed.csv", index=False)
print("Saved processed pairs to combined_pairs_processed.csv")