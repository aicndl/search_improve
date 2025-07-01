import pandas as pd

# Load the paraphrases output
input_file = "paraphrases_output.csv"
df = pd.read_csv(input_file)

# Group by base_keyword and collect only the 3 paraphrases (skip explanation lines)
cleaned_rows = []
current_keyword = None
paraphrases = []

for idx, row in df.iterrows():
    keyword = row['base_keyword']
    para = str(row['paraphrase'])
    # If this is a new keyword, reset
    if keyword != current_keyword:
        if current_keyword is not None and len(paraphrases) == 3:
            for p in paraphrases:
                cleaned_rows.append({"base_keyword": current_keyword, "paraphrase": p})
        current_keyword = keyword
        paraphrases = []
    # Skip explanation lines (they usually start with 'Dưới đây' or similar)
    if para.startswith("Dưới đây") or para.startswith("\"Dưới đây"):
        continue
    # Remove leading numbering (e.g., '1. ', '2. ', etc.)
    para = para.lstrip('1234567890. -•–').strip()
    if para:
        paraphrases.append(para)
# Add the last group
if current_keyword is not None and len(paraphrases) == 3:
    for p in paraphrases:
        cleaned_rows.append({"base_keyword": current_keyword, "paraphrase": p})

# Save cleaned output
output_file = "paraphrases_output_cleaned.csv"
pd.DataFrame(cleaned_rows).to_csv(output_file, index=False, encoding='utf-8')
print(f"✅ Cleaned paraphrases saved to {output_file}")
