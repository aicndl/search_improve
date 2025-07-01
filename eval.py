import pandas as pd
import json

# Load the results file
with open("search_results_single.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results)

# Ensure required fields exist
required_columns = {'keyword', 'expected', 'top5_actual', 'case_type'}
missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing fields in JSON: {missing}")

# Compute is_hit: expected in top5_actual
df['is_hit'] = df.apply(lambda row: row['expected'] in row['top5_actual'], axis=1)

# Calculate metrics per category
metrics = df.groupby('case_type').agg(
    total_cases=('is_hit', 'count'),
    correct_hits=('is_hit', 'sum')
).reset_index()

metrics['hit_rate'] = metrics['correct_hits'] / metrics['total_cases']
metrics = metrics.sort_values(by='hit_rate', ascending=False)

# Save to CSV and JSON
metrics.to_csv("search_evaluation_mini.csv", index=False)
metrics.to_json("search_evaluation_mini.json", orient="records", indent=2, force_ascii=False)

print("✅ Evaluation complete.")
print(metrics)