import json
import unidecode
print("Script started")
# Load the search results
with open("search_results_single.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# Build the dictionary: normalized keyword -> list of expected values
search_dict = {}

for entry in results:
    keyword = entry["keyword"].strip().lower()
    norm_keyword = unidecode.unidecode(keyword)
    expected = entry["expected"]

    # Add to dictionary, allowing multiple expected values per normalized keyword
    if norm_keyword not in search_dict:
        search_dict[norm_keyword] = set()
    search_dict[norm_keyword].add(expected)

# Convert sets to lists for JSON serialization
search_dict = {k: list(v) for k, v in search_dict.items()}

# Save the dictionary
with open("search_dict.json", "w", encoding="utf-8") as f:
    json.dump(search_dict, f, ensure_ascii=False, indent=2)

print(f"✅ Dictionary created with {len(search_dict)} normalized keywords. Saved to search_dict.json.")

# Example usage function
def search_with_dict(query, search_dict):
    norm_query = unidecode.unidecode(query.strip().lower())
    return search_dict.get(norm_query, [])

# Example: test a query
test_query = "chuyen tien"
with open("search_dict.json", "r", encoding="utf-8") as f:
    loaded_dict = json.load(f)
print(f"Results for '{test_query}':", search_with_dict(test_query, loaded_dict))