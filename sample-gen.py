import pandas as pd
import requests
import time
import csv
import json

# Load model config
with open("model_config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
v_chat_config = config["models"][0]

API_URL = f"{v_chat_config['apiBase']}/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {v_chat_config.get('OPENAI_API_KEY', 'sk-local')}"
}

# Load input data
df = pd.read_csv("search-test.csv")
unique_keywords = df['base_keyword'].dropna().unique()

# Format prompt
def make_prompt(keyword):
    return f"Viết lại 3 câu khác nhau có cùng nghĩa với câu: \"{keyword}\". Trả lời bằng danh sách liệt kê rõ ràng."

# Store results
paraphrase_data = []

for keyword in unique_keywords:
    try:
        prompt = make_prompt(keyword)
        payload = {
            "model": v_chat_config["model"],
            "messages": [
                {"role": "system", "content": "Bạn là một trợ lý ngôn ngữ tiếng Việt, chuyên viết lại câu giữ nguyên nghĩa."},
                {"role": "user", "content": prompt}
            ],
            "temperature": v_chat_config.get("temperature", 0.5),
            "max_tokens": v_chat_config.get("maxTokens", 32000)
        }
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=v_chat_config.get("timeout", 100))
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        # Only keep lines that are actual paraphrases (skip explanation lines)
        lines = [line.strip("-•–. ").strip() for line in content.split("\n") if line.strip()]
        # Remove lines that start with 'Dưới đây' or similar explanations
        paraphrases = [l for l in lines if not l.lower().startswith("dưới đây")][:3]
        for para in paraphrases:
            paraphrase_data.append({"base_keyword": keyword, "paraphrase": para})
        print(f"✅ {keyword} → {len(paraphrases)} paraphrases")
    except Exception as e:
        print(f"❌ {keyword} → Error: {str(e)}")
        continue
    time.sleep(1.2)  # throttle requests to avoid overload

# Save to CSV
output_file = "paraphrases_output.csv"
pd.DataFrame(paraphrase_data).to_csv(output_file, index=False, encoding='utf-8')
print(f"✅ All done! Saved to {output_file}")
