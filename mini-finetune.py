import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Load your data
df = pd.read_csv("vietnamese_synonym_dataset_prompt_generated.csv")
train_examples = [
    InputExample(texts=[row['keyword_1'], row['keyword_2']], label=float(row['label']))
    for _, row in df.iterrows()
]

# Use your specified model
model = SentenceTransformer("all-MiniLM-L6-v2")
train_loss = losses.CosineSimilarityLoss(model)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Fine-tune using the fit method
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=int(len(train_dataloader) * 0.1),
    output_path="output/finetuned-vietnamese-model"
)

print("Model fine-tuned and saved to output/finetuned-vietnamese-model")