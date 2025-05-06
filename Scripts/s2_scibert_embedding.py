import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Load data
with open("openalex_9000.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

# SciBERT
model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model = model.to(device)
# model.eval().cuda()
model.eval()


# Embedding function
@torch.no_grad()
def embed_text(text):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    # tokens = {k: v.cuda() for k, v in tokens.items()}
    tokens = {k: v.to(device) for k, v in tokens.items()}
    output = model(**tokens)
    cls_embedding = output.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze(0).cpu().numpy()


# Embedding
embeddings = []
for i, paper in enumerate(papers):
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    text = title + " [SEP] " + abstract

    try:
        vec = embed_text(text)
        embeddings.append(vec)
    except Exception as e:
        print(f"Skipped paper {i} due to error: {e}")
        continue

    if (i + 1) % 50 == 0:
        print(f"Embedded {i + 1}/{len(papers)} papers")

# Save as .npy
embeddings_array = np.stack(embeddings)
np.save("sci_bert_embeddings_9000.npy", embeddings_array)
print("All embeddings saved to sci_bert_embeddings_9000.npy")
