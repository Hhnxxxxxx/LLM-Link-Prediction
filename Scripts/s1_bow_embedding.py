import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load openalex_9000.json
with open("openalex_9000.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

texts = [
    f"{p.get('title', '')} [SEP] {p.get('abstract', '')}"
    for p in papers
]

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    token_pattern=r"(?u)\b\w\w+\b"
)

X = vectorizer.fit_transform(texts)
X_dense = X.toarray().astype("float32")

# Save as .npy
np.save("bow_embeddings_9000.npy", X_dense)

print(f"Done! BoW embedding shape: {X_dense.shape}")
