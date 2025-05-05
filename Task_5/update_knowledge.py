import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

# Paths
data_path = 'data/arxiv_cs_subset_8000.txt'
vector_store_path = 'vector_store/'

# Model for embeddings
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Read data
print("Loading data...")
with open(data_path, 'r', encoding='utf-8') as f:
    texts = f.read().split('\n\n')  # Each paper separated by blank line

print(f"Total papers to embed: {len(texts)}")

# Parameters
batch_size = 800  # You can increase if you have faster processing capabilities.  

all_embeddings = []

print("Generating embeddings in batches...")
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]
    batch_embeddings = embedder.encode(batch_texts, convert_to_numpy=True)
    all_embeddings.append(batch_embeddings)
    print(f"Embedded {i + len(batch_texts)} / {len(texts)}")

# Merge all batches
embeddings = np.vstack(all_embeddings)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
if not os.path.exists(vector_store_path):
    os.makedirs(vector_store_path)

faiss.write_index(index, os.path.join(vector_store_path, 'faiss.index'))

# Save texts separately
with open(os.path.join(vector_store_path, 'texts.pkl'), 'wb') as f:
    pickle.dump(texts, f)

print("FAISS vector store updated successfully!")
