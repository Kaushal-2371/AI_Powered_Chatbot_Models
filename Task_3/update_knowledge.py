from sentence_transformers import SentenceTransformer
import faiss, os
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")
index_file = "vector_store/faiss_index"
data_dir = "data"

docs = []
for fname in os.listdir(data_dir):
    with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
        docs.append(f.read())

embeddings = model.encode(docs, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, index_file)
with open("vector_store/docs.pkl", "wb") as f:
    pickle.dump(docs, f)
