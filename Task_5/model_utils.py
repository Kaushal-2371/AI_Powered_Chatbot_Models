import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# Load FAISS index
vector_store_path = 'vector_store/'
index = faiss.read_index(vector_store_path + 'faiss_index')

# Load texts
with open(vector_store_path + 'texts.pkl', 'rb') as f:
    texts = pickle.load(f)

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def get_top_k_chunks(query, k=3):
    """Find top k similar documents for a query."""
    query_vector = embedder.encode([query])
    distances, indices = index.search(np.array(query_vector), k)
    retrieved_texts = [texts[idx] for idx in indices[0]]
    return retrieved_texts

def generate_response(user_query,context):
    """Retrieve relevant info and generate final response using Ollama."""
    relevant_chunks = get_top_k_chunks(user_query)
    context = "\n\n".join(relevant_chunks)

    final_prompt = f"""You are a domain expert in scientific research. 
You have access to the following papers:\n{context}\n\n
Now answer the following user query in detail and simple terms:\n
{user_query}
"""

    # Talking to the local Ollama server
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": final_prompt}])

    return response['message']['content']
