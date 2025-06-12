import streamlit as st
import faiss, pickle
from sentence_transformers import SentenceTransformer
from model_utils import ask_llama3

st.title("Smart Chatbot ")
query = st.text_input("Ask something:")

if query:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("vector_store/faiss_index")
    with open("vector_store/docs.pkl", "rb") as f:
        docs = pickle.load(f)

    q_embed = model.encode([query])
    D, I = index.search(q_embed, k=3)
    relevant_chunks = [docs[i] for i in I[0]]

    context = "\n\n".join(relevant_chunks)
    response = ask_llama3(query, context)
    st.markdown(f"**Bot:** {response}")
