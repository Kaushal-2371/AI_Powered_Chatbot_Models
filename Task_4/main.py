import streamlit as st
import faiss, pickle
from sentence_transformers import SentenceTransformer
from model_utils import ask_llama3
from langdetect import detect

st.title("Smart Chatbot 🤖")
query = st.text_input("Ask something:")

def detect_language(text):
    try:
        lang_code = detect(text)
        lang_map = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'hi': 'Hindi',
            'de': 'German'
        }
        return lang_map.get(lang_code, 'English')  # Default to English
    except:
        return 'English'


if query:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("vector_store/faiss_index")
    with open("vector_store/docs.pkl", "rb") as f:
        docs = pickle.load(f)

    q_embed = model.encode([query])
    D, I = index.search(q_embed, k=3)
    relevant_chunks = [docs[i] for i in I[0]]

    context = "\n\n".join(relevant_chunks)
    language = detect_language(query)
    prompt = f"Respond in {language}. Use the following context if needed:\n\n{context}\n\nUser: {query}"
    response = ask_llama3(prompt,context)
    st.markdown(response)


