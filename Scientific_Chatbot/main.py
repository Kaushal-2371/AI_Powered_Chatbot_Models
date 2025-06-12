import streamlit as st
import faiss, pickle
from sentence_transformers import SentenceTransformer
from model_utils import generate_response

st.set_page_config(page_title="Expert Chatbot", layout="wide")
st.title("Expert Chatbot")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input from user
query = st.chat_input("Ask me anything...")

if query:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("vector_store/faiss_index")
    with open("vector_store/texts.pkl", "rb") as f:
        docs = pickle.load(f)

    q_embed = model.encode([query])
    D, I = index.search(q_embed, k=3)
    relevant_chunks = [docs[i] for i in I[0]]

    # Build context from chat history + relevant info
    previous_turns = "\n".join([f"User: {u}\nBot: {b}" for u, b in st.session_state.chat_history])
    context = previous_turns + "\n\n" + "\n".join(relevant_chunks)

    # Get response
    response = generate_response(query, context)

    # Save to history
    st.session_state.chat_history.append((query, response))

# Display chat
for user_msg, bot_msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)
