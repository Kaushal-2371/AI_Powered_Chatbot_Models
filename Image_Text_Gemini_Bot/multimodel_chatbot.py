import google.generativeai as genai
import streamlit as st
from PIL import Image
import requests
import io

# Configure Google Generative AI
genai.configure(api_key="")  # Add your Google API key

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')  

# Function to generate text response
def generate_text_response(prompt):
    response = model.generate_content(prompt)
    return response.text

# Function to generate image response
def generate_image_response(prompt):
    response = requests.get(f"https://picsum.photos/400/300?random={prompt}")
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    return None

# Streamlit App
st.title("Multi-Modal Chatbot")
st.write("This chatbot can handle both text and image inputs and generate relevant responses.")

# Input options
input_type = st.radio("Choose input type:", ["Text", "Image"])

if input_type == "Text":
    # Text input
    text_input = st.text_area("Enter your text prompt:")
    if st.button("Generate Text Response"):
        if text_input.strip() == "":
            st.warning("Please enter a text prompt.")
        else:
            response = generate_text_response(text_input)
            st.subheader("Text Response:")
            st.write(response)

elif input_type == "Image":
    # Image input
    image_input = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
    if image_input is not None:
        image = Image.open(image_input)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("Generate Image Response"):
            response = generate_image_response("Generate an image based on the uploaded image.")
            if response:
                st.subheader("Generated Image:")
                st.image(response, use_container_width=True)
            else:
                st.error("Failed to generate an image.")