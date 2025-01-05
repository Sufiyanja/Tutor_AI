import os
from huggingface_hub import InferenceClient
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch the Hugging Face API key from the .env file
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    st.error("Hugging Face API key is missing. Please check your .env file.")

# Initialize Hugging Face client
client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

# Streamlit App UI
st.title("Totut AI - Llama 3.3 70B Chatbot")
st.subheader("Interact with your AI assistant")
st.caption("Powered by Llama-3.3-70B-Instruct")

# Input and parameters
user_input = st.text_area("Your Query:", "What is the capital of France?", height=100)
max_tokens = st.slider("Max Tokens:", 50, 500, 300)
generate_button = st.button("Generate Response")

# Response generation
if generate_button:
    if user_input.strip():
        try:
            # Prepare the input messages
            messages = [{"role": "user", "content": user_input}]

            # Send the request to the Hugging Face model
            st.info("Generating response, please wait...")
            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct",
                messages=messages,
                max_tokens=max_tokens
            )

            # Extract and display the response
            response = completion.choices[0].message.content
            st.subheader("AI's Response:")
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid query.")

# Footer
st.write("---")
st.caption("Powered by Hugging Face API and Meta Llama Models")
