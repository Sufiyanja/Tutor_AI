import os
from huggingface_hub import InferenceClient
import streamlit as st
from dotenv import load_dotenv
import asyncio

# Load environment variables from .env file
load_dotenv()

# Fetch the Hugging Face API key from .env
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    st.error("Hugging Face API key is missing. Please check your .env file and reload the app.")
    st.stop()

# Initialize Hugging Face client
client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

# Cache Hugging Face model calls for repeated queries
@st.cache_data
def query_model(model_name, messages, max_tokens):
    """
    Sends a query to the Hugging Face model and returns the response.
    Cached for faster repeated queries.
    """
    try:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise RuntimeError(f"Error querying the model: {e}")

# Streamlit App UI
st.title("🤖 EDU CHAT-BOT")
st.subheader("An AI Assistant for Education")
st.caption("Powered by Hugging Face models for intelligent, domain-specific responses.")

# Model selection dropdown
available_models = {
    "Qwen 2.5 72B Instruct": "Qwen/Qwen2.5-72B-Instruct",
    "Llama 3.3 70B Instruct": "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
}
selected_model_name = st.selectbox("Select an AI Model:", list(available_models.keys()))
selected_model = available_models[selected_model_name]

# User input section
user_input = st.text_area("💬 Your Query:", placeholder="Type your question here...", height=100)
max_tokens = st.slider("🎯 Max Tokens:", 50, 200, 150, help="Set a limit to control response time.")
generate_button = st.button("🚀 Generate Response")

# Asynchronous response generation for faster interaction
async def generate_response(model_name, user_query, tokens):
    """
    Asynchronously queries the Hugging Face model.
    """
    try:
        messages = [{"role": "user", "content": user_query}]
        with st.spinner(f"Generating response using '{model_name}'..."):
            completion = query_model(model_name, messages, tokens)
            response = completion.choices[0].message.content if completion.choices else None
            return response or "No response received. Try again."
    except Exception as e:
        return f"Error: {e}"

# Response generation logic
if generate_button and user_input.strip():
    # Trigger asynchronous task
    response = asyncio.run(generate_response(selected_model, user_input.strip(), max_tokens))
    st.subheader("🧠 AI's Response:")
    st.success(response)
elif generate_button:
    st.warning("Please enter a query to proceed.")

# Footer
st.write("---")
st.caption("🔗 Powered by Hugging Face API and Multiple AI Models")
