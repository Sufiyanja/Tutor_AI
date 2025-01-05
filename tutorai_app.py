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
st.title("AI Chatbot with Model Selection")
st.subheader("Interact with an AI assistant")
st.caption("Powered by multiple Hugging Face models")

# Model selection dropdown
available_models = {
    "Qwen 2.5 72B Instruct": "Qwen/Qwen2.5-72B-Instruct",
    "Llama 3.3 70B Instruct": "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
}
selected_model_name = st.selectbox("Select a Model:", list(available_models.keys()))
selected_model = available_models[selected_model_name]

# User input section
user_input = st.text_area("Your Query:", placeholder="Type your question here...", height=100)
max_tokens = st.slider("Max Tokens:", 50, 500, 300)
generate_button = st.button("Generate Response")

# Response generation logic
if generate_button:
    if user_input.strip():
        try:
            # Prepare input messages
            messages = [{"role": "user", "content": user_input}]

            # Query the selected Hugging Face model
            st.info(f"Generating response using '{selected_model_name}', please wait...")
            completion = client.chat.completions.create(
                model=selected_model,
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
        st.warning("Please enter a query to proceed.")

# Footer
st.write("---")
st.caption("Powered by Hugging Face API and Multiple AI Models")
