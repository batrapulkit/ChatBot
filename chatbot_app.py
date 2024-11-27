import streamlit as st
import tensorflow as tf
import numpy as np

# Streamlit page configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

# Heading and description
st.title("AI Chatbot")
st.write(
    """
    This is an AI-powered chatbot. Type your message, and the model will respond!
    """
)

# Load pre-trained model
@st.cache_resource
def load_chatbot_model():
    # Replace 'your_chatbot_model_path' with the path to your trained model
    model_path = "model.h5"  # Update with your model path
    model = tf.keras.models.load_model(model_path)
    return model

# Initialize the model
model = load_chatbot_model()

# Define input preprocessing function (example)
def preprocess_input(input_text):
    # Preprocess the user input (tokenization, padding, etc.)
    # Modify according to your model's requirements
    # For example, if you're using tokenization:
    input_seq = tokenizer.texts_to_sequences([input_text])  # Example for tokenizer
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=100)  # Adjust maxlen as needed
    return input_seq

# Define output postprocessing function (example)
def postprocess_output(model_output):
    # Convert the model output into a human-readable form
    # For example, if it's a class prediction:
    response = " ".join(model_output)  # Modify as per your output type
    return response

# Text input for chatbot interaction
st.write("\n\n### Chatbot Interaction")

user_input = st.text_input("Type a message to the chatbot")

if st.button("Send"):
    if user_input:
        # Preprocess input
        processed_input = preprocess_input(user_input)

        # Get the response from the model
        with st.spinner("Getting response..."):
            try:
                response = model.predict(processed_input)  # Assuming the model is set for text-based input
                chatbot_reply = postprocess_output(response)
                st.text_area("Chatbot Reply:", value=chatbot_reply, height=100)
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("Please enter a message to interact with the chatbot.")

# Final footer (optional)
st.write("\n\n---")
st.write("Made with ❤️ by [Your Name]")

