import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.models import load_model
import numpy as np

# Custom function to load the model and handle DepthwiseConv2D deserialization issue
def load_custom_model(model_path):
    # Define a custom_objects dictionary to handle custom layer deserialization
    custom_objects = {
        'DepthwiseConv2D': DepthwiseConv2D
    }

    # Try loading the model with custom objects
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load your model
model = load_custom_model('model.h5')

# Streamlit interface for chat input
if model is not None:
    st.title("Chatbot")
    
    # Placeholder for chatbot conversation
    st.write("Ask me anything...")

    # User input
    user_input = st.text_input("You: ", "")

    if user_input:
        # Example pre-processing (adjust as per your tokenizer and model)
        # Assuming you have a tokenizer or preprocessing pipeline, you can modify accordingly
        processed_input = np.array([user_input])  # Example preprocessing step
        
        # Get prediction from model (adjust for your model's specific input/output format)
        response = model.predict(processed_input)
        
        # Example output (adjust as per your model's output)
        st.write(f"Bot: {response[0]}")
else:
    st.write("Failed to load the model.")
