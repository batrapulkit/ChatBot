import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import numpy as np

# Override the DepthwiseConv2D layer to handle custom deserialization
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Remove 'groups' argument from the constructor to avoid the error
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)

# Custom function to load the model with a custom layer handling
def load_custom_model(model_path):
    # Define custom objects for loading the model
    custom_objects = {
        'DepthwiseConv2D': CustomDepthwiseConv2D  # Use the custom class
    }
    
    # Load the model with custom objects to handle the layer deserialization
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the model
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
