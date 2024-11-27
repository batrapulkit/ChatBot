import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Override the DepthwiseConv2D class to handle the 'groups' argument
class CustomDepthwiseConv2D(layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Remove the 'groups' argument from kwargs to avoid errors
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)

# Function to load model with custom layers
def load_custom_model(model_path):
    custom_objects = {
        'DepthwiseConv2D': CustomDepthwiseConv2D  # Use the custom class
    }
    
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load model
model = load_custom_model('model.h5')

# Tokenizer setup (adjust vocab size as per your model)
tokenizer = Tokenizer(num_words=10000)  # Adjust vocab size as needed

# Check if model is loaded
if model is not None:
    st.title("Chatbot")

    # User input
    user_input = st.text_input("You: ", "")

    if user_input:
        try:
            # Tokenize and pad the input
            sequences = tokenizer.texts_to_sequences([user_input])
            padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as needed
            padded_sequences = np.array(padded_sequences, dtype=np.int32)

            # Predict the response
            response = model.predict(padded_sequences)

            # Display the response
            st.write(f"Bot: {response[0]}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.write("Failed to load the model.")
