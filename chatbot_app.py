import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Override the DepthwiseConv2D class to handle the 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)

# Function to load the model with custom layers
def load_custom_model(model_path):
    custom_objects = {
        'DepthwiseConv2D': CustomDepthwiseConv2D
    }
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Initialize Tokenizer
tokenizer = Tokenizer(num_words=10000)  # Adjust vocab size as per your model

# Load the model
model = load_custom_model('model.h5')

if model:
    st.title("Chatbot")
    st.subheader("AI-Powered Chatbot")

    # User input
    user_input = st.text_input("You: ", "")

    if user_input:
        try:
            # Tokenize and preprocess the user input
            sequences = tokenizer.texts_to_sequences([user_input])
            maxlen = model.input_shape[1]  # Match the model's expected input length
            padded_sequences = pad_sequences(sequences, maxlen=maxlen)
            padded_sequences = np.array(padded_sequences, dtype=np.int32)

            # Predict the response
            response = model.predict(padded_sequences)

            # Display the response
            st.write(f"Bot: {response[0]}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.error("Failed to load the model.")
