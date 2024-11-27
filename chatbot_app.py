import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Function to load the model
def load_custom_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Tokenizer setup (should be the same tokenizer used during training)
tokenizer = Tokenizer(num_words=10000)  # Adjust vocab size as per your model

# Load model (text-based model, not image model)
model = load_custom_model('chatbot_model.h5')

# Check if the model is loaded
if model is not None:
    st.title("Chatbot")

    # User input
    user_input = st.text_input("You: ", "")

    if user_input:
        try:
            # Tokenize the input text
            sequences = tokenizer.texts_to_sequences([user_input])

            # Pad the sequences to match model input shape
            padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen based on your model
            padded_sequences = np.array(padded_sequences, dtype=np.int32)

            # Predict the response
            response = model.predict(padded_sequences)

            # Display the response
            st.write(f"Bot: {response[0]}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.write("Failed to load the model.")
