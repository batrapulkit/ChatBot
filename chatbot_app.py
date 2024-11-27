import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the model
def load_custom_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Tokenizer setup (Adjust as needed)
tokenizer = Tokenizer(num_words=10000)

# Load model
model = load_custom_model('model.h5')

if model:
    st.title("Chatbot")
    user_input = st.text_input("You: ", "")

    if user_input:
        try:
            # Preprocess input
            sequences = tokenizer.texts_to_sequences([user_input])
            maxlen = model.input_shape[1]  # Match sequence length to model input
            padded_sequences = pad_sequences(sequences, maxlen=maxlen)

            # Predict response
            response = model.predict(padded_sequences)
            st.write(f"Bot: {response}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.error("Failed to load the model.")
