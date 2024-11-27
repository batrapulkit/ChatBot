import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the model
def load_text_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Initialize Tokenizer
tokenizer = Tokenizer(num_words=10000)

# Load the text-based model
model = load_text_model('model.h5')

if model:
    st.title("Chatbot")
    user_input = st.text_input("You: ", "")

    if user_input:
        try:
            # Tokenize and preprocess user input
            sequences = tokenizer.texts_to_sequences([user_input])
            maxlen = model.input_shape[1]  # Match the model's expected input length
            padded_sequences = pad_sequences(sequences, maxlen=maxlen)

            # Predict the response
            response = model.predict(padded_sequences)

            # Display the response
            st.write(f"Bot: {response[0]}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.error("Failed to load the model.")
