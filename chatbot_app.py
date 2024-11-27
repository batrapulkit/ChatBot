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

# Tokenizer setup (adjust vocab size as per your model)
tokenizer = Tokenizer(num_words=10000)  # Adjust vocab size as needed

# Load the model
model = load_custom_model('model.h5')

# Check if model is loaded
if model is not None:
    st.title("Chatbot")
    st.subheader("AI-Powered Chatbot")

    # User input
    user_input = st.text_input("You: ", "")

    if user_input:
        try:
            # Tokenize and preprocess the input text
            sequences = tokenizer.texts_to_sequences([user_input])

            # Pad the sequences to match model input shape
            maxlen = model.input_shape[1]  # Extract the expected sequence length
            padded_sequences = pad_sequences(sequences, maxlen=maxlen)
            padded_sequences = np.array(padded_sequences, dtype=np.int32)

            # Predict the response
            response = model.predict(padded_sequences)

            # Display the response
            st.write(f"Bot: {response[0]}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.write("Failed to load the model.")
