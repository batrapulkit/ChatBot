import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle  # To load the tokenizer if it's saved

# Function to load the model
def load_custom_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to load the tokenizer
def load_tokenizer(tokenizer_path):
    try:
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)  # Load the tokenizer using pickle
        return tokenizer
    except Exception as e:
        st.error(f"Error loading the tokenizer: {e}")
        return None

# Load the model and tokenizer
model = load_custom_model('model.h5')
tokenizer = load_tokenizer('tokenizer.pkl')  # Ensure you have the correct path for the tokenizer

# Check if model and tokenizer are loaded
if model is not None and tokenizer is not None:
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

            # Check if the output is a sequence or class prediction
            if len(response.shape) > 1:  # If the output is a sequence, you might need to decode
                response_text = ' '.join([str(int(word)) for word in response[0]])  # Example, adjust as needed
            else:
                response_text = str(response[0])  # For simple classification, convert it to a string

            # Display the response
            st.write(f"Bot: {response_text}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.write("Failed to load the model or tokenizer.")
