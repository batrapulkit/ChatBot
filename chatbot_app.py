import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model (add custom objects if needed)
def load_custom_model(model_path):
    custom_objects = {
        'DepthwiseConv2D': CustomDepthwiseConv2D  # Add any custom layers here
    }
    
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

from tensorflow.keras import layers

# Define the custom layer (example)
class CustomDepthwiseConv2D(layers.Layer):
    def __init__(self, **kwargs):
        super(CustomDepthwiseConv2D, self).__init__(**kwargs)
        # Define the custom layer's behavior here

    def build(self, input_shape):
        # Initialize any variables or layers
        pass

    def call(self, inputs):
        # Implement the forward pass logic here
        pass

# Load the model with the custom layer
def load_custom_model(model_path):
    custom_objects = {
        'CustomDepthwiseConv2D': CustomDepthwiseConv2D
    }
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None


# Load the model
model = load_custom_model('model.h5')

# Tokenizer setup (adjust according to your trained model's vocab)
tokenizer = Tokenizer(num_words=10000)  # Adjust vocab size according to your model

# Assuming the tokenizer was already fitted with training data
# tokenizer.fit_on_texts(training_data)  # This should be done during training

if model is not None:
    st.title("Chatbot")

    # User input
    user_input = st.text_input("You: ", "")

    if user_input:
        try:
            # Tokenize the input text
            sequences = tokenizer.texts_to_sequences([user_input])
            padded_sequences = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as per your model's expected input shape
            
            # Ensure the data type is correct (e.g., np.int32)
            padded_sequences = np.array(padded_sequences, dtype=np.int32)

            # Get prediction from model
            response = model.predict(padded_sequences)
            st.write(f"Bot: {response[0]}")

        except ValueError as e:
            st.error(f"Error with input data type: {e}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.write("Failed to load the model.")


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
