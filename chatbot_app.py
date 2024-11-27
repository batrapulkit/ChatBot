import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

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

# Tokenizer setup (adjust vocab size as per your model, if it's text-based)
tokenizer = Tokenizer(num_words=10000)  # Adjust vocab size as needed

# Load model
model = load_custom_model('model.h5')

# Check if model is loaded
if model is not None:
    st.title("AI Assistant")
    st.write("Choose between Text-based Chatbot or Image Prediction")

    # Input type selection
    mode = st.radio("Choose Mode", ("Chatbot (Text)", "Image Prediction"))

    if mode == "Chatbot (Text)":
        st.subheader("Chatbot")

        # User input
        user_input = st.text_input("You: ", "")

        if user_input:
            try:
                # Tokenize and preprocess the input text
                sequences = tokenizer.texts_to_sequences([user_input])

                # Pad the sequences to match model input shape
                maxlen = model.input_shape[1] if len(model.input_shape) == 2 else 224  # Adjust if needed
                padded_sequences = pad_sequences(sequences, maxlen=maxlen)
                padded_sequences = np.array(padded_sequences, dtype=np.int32)

                # Predict the response
                response = model.predict(padded_sequences)

                # Process and display the response
                st.write(f"Bot: {response[0]}")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

    elif mode == "Image Prediction":
        st.subheader("Image Prediction")

        # File uploader for image input
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                # Load and preprocess the uploaded image
                img = Image.open(uploaded_file).convert("RGB")
                img = img.resize((224, 224))  # Resize image
                img_array = np.array(img) / 255.0  # Normalize pixel values
                img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

                # Display the uploaded image
                st.image(img, caption="Uploaded Image", use_column_width=True)

                # Predict using the model
                predictions = model.predict(img_array)

                # Display the predictions
                st.write("Predictions:", predictions)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

else:
    st.write("Failed to load the model.")
