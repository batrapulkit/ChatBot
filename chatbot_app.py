import streamlit as st
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D

# Create a custom version of DepthwiseConv2D that doesn't take `groups`
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Remove the 'groups' argument if it exists
        super().__init__(*args, **kwargs)

# Custom model loading
def load_model_with_custom_layer(model_path):
    # Load the model, ensuring that our custom DepthwiseConv2D is used
    custom_objects = {
        'DepthwiseConv2D': CustomDepthwiseConv2D
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Load the model in your Streamlit app
model = load_model_with_custom_layer('model.h5')

# # Load the chatbot model
# model = load_model("model.h5")

# Preprocessing and postprocessing functions
def preprocess_input(user_input):
    # Add your input preprocessing logic (e.g., tokenization, padding)
    return user_input

def postprocess_output(model_output):
    # Add your output postprocessing logic (e.g., decoding)
    return model_output

# Streamlit app
st.title("Chatbot")
st.write("Interact with the chatbot by typing your message below:")

# User input
user_input = st.text_input("Your Message:")

if st.button("Send"):
    if user_input:
        processed_input = preprocess_input(user_input)
        response = model.predict([processed_input])  # Update based on your model's input format
        chatbot_reply = postprocess_output(response)
        st.text_area("Chatbot Reply:", value=chatbot_reply, height=100)
    else:
        st.warning("Please enter a message!")
