import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import DepthwiseConv2D

# Function to preprocess user input (you can modify it based on your model)
def preprocess_input(user_input):
    # For example, we can tokenize the input or convert it to a format that the model expects.
    # This is a simple example where we assume the input is already tokenized into a list of integers.
    # Replace this with actual preprocessing for your model.
    tokenized_input = user_input.lower().split()
    return tokenized_input

# Function to post-process the model output (again, modify this based on your model's output)
def postprocess_output(response):
    # Convert the model's response into a string that can be displayed
    # For example, assume the response is a list of probabilities or a string
    return response[0]  # This is just an example, adjust as per your model's response

# Load your pre-trained model
model = load_model('model.h5')  # Ensure the model is available in your project directory

# Streamlit UI
st.title("Chatbot")

user_input = st.text_input("You: ", "")

if st.button("Send"):
    if user_input:
        # Preprocess the input from the user
        processed_input = preprocess_input(user_input)

        # Convert the list to a numpy array (model expects numpy array or tensor)
        max_sequence_length = 100  # Adjust this based on your model's requirements
        input_data = pad_sequences([processed_input], maxlen=max_sequence_length)

        # Get the model's response
        response = model.predict(input_data)

        # Post-process the model's response (adjust as needed)
        chatbot_reply = postprocess_output(response)

        # Display the chatbot's reply
        st.text_area("Chatbot Reply:", value=chatbot_reply, height=100)
    else:
        st.write("Please enter a message.")
