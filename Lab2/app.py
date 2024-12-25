import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the TFLite model
def load_model():
    interpreter = tf.lite.Interpreter(model_path="fruit_classification_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Preprocess the uploaded image to fit the model’s input shape
def preprocess_image(image: Image.Image, target_size=(32, 32)):
    image = image.resize(target_size)
    image_array = np.array(image).astype(np.float32)
    
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Make a prediction with the TFLite model
def predict(interpreter, image_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Main Streamlit app
st.title("Fruit Classification")

st.write("Upload an image of a fruit to classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.sidebar.image("tofcv1.png",width=200)

    # Load the model
    interpreter = load_model()

    # Preprocess and predict
    image_array = preprocess_image(image)
    prediction = predict(interpreter, image_array)

    # Assuming the model outputs probabilities for each fruit class
    fruit_classes = ["apple","avocado","banana","cherry","kiwi","mango", "orange","pineapple", "strawberries","watermelon"]  # Update this list with your actual classes
    predicted_class = fruit_classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display the result
    st.write(f"**Predicted Fruit:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")
