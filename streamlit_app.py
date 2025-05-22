import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

model = load_model("mnist_cnn_model.h5")

st.title("Handwritten Digit Classifier")
st.write("Upload a 28x28 grayscale image of a digit.")

uploaded_file = st.file_uploader("Choose a digit image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(image, caption='Uploaded Image')
    image = image.resize((28, 28))
    image = ImageOps.invert(image)
    img_array = np.array(image).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    st.write(f"### Predicted Digit: {predicted_digit}")

