import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

model = tf.keras.models.load_model("/content/HDR_Model.keras")

st.header('Handwritten Digit Recognizer')

# Use st.file_uploader to allow users to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image using OpenCV
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)[:,:,0]

    # Preprocess the image (if needed)
    # You may need to resize, normalize, or perform other preprocessing steps based on your model's requirements

    # Perform the prediction
    image = np.invert(np.array([image]))
    output = model.predict(image)

    # Display the result
    stn = 'Digit in the image looks like ' + str(np.argmax(output))
    st.markdown(stn)
    
    # Reshape the image to (28, 28) for display
    image = np.reshape(image, (28, 28))

    # Display the reshaped image
    st.image(image, caption='Uploaded Image', use_column_width=True)
