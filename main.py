# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
import cv2
import os
import time
import streamlit as st
import numpy as np
# Load the cascade
face_cascade = cv2.CascadeClassifier('C:/Users/Ameni/PycharmProjects/face recognition/haarcascade_frontalface_default.xml')

# File uploader to allow user to upload their image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    min_neighbors = st.slider('minNeighbors', 1, 10, 5)
    scale_factor = st.slider('scaleFactor', 1.1, 2.0, 1.1)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30))

    # Create a directory to save the images
    save_directory = 'path/to/save/directory'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Choose the color of the rectangles
    rectangle_color = st.color_picker('Rectangle Color', '#00FF00')

    # Convert the color from hex format to BGR format
    rectangle_color_bgr = tuple(int(rectangle_color[i:i+2], 16) for i in (1, 3, 5))

    # Draw bounding boxes around the detected faces and save the images
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(image, (x, y), (x+w, y+h), rectangle_color_bgr, 3)

        # Save the image with detected faces
        timestamp = int(time.time())
        save_path = os.path.join(save_directory, f'detected_faces_{timestamp}_{i}.jpg')
        cv2.imwrite(save_path, image[y:y+h, x:x+w])

        print(f"Image with detected faces saved at: {save_path}")

    # Display the output image
    st.image(image, channels='BGR', caption='Face Detection')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
