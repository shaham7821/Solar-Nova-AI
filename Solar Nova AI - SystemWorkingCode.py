#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = load_model(r"C:\Users\shaha\OneDrive\Desktop\project\dataset\FineTuned_Mobilenet.h5")  # Replace 'your_model.h5' with the path to your model file

# Initialize the camera
camera = cv2.VideoCapture(0)  # Use 0 for the default camera, change if necessary

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    # Preprocess the frame
    img = cv2.resize(frame, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions every two seconds
    if time.time() % 2 < 0.1:  # Check if it's been 2 seconds (with some tolerance)
        predictions = model.predict(img_array)

        # Decode predictions
        class_labels = ['clean', 'dirt']  # Define your class labels here
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]
        confidence = predictions[0][predicted_class]

        if confidence > 0.9:
            output_label = "dirt"
        else:
            output_label = "clean"

        print("Predicted class:", output_label)
        print("Confidence:", confidence)

    # Display the resulting frame
    cv2.imshow('Camera', frame)
    
    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()

