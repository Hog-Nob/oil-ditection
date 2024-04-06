#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit as st

# Set random seeds for NumPy and TensorFlow
np.random.seed(0)

# Load your pre-trained face oil detection model
oil_model = load_model('newmodel.h5')

# Define data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,  # randomly rotate images by up to 20 degrees
    width_shift_range=0.2,  # randomly shift images horizontally by up to 20% of the width
    height_shift_range=0.2,  # randomly shift images vertically by up to 20% of the height
    shear_range=0.2,  # apply shear transformation with a maximum shear angle of 20 degrees
    zoom_range=0.2,  # randomly zoom into images by up to 20%
    horizontal_flip=True,  # randomly flip images horizontally
    vertical_flip=True,  # randomly flip images vertically
    brightness_range=[0.8, 1.2],  # randomly adjust brightness of images
    fill_mode='nearest'  # how to fill in newly created pixels after transformations
)

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(face_img, target_size=(224, 224)):
    # Apply data augmentation to the input image
    face_img = datagen.random_transform(face_img)
    
    # Resize the augmented image to the target size
    processed_img = cv2.resize(face_img, target_size)
    # Normalize pixel values to the range [0, 1]
    processed_img = processed_img.astype('float32') / 255.0
    # Add batch dimension for compatibility with model input shape
    processed_img = np.expand_dims(processed_img, axis=0)
    return processed_img

def map_to_level(oil_prediction):
    if oil_prediction < 0.25:
        return "Low level"
    elif oil_prediction < 0.5:
        return "Normal level"
    elif oil_prediction < 0.75:
        return "Middle level"
    else:
        return "High level"

def predict_skin_type(image_path):
    # Load the input image
    input_image = cv2.imread(image_path)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(input_image, scaleFactor=1.1, minNeighbors=5)
    
    if len(faces) == 0:
        return "No face detected", 0.0, "N/A"
    
    # Assuming only one face is detected, extract and preprocess the face region
    x, y, w, h = faces[0]
    face_img = input_image[y:y+h, x:x+w]
    processed_image = preprocess_image(face_img)
    
    # Use the pre-trained model to predict oiliness level
    oil_prediction = oil_model.predict(processed_image)
    
    # Interpret the prediction result
    skin_type = 'Oily' if oil_prediction[0][0] > 0.5 else 'Non-oily'
    percentage = oil_prediction[0][0]
    oiliness_level = map_to_level(oil_prediction[0][0])
    
    return skin_type, percentage, oiliness_level

def recommend_treatment(oiliness_level):
    if oiliness_level == "Low level":
        return [
            "Use a gentle, hydrating cleanser that doesn't strip away natural oils.",
            "Moisturize regularly with a rich, creamy moisturizer to keep skin hydrated.",
            "Use products with ingredients like hyaluronic acid, glycerin, and ceramides to lock in moisture.",
            "Limit hot showers and baths, as hot water can further dry out the skin.",
            "Exfoliate gently to remove dead skin cells and promote cell turnover.",
            "Use a humidifier in dry indoor environments to add moisture to the air."
        ]
    elif oiliness_level == "Normal level":
        return [
            "Use a gentle cleanser suitable for your skin type to maintain balance.",
            "Moisturize regularly to keep skin hydrated and balanced.",
            "Use sunscreen daily to protect against UV damage and premature aging.",
            "Maintain a healthy diet and stay hydrated for overall skin health."
        ]
    elif oiliness_level == "Middle level":
        return [
            "Use a mild cleanser that doesn't overly dry out or irritate the skin.",
            "Use a lightweight, oil-free moisturizer on areas that tend to be dry.",
            "Use targeted treatments for specific skin concerns, such as acne or dry patches.",
            "Consider using a mattifying primer or oil-absorbing products on oily areas.",
            "Adjust your skincare routine based on how your skin feels in different areas."
        ]
    elif oiliness_level == "High level":
        return [
            "Use a gentle, foaming cleanser to remove excess oil and impurities.",
            "Use oil-free or mattifying moisturizers to hydrate without adding excess oil.",
            "Use products with ingredients like salicylic acid or benzoyl peroxide to control acne and breakouts.",
            "Use a clay mask or exfoliating treatment 1-2 times a week to help control oil production.",
            "Avoid heavy or greasy products that can clog pores and exacerbate oiliness."
        ]
    else:
        return ["Skin type not recognized."]

def recommend_beauty_products(oiliness_level):
    if oiliness_level == "Low level":
        return [
            "Hydrating Cleanser: CeraVe Hydrating Facial Cleanser",
            "Rich Moisturizer: Cetaphil Rich Hydrating Night Cream",
            "Hydrating Serum: Neutrogena Hydro Boost Hydrating Serum"
        ]
    elif oiliness_level == "Normal level":
        return [
            "Gentle Cleanser: Cetaphil Gentle Skin Cleanser",
            "Moisturizer: Neutrogena Hydro Boost Water Gel",
            "Sunscreen: La Roche-Posay Anthelios Melt-in Milk Sunscreen"
        ]
    elif oiliness_level == "Middle level":
        return [
            "Mild Cleanser: Cetaphil Daily Facial Cleanser",
            "Lightweight Moisturizer: La Roche-Posay Toleriane Double Repair Face Moisturizer",
            "Targeted Treatments: The Ordinary Niacinamide 10% + Zinc 1% Serum for oily areas, The Ordinary Hyaluronic Acid 2% + B5 for dry areas"
        ]
    elif oiliness_level == "High level":
        return [
            "Foaming Cleanser: CeraVe Foaming Facial Cleanser",
            "Oil-Free Moisturizer: Paula's Choice Skin Balancing Invisible Finish Moisture Gel",
            "Acne Control Treatment: The Ordinary Salicylic Acid 2% Solution"
        ]
    else:
        return ["Skin type not recognized."]   
    
# Streamlit App
st.title('Skin Type Prediction and Recommendations')

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform predictions and recommendations
    skin_type, percentage, oiliness_level = predict_skin_type(uploaded_file)
    st.write(f"Predicted Skin Type: {skin_type} | Oiliness Percentage: {percentage*100:.2f}% | Oiliness Level: {oiliness_level}")

    treatment_info = recommend_treatment(oiliness_level)
    st.write("\nTreatment Recommendations:")
    for info in treatment_info:
        st.write("- " + info)

    product_info = recommend_beauty_products(oiliness_level)
    st.write("\nBeauty Products Recommendations:")
    for info in product_info:
        st.write("- " + info)

