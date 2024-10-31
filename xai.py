import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import shap

# Load your trained model
model = load_model('disease_classification_model.keras')

# Define the class names
class_names = ['BA- cellulitis', 'BA-impetigo', 'FU-athlete-foot', 
               'FU-nail-fungus', 'FU-ringworm', 
               'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles']

# Define the prediction function for SHAP
def f(x):
    return model.predict(x)

# Load and preprocess the input image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Adjust size as needed
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Path to your input image
img_path = 'download.jpeg'  # Update with your image path
X_single = load_and_preprocess_image(img_path)

# Create a masker for SHAP
masker = shap.maskers.Image("inpaint_telea", X_single.shape[1:])  # Use shape without batch dimension

# Create an explainer
explainer = shap.Explainer(f, masker, output_names=class_names)

# Explain the single image
shap_values = explainer(X_single, max_evals=100)

# Check the shape of the SHAP values
print("SHAP values shape:", shap_values.shape)

# Since we expect shap_values to have one output for each class,
# make sure to handle it correctly for a single image
if len(shap_values) == 1:  # If it's a single image
    shap.image_plot(shap_values[0])  # Plot only the first (and only) explanation
else:
    shap.image_plot(shap_values)  # For multiple images (not the case here)

# Print the model's prediction
predicted_class = np.argmax(model.predict(X_single), axis=1)[0]
print(f"Predicted class: {class_names[predicted_class]}")
