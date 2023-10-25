import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load a pre-trained deep learning model (e.g., VGG16)
base_model = VGG16(weights='imagenet', include_top=False)

# Define a function to extract features from an image
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    features = base_model.predict(x)
    return features.flatten()

# Directory containing images
image_folder = "images_to_compare/"

# List all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp'))]

# Load and extract features for all images in the folder
image_features = {}
for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    features = extract_features(img_path)
    image_features[img_file] = features

# Input query image
query_image_path = "query_image.jpg"  # Replace with the path to your query image
query_features = extract_features(query_image_path)

# Calculate cosine similarity between query image and all images in the folder
similarities = {}
for img_file, features in image_features.items():
    similarity = cosine_similarity([query_features], [features])
    similarities[img_file] = similarity[0][0]

# Sort images by similarity
sorted_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

# Display matching images
for img_file, similarity in sorted_images:
    print(f"Image: {img_file}, Similarity: {similarity:.2f}")
