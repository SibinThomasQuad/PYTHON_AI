import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights='imagenet')

# Load and preprocess an image
image_path = 'img.jpg'  # Replace with the path to your image
img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

# Predict the objects in the image
predictions = model.predict(img)
decoded_predictions = decode_predictions(predictions, top=5)[0]

# Display the top 5 predicted objects
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")

# The label with the highest score is the most likely prediction
best_prediction = decoded_predictions[0]
print(f"The image is most likely a {best_prediction[1]} with a confidence of {best_prediction[2]:.2f}.")
