from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from PIL import Image
import cv2
import numpy as np

# Load the trained model
loaded_model = load_model('final_model.h5')

# Load and preprocess a test image
img_path = 'path/to/test_image.png'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make predictions
predictions = loaded_model.predict(img_array)

# Get the predicted class label
labels = ['very strong', 'strong', 'normal', 'weak', 'very weak']
predicted_label = labels[np.argmax(predictions)]

print(f'The image belongs to the class: {predicted_label}')
