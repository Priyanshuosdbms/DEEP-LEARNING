"""
import os
import numpy as np
from keras import layers, models
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.utils import to_categorical

# Path to your images
images_path = "C:\\Users\\RK Niranjan\\Downloads\\images"

# Function to load and preprocess images from a directory
def load_images_from_folder(folder, target_size=(28, 28)):
    loaded_images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert('L')  # Convert to grayscale
        img = img.resize(target_size)  # Resize to a common shape
        if img is not None:
            loaded_images.append(np.array(img))
    return np.array(loaded_images)

# Load and preprocess images
images = load_images_from_folder(images_path)

# Check if images are loaded
print(f"Number of images loaded: {len(images)}")

# Assuming 'labels' are one-hot encoded, convert to integers
labels = np.array([0] * len(images))  # Replace with your label generation logic
labels = to_categorical(labels)  # Convert to one-hot encoding

# Adjust the output units based on the number of classes in your dataset
output_units = len(np.unique(np.argmax(labels, axis=1)))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)

# Define the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(output_units, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Make predictions
predictions = model.predict(X_test)

# You can add more code to analyze the results and visualize the predictions
"""
'''
import imghdr
import os

# Remove dodgy images
# !pip3 install opencv-python
import cv2
import tensorflow as tf

data_dir = 'C:/Users/RK Niranjan/Desktop/heart-disease/ImageClassification-main/data'

image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)

# Load Data
# Load Data
import numpy as np
from matplotlib import pyplot as plt

data = tf.keras.utils.image_dataset_from_directory(data_dir)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# ... (rest of your code)

# Scale Data
data = data.map(lambda x, y: (x / 255, y))

# Split Data
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Build Deep Learning Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

# Train
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=4, validation_data=val, callbacks=[tensorboard_callback])

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

from keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(pre.result(), re.result(), acc.result())


# Load the image using OpenCV
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# Load the image using OpenCV
img = cv2.imread('C:/Users/RK Niranjan/Desktop/heart-disease/ImageClassification-main/data/test')
print(f"Attempting to load image from: {img}")
# Check if the image is loaded successfully
if img is None:
    print("Error: Unable to load the image.")
else:
    # Resize the image using OpenCV
    resize = cv2.resize(img, (256, 256))

    # Display the resized image
    plt.imshow(resize)
    plt.show()

    # Preprocess the image for prediction
    resize = resize / 255.0  # Normalize to [0, 1]
    resize = np.expand_dims(resize, axis=0)  # Add batch dimension

    # Make predictions
    yhat = model.predict(resize)

    # Print the prediction result
    if yhat > 0.5:
        print(f'Predicted class is Sad')
    else:
        print(f'Predicted class is Happy')
'''

import gzip
import numpy as np
from keras.utils import to_categorical
from keras import layers, models


def load_data(filepath, num_samples):
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    data = data.reshape((num_samples, 28, 28, 1)).astype(np.float32) / 255.0
    return data


def load_labels(filepath, num_samples):
    with gzip.open(filepath, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return to_categorical(labels, num_classes=10)


# File paths
train_images_path = 'train-images-idx3-ubyte.gz'
train_labels_path = 'train-labels-idx1-ubyte.gz'
test_images_path = 't10k-images-idx3-ubyte.gz'
test_labels_path = 't10k-labels-idx1-ubyte.gz'

# Number of samples
num_train_samples = 60000
num_test_samples = 10000

# Load training data and labels
train_images = load_data(train_images_path, num_train_samples)
train_labels = load_labels(train_labels_path, num_train_samples)

# Load test data and labels
test_images = load_data(test_images_path, num_test_samples)
test_labels = load_labels(test_labels_path, num_test_samples)

print("Loading Complete !!")

# Split the training data into training and validation sets
split_ratio = 0.8  # 80% for training, 20% for validation
split_index = int(num_train_samples * split_ratio)

# Training set
train_images_subset = train_images[:split_index]
train_labels_subset = train_labels[:split_index]

# Validation set
val_images = train_images[split_index:]
val_labels = train_labels[split_index:]

# Build the model
'''
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))
'''


# Function to build a robust CNN model
def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Adding dropout for regularization
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model = build_model()
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images_subset, train_labels_subset, epochs=4, validation_data=(val_images, val_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Make predictions (if needed)
# predictions = model.predict(new_data)
from PIL import Image
import numpy as np

# Load and preprocess the test image
test_image_path = 'test.jpg'
img = Image.open(test_image_path).convert('L')  # Convert to grayscale
img = img.resize((28, 28))
img_array = np.array(img)
img_array = img_array.reshape((1, 28, 28, 1)).astype('float32') / 255.0  # Normalize pixel values to [0, 1]

# Make predictions
predictions = model.predict(img_array)

# Get the predicted class label
predicted_class = np.argmax(predictions[0])

# Print the predicted class
print(predicted_class[5])
print(predicted_class[6])
print(f'Predicted Class: {predicted_class}')
