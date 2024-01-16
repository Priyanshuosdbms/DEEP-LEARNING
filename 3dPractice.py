import tensorflow as tf
from keras import layers, models
import os
import nibabel as nib
import numpy as np


# Define the 3D CNN model
def create_3d_cnn_model(input_shape):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # Flatten layer and fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Example usage
input_shape = (64, 64, 64, 3)  # Adjust the input shape based on your 3D image size and channels
model = create_3d_cnn_model(input_shape)

# Display the model summary
model.summary()


def load_3d_image(file_path):
    img = nib.load(file_path)
    img_data = img.get_fdata()
    return img_data


def preprocess_3d_image(img_data):
    # Implement your preprocessing steps here
    # This might include normalization, resizing, etc.
    return img_data


def your_3d_image_generator(data_dir, batch_size):
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.nii')]
    while True:
        batch_files = np.random.choice(image_files, batch_size)
        batch_images = [load_3d_image(os.path.join(data_dir, f)) for f in batch_files]
        batch_images = [preprocess_3d_image(img) for img in batch_images]

        # Your labels should be prepared accordingly
        # For example, if your images are categorized into classes 'very strong', 'strong', etc.
        # You should have a corresponding array of labels for each image in the batch.

        batch_labels = np.random.randint(0, 5, size=batch_size)  # Placeholder for demonstration

        yield np.array(batch_images), np.array(batch_labels)


# Assuming you have a generator for 3D images and labels
train_data_dir = ''
test_data_dir = ''
train_generator = your_3d_image_generator(train_data_dir, batch_size=32)
test_generator = your_3d_image_generator(test_data_dir, batch_size=32)

# Train the model
model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

# Save the model
model.save('3d_cnn_model.h5')
