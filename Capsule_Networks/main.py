import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define the Capsule Layer
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, capsule_dim, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.routings = routings
        self.kernel_initializer = tf.keras.initializers.get("he_normal")
        self.bias_initializer = tf.keras.initializers.get("zeros")

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        self.W = self.add_weight(
            name="W",
            shape=(1, input_dim_capsule, self.num_capsules * self.capsule_dim),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.built = True

    def call(self, inputs, **kwargs):
        inputs_expand = tf.expand_dims(inputs, axis=-1)
        inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_capsules * self.capsule_dim, 1])
        inputs_hat = tf.einsum("ijk,kmn->ijmn", inputs_tiled, self.W)
        b = tf.zeros_like(inputs_hat[:, :, :, 0])
        for _ in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            s = tf.reduce_sum(tf.multiply(c[:, :, :, tf.newaxis], inputs_hat), axis=1)
            v = self.squash(s)
            b += tf.reduce_sum(tf.multiply(v, inputs_hat), axis=-1)
        return v

    def squash(self, s):
        s_norm = tf.norm(s, axis=-1, keepdims=True)
        s_squash = s_norm ** 2 / (1 + s_norm ** 2) * s / (s_norm + tf.keras.backend.epsilon())
        return s_squash

# Build the Capsule Network model
def build_capsule_model(input_shape):
    input_layer = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(256, (9, 9), activation="relu")(input_layer)
    primary_capsule = CapsuleLayer(num_capsules=8, capsule_dim=32)(conv1)
    digit_capsule = CapsuleLayer(num_capsules=10, capsule_dim=16, routings=3)(primary_capsule)
    flat_capsule = layers.Flatten()(digit_capsule)
    output_layer = layers.Dense(10, activation="softmax")(flat_capsule)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model

# Compile and train the model
input_shape = x_train.shape[1:]
capsule_model = build_capsule_model(input_shape)
capsule_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
capsule_model.summary()

capsule_model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate the model
accuracy = capsule_model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test)[1]
print(f"Test Accuracy: {accuracy * 100:.2f}%")
