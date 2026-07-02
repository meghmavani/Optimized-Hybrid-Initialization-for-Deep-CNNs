import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


class ScaledUniformInitializer(initializers.Initializer):
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, shape, dtype=None):
        """Initialize weights with a uniform distribution scaled by input size."""
        fan_in = np.prod(shape[:-1]) 
        limit = self.scale / np.sqrt(fan_in)  
        return tf.random.uniform(shape, minval=-limit, maxval=limit, dtype=dtype)

    def get_config(self):
        return {"scale": self.scale}


def create_cnn_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer=ScaledUniformInitializer(scale=2.0), input_shape=(32, 32, 3)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer=ScaledUniformInitializer(scale=2.0)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer=ScaledUniformInitializer(scale=2.0)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_initializer=ScaledUniformInitializer(scale=2.0)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

print("Training model with Custom Scaled Uniform Initialization")
model = create_cnn_model()
history = model.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test), verbose=2)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Custom Initialization Accuracy: {test_acc:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy for Custom Scaled Uniform Initialization')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
