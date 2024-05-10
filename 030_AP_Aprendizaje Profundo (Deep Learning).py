Aprendizaje Profundo (Deep Learning)       

import tensorflow as tf
from tensorflow.keras import layers, models

# Definir la arquitectura de la red neuronal convolucional (CNN)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Cargar y preprocesar datos de ejemplo (por ejemplo, el conjunto de datos MNIST)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Agregar una dimensión al conjunto de datos para que sea compatible con la entrada de la CNN
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Precisión del modelo en el conjunto de prueba:", test_acc)
