#Salazar Chavez Cristian Uriel
#21310215      

import tensorflow as tf
from tensorflow.keras import layers, models

# Definir la arquitectura de la red neuronal artificial (ANN)
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Capa de entrada: aplanar la imagen 28x28
    layers.Dense(128, activation='relu'),  # Capa oculta: 128 neuronas con función de activación ReLU
    layers.Dense(10, activation='softmax') # Capa de salida: 10 neuronas para clasificar las 10 clases posibles
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Cargar y preprocesar datos de ejemplo (por ejemplo, el conjunto de datos MNIST)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Precisión del modelo en el conjunto de prueba:", test_acc)
