#Salazar Chavez Cristian Uriel
#21310215        

import numpy as np

# Función de activación Sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Función de activación ReLU (Rectified Linear Unit)
def relu(x):
    return np.maximum(0, x)

# Función de activación Tangente Hiperbólica
def tanh(x):
    return np.tanh(x)

# Función de activación Softmax
def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

# Datos de ejemplo
x = np.array([[1, 2, 3],
              [-1, 0, 1]])

# Aplicar las funciones de activación a los datos de ejemplo
sigmoid_output = sigmoid(x)
relu_output = relu(x)
tanh_output = tanh(x)
softmax_output = softmax(x)

# Mostrar los resultados
print("Función de activación Sigmoide:")
print(sigmoid_output)
print("\nFunción de activación ReLU:")
print(relu_output)
print("\nFunción de activación Tangente Hiperbólica:")
print(tanh_output)
print("\nFunción de activación Softmax:")
print(softmax_output)
