#Salazar Chavez Cristian Uriel
#21310215        

import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializar los pesos de las capas ocultas y de salida
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        
        # Inicializar los sesgos de las capas ocultas y de salida
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_propagation(self, inputs):
        # Calcular la salida de las capas ocultas y de salida
        self.hidden_output = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        self.output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)
    
    def backward_propagation(self, inputs, targets, learning_rate):
        # Calcular el error y actualizar los pesos y sesgos utilizando retropropagación
        output_error = targets - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)
        
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        self.weights_hidden_output += learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.weights_input_hidden += learning_rate * np.dot(inputs.T, hidden_delta)
        
        self.bias_output += learning_rate * np.sum(output_delta)
        self.bias_hidden += learning_rate * np.sum(hidden_delta)
    
    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            # Realizar propagación hacia adelante
            self.forward_propagation(inputs)
            
            # Realizar retropropagación del error
            self.backward_propagation(inputs, targets, learning_rate)
            
            # Calcular y mostrar el error en cada época
            error = np.mean(np.abs(targets - self.output))
            print(f'Epoch {epoch + 1}/{epochs}, Error: {error}')

# Ejemplo de uso
# Datos de entrada y salida
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Crear una red neuronal con 2 neuronas de entrada, 2 ocultas y 1 de salida
network = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# Entrenar la red neuronal
network.train(inputs, targets, epochs=10000, learning_rate=0.1)
