#Salazar Chavez Cristian Uriel
#21310215

import numpy as np

# Función de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Definición de una red neuronal simple
class NeuralNetwork:
    def __init__(self):
        # Parámetros de la red neuronal (pesos y sesgos)
        self.weights = np.random.randn(2, 1)  # Pesos
        self.bias = np.random.randn(1)        # Sesgo

    # Método para predecir
    def predict(self, x):
        return sigmoid(np.dot(x, self.weights) + self.bias)

# Función principal
def main():
    # Datos de entrada
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # Crear una instancia de la red neuronal
    neural_network = NeuralNetwork()

    # Realizar predicciones para los datos de entrada
    for i, input_point in enumerate(input_data):
        # Predicción
        prediction = neural_network.predict(input_point)
        
        # Agregar incertidumbre: añadiendo ruido gaussiano a la predicción
        noisy_prediction = prediction + np.random.normal(scale=0.1)
        
        print(f"Entrada: {input_point}, Predicción: {prediction}, Predicción con Incertidumbre: {noisy_prediction}")

if __name__ == "__main__":
    main()
