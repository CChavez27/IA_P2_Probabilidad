#Salazar Chavez Cristian Uriel
#21310215       

import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for pattern in patterns:
            pattern = pattern.reshape(-1, 1)
            self.weights += np.dot(pattern, pattern.T)
            np.fill_diagonal(self.weights, 0)

    def predict(self, pattern, max_iterations=100):
        pattern = pattern.reshape(-1, 1)
        for _ in range(max_iterations):
            new_pattern = np.sign(np.dot(self.weights, pattern))
            if np.array_equal(new_pattern, pattern):
                return new_pattern
            pattern = new_pattern
        return pattern

# Ejemplo de uso
patterns = np.array([[1, 1, -1, -1],
                     [-1, -1, 1, 1],
                     [1, -1, 1, -1]])

hopfield = HopfieldNetwork(num_neurons=len(patterns[0]))
hopfield.train(patterns)

# Introducimos un patrón ruidoso para recuperar el patrón original
noisy_pattern = np.array([-1, -1, 1, -1])

retrieved_pattern = hopfield.predict(noisy_pattern)
print("Patrón original recuperado:", retrieved_pattern.flatten())
