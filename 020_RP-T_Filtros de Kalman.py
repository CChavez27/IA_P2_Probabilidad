#Salazar Chavez Cristian Uriel
#21310215

import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, A, H, Q, R, x0, P0):
        self.A = A  # Matriz de transición de estado
        self.H = H  # Matriz de observación
        self.Q = Q  # Covarianza del proceso
        self.R = R  # Covarianza de la medición
        self.x = x0  # Estado inicial
        self.P = P0  # Covarianza inicial

    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

# Parámetros del modelo
dt = 1.0  # Intervalo de tiempo
A = np.array([[1, dt], [0, 1]])  # Matriz de transición de estado
H = np.array([[1, 0]])  # Matriz de observación
Q = np.array([[0.1, 0], [0, 0.1]])  # Covarianza del proceso
R = np.array([[0.1]])  # Covarianza de la medición
x0 = np.array([[0], [0]])  # Estado inicial
P0 = np.array([[1, 0], [0, 1]])  # Covarianza inicial

# Crear un filtro de Kalman
kf = KalmanFilter(A, H, Q, R, x0, P0)

# Generar datos de ejemplo
np.random.seed(0)
num_steps = 50
true_position = 0.1 * np.arange(num_steps)
measurements = true_position + np.random.normal(0, 0.5, num_steps)

# Ejecutar el filtro de Kalman
estimated_positions = []
for z in measurements:
    kf.predict()
    kf.update(z)
    estimated_positions.append(kf.x[0, 0])

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(true_position, label='Posición Verdadera')
plt.plot(measurements, 'ro', label='Mediciones')
plt.plot(estimated_positions, label='Posición Estimada')
plt.title('Filtro de Kalman: Predicción')
plt.xlabel('Paso de Tiempo')
plt.ylabel('Posición')
plt.legend()
plt.grid(True)
plt.show()
