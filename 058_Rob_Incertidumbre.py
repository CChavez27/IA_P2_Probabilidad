#Salazar Chavez Cristian Uriel
#21310215         

import numpy as np
import matplotlib.pyplot as plt

# Parámetros del filtro de Kalman
dt = 0.1  # Paso de tiempo
A = np.array([[1, dt], [0, 1]])  # Matriz de transición de estado
B = np.array([[0.5*dt**2], [dt]])  # Matriz de control
H = np.array([[1, 0]])  # Matriz de observación
Q = np.array([[0.1, 0], [0, 0.1]])  # Covarianza del proceso (ruido del sistema)
R = np.array([[1]])  # Covarianza de la medición (ruido del sensor)

# Estado inicial y covarianza inicial
x = np.array([[0], [0]])  # Estado inicial [posición, velocidad]
P = np.eye(2) * 10  # Covarianza inicial

# Simulación de la trayectoria verdadera y observaciones
true_position = []
measurements = []
for i in range(100):
    true_position.append(x[0, 0])
    z = H @ x + np.random.normal(0, np.sqrt(R[0, 0]))
    measurements.append(z[0, 0])
    u = np.array([[0]])  # No hay control (aceleración cero)
    x = A @ x + B @ u + np.random.multivariate_normal([0, 0], Q).reshape((2, 1))

# Filtro de Kalman
filtered_position = []
for z in measurements:
    # Predicción del estado y covarianza
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q
    
    # Actualización utilizando la observación
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x = x_pred + K @ y
    P = (np.eye(2) - K @ H) @ P_pred
    
    filtered_position.append(x[0, 0])

# Visualización de la trayectoria verdadera y la estimada por el filtro de Kalman
plt.plot(true_position, label='Trayectoria Verdadera')
plt.plot(measurements, 'ro', label='Observaciones')
plt.plot(filtered_position, label='Estimación del Filtro de Kalman')
plt.xlabel('Tiempo')
plt.ylabel('Posición')
plt.title('Filtro de Kalman para Estimación de Posición')
plt.legend()
plt.grid(True)
plt.show()
