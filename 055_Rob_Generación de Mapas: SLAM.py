#Salazar Chavez Cristian Uriel
#21310215

import numpy as np
import matplotlib.pyplot as plt

# Función para actualizar el estado y la covarianza del filtro de Kalman extendido (EKF)
def ekf_update(mu, sigma, z, R, H):
    # Paso de predicción
    mu_bar = mu
    sigma_bar = sigma
    
    # Paso de actualización
    K = sigma_bar @ H.T @ np.linalg.inv(H @ sigma_bar @ H.T + R)
    mu = mu_bar + K @ (z - H @ mu_bar)
    sigma = (np.eye(len(mu)) - K @ H) @ sigma_bar
    
    return mu, sigma

# Parámetros del entorno y del robot
landmarks = np.array([[2, 2], [8, 8], [5, 12]])  # Posiciones conocidas de los hitos
num_landmarks = len(landmarks)
robot_start = np.array([0, 0])  # Posición inicial del robot
robot_true_motion = np.array([1, 1])  # Movimiento verdadero del robot
robot_sensor_noise = 0.1  # Ruido del sensor

# Parámetros del filtro de Kalman extendido (EKF)
mu = np.zeros(3 + 2 * num_landmarks)  # Estado inicial [x, y, theta, l1_x, l1_y, l2_x, l2_y, ..., ln_x, ln_y]
sigma = np.eye(3 + 2 * num_landmarks) * 0.1  # Covarianza inicial

# Ciclo de tiempo
num_steps = 100
trajectory = []

for t in range(num_steps):
    # Movimiento verdadero del robot (en este ejemplo, se supone un movimiento lineal)
    robot_true_motion += np.array([0.1, 0.1])
    
    # Simular medición de rango (distancia a los hitos) con ruido
    true_distances = np.linalg.norm(landmarks - robot_true_motion, axis=1)
    noisy_distances = true_distances + np.random.normal(0, robot_sensor_noise, num_landmarks)
    
    # Actualizar el estado del filtro de Kalman extendido (EKF)
    mu[0:3], sigma[0:3, 0:3] = ekf_update(mu[0:3], sigma[0:3, 0:3], robot_true_motion, np.eye(3), np.eye(3))
    for i in range(num_landmarks):
        if noisy_distances[i] < 20:  # Solo actualizar si el rango es razonable
            z = np.array([noisy_distances[i]])
            landmark_index = 3 + 2 * i
            H = np.zeros((1, 3 + 2 * num_landmarks))
            H[:, 0:3] = -np.array([[(landmarks[i][0] - mu[0]) / true_distances[i], (landmarks[i][1] - mu[1]) / true_distances[i], 0]])
            H[:, landmark_index:landmark_index + 2] = np.array([[(landmarks[i][0] - mu[0]) / true_distances[i], (landmarks[i][1] - mu[1]) / true_distances[i]]])
            R = np.eye(1) * robot_sensor_noise
            mu, sigma = ekf_update(mu, sigma, z, R, H)
    
    # Guardar la posición actual del robot
    trajectory.append(mu[0:2])

# Visualizar el entorno y la trayectoria del robot
plt.figure(figsize=(10, 6))
plt.plot(trajectory[0][0], trajectory[0][1], 'go', markersize=10, label='Inicio')
plt.plot(trajectory[-1][0], trajectory[-1][1], 'ro', markersize=10, label='Fin')
plt.plot(trajectory[:, 0], trajectory[:, 1], '-b', label='Trayectoria del Robot')
plt.scatter(landmarks[:, 0], landmarks[:, 1], color='orange', marker='x', label='Hitos')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simulación SLAM con Filtro de Kalman Extendido')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
