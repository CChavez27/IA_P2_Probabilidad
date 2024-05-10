#Salazar Chavez Cristian Uriel
#21310215

import numpy as np
import matplotlib.pyplot as plt

# Función para generar partículas aleatorias
def generate_particles(num_particles, x_range, y_range):
    particles = []
    for _ in range(num_particles):
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        particles.append([x, y, 1.0/num_particles])  # [x, y, weight]
    return np.array(particles)

# Función para mover las partículas
def move_particles(particles, delta_x, delta_y):
    for i in range(len(particles)):
        particles[i][0] += np.random.normal(delta_x, 1)
        particles[i][1] += np.random.normal(delta_y, 1)
    return particles

# Función para calcular la probabilidad de observación dada la posición
def observation_prob(observation, particle):
    # En este ejemplo, simplemente se asume una probabilidad constante
    return 1.0

# Función para actualizar los pesos de las partículas según las observaciones
def update_weights(particles, observation):
    for i in range(len(particles)):
        prob = observation_prob(observation, particles[i])
        particles[i][2] *= prob
    total_weight = sum(particle[2] for particle in particles)
    for i in range(len(particles)):
        particles[i][2] /= total_weight
    return particles

# Función para resamplear las partículas
def resample_particles(particles):
    num_particles = len(particles)
    new_particles = []
    cumulative_weights = np.cumsum([particle[2] for particle in particles])
    for _ in range(num_particles):
        rand_val = np.random.uniform(0, 1)
        idx = np.searchsorted(cumulative_weights, rand_val)
        new_particles.append(particles[idx].copy())
    return np.array(new_particles)

# Función para estimar la posición del robot
def estimate_position(particles):
    x_est = np.mean([particle[0] for particle in particles])
    y_est = np.mean([particle[1] for particle in particles])
    return x_est, y_est

# Parámetros de la simulación
num_particles = 1000
x_range = (0, 100)
y_range = (0, 100)

# Generar partículas iniciales
particles = generate_particles(num_particles, x_range, y_range)

# Movimiento del robot (delta_x, delta_y)
delta_x = 1
delta_y = 1

# Observaciones del entorno
observation = (50, 50)  # En este ejemplo, el robot observa un objeto en (50, 50)

# Actualizar pesos de las partículas según la observación
particles = update_weights(particles, observation)

# Resamplear las partículas
particles = resample_particles(particles)

# Estimar la posición del robot
x_est, y_est = estimate_position(particles)
print("Estimación de la posición del robot:", x_est, y_est)

# Visualización de las partículas y la estimación de la posición del robot
plt.scatter(particles[:, 0], particles[:, 1], s=5, color='blue', alpha=0.5, label='Partículas')
plt.scatter(x_est, y_est, color='red', marker='x', label='Estimación')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Localización de Monte Carlo')
plt.legend()
plt.grid(True)
plt.show()
