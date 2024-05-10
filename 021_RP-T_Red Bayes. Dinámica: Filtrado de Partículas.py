#Salazar Chavez Cristian Uriel
#21310215   

import numpy as np
import matplotlib.pyplot as plt

# Definir la dinámica del sistema
def transition_model(x_prev, u):
    return x_prev + u + np.random.normal(0, 0.1, size=x_prev.shape)

# Definir el modelo de observación
def observation_model(x):
    return x + np.random.normal(0, 0.1, size=x.shape)

# Generar datos de ejemplo
np.random.seed(0)
true_states = []
observations = []
num_steps = 50
initial_state = np.array([0])
for _ in range(num_steps):
    true_state = transition_model(initial_state, np.array([0.1]))
    observation = observation_model(true_state)
    true_states.append(true_state)
    observations.append(observation)
    initial_state = true_state

# Inicializar partículas
num_particles = 100
particles = np.random.normal(0, 1, size=(num_particles, 1))

# Implementar el filtro de partículas
for t in range(num_steps):
    # Predicción
    particles = transition_model(particles, np.random.normal(0, 0.1, size=(num_particles, 1)))
    
    # Actualización de pesos
    weights = np.exp(-0.5 * np.sum((observations[t] - particles)**2, axis=1))
    weights /= np.sum(weights)
    
    # Remuestreo
    indices = np.random.choice(range(num_particles), size=num_particles, p=weights)
    particles = particles[indices]

# Estimación del estado
estimated_state = np.mean(particles)

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(range(num_steps), [x[0] for x in true_states], label='Estado Verdadero', color='blue')
plt.scatter(range(num_steps), observations, label='Observaciones', color='red', marker='x')
plt.axhline(y=estimated_state, linestyle='--', color='green', label='Estado Estimado')
plt.xlabel('Paso de Tiempo')
plt.ylabel('Valor')
plt.title('Filtrado de Partículas')
plt.legend()
plt.grid(True)
plt.show()
