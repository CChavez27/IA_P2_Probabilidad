#Salazar Chavez Cristian Uriel
#21310215        

import numpy as np
import matplotlib.pyplot as plt

# Función de densidad de probabilidad objetivo (distribución normal estándar)
def target_distribution(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

# Función de densidad de probabilidad de propuesta (distribución normal)
def proposal_distribution(x, sigma):
    return np.exp(-0.5 * (x / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)

# Muestreador de Metropolis-Hastings
def metropolis_hastings(target_distribution, proposal_distribution, num_samples, sigma):
    samples = []
    current_sample = np.random.normal(0, 1)  # Inicializar con una muestra aleatoria
    for _ in range(num_samples):
        proposed_sample = np.random.normal(current_sample, sigma)
        acceptance_ratio = (target_distribution(proposed_sample) * proposal_distribution(current_sample, sigma)) / \
                           (target_distribution(current_sample) * proposal_distribution(proposed_sample, sigma))
        if np.random.uniform(0, 1) < acceptance_ratio:
            current_sample = proposed_sample
        samples.append(current_sample)
    return samples

# Generar muestras utilizando el muestreador de Metropolis-Hastings
num_samples = 10000
sigma = 0.5  # Parámetro de la distribución de propuesta
samples = metropolis_hastings(target_distribution, proposal_distribution, num_samples, sigma)

# Graficar el histograma de las muestras generadas
plt.hist(samples, bins=50, density=True, alpha=0.5, color='g', label='Muestras')
# Graficar la función de densidad de probabilidad objetivo
x = np.linspace(-5, 5, 100)
plt.plot(x, target_distribution(x), color='r', label='Distribución Objetivo')
plt.xlabel('Valor')
plt.ylabel('Densidad de Probabilidad')
plt.title('Muestreador de Metropolis-Hastings')
plt.legend()
plt.show()
