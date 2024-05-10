#Salazar Chavez Cristian Uriel
#21310215     

import numpy as np
from scipy.stats import norm

# Generar datos de ejemplo
np.random.seed(0)
data = np.random.normal(loc=2, scale=1, size=100)

# Definir la función de verosimilitud para una distribución normal
def likelihood(data, mu, sigma):
    return np.prod(norm.pdf(data, loc=mu, scale=sigma))

# Definir la distribución previa para los parámetros (media y desviación estándar)
def prior(mu, sigma):
    # Se asume una distribución uniforme para la media y la desviación estándar
    return 1

# Definir la función de verosimilitud ponderada
def weighted_likelihood(data, mu_values, sigma_values):
    likelihoods = np.zeros((len(mu_values), len(sigma_values)))
    for i, mu in enumerate(mu_values):
        for j, sigma in enumerate(sigma_values):
            likelihoods[i, j] = likelihood(data, mu, sigma) * prior(mu, sigma)
    return likelihoods / np.sum(likelihoods)

# Definir los valores posibles de los parámetros
mu_values = np.linspace(0, 4, 100)
sigma_values = np.linspace(0.1, 2, 100)

# Calcular la verosimilitud ponderada
weighted_likelihoods = weighted_likelihood(data, mu_values, sigma_values)

# Encontrar los valores de los parámetros que maximizan la verosimilitud ponderada
max_likelihood_index = np.unravel_index(np.argmax(weighted_likelihoods), weighted_likelihoods.shape)
estimated_mu = mu_values[max_likelihood_index[0]]
estimated_sigma = sigma_values[max_likelihood_index[1]]

# Imprimir los resultados
print("Parámetros estimados:")
print("Media estimada:", estimated_mu)
print("Desviación estándar estimada:", estimated_sigma)
