#Salazar Chavez Cristian Uriel
#21310215

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parámetros de la distribución normal
mean = 0  # Media
std_dev = 1  # Desviación estándar

# Generar muestras aleatorias de una distribución normal
samples = np.random.normal(mean, std_dev, 1000)

# Calcular la función de densidad de probabilidad (PDF) teórica
x = np.linspace(-5, 5, 100)
pdf = norm.pdf(x, mean, std_dev)

# Trama del histograma de las muestras aleatorias
plt.hist(samples, bins=30, density=True, alpha=0.5, color='g', label='Histograma de Muestras')

# Trama de la función de densidad de probabilidad (PDF) teórica
plt.plot(x, pdf, color='r', label='PDF Teórica')

# Etiquetas y leyenda
plt.xlabel('Valor')
plt.ylabel('Densidad de Probabilidad')
plt.title('Distribución Normal')
plt.legend()

# Mostrar la gráfica
plt.show()
