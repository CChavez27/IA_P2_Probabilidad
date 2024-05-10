#Salazar Chavez Cristian Uriel
#21310215           

import numpy as np
import matplotlib.pyplot as plt

# Parámetros del proceso estacionario
num_samples = 1000  # Número de muestras
mean = 0  # Media
std = 1  # Desviación estándar

# Generar muestras del proceso estacionario (ruido blanco gaussiano)
samples = np.random.normal(mean, std, size=num_samples)

# Graficar las muestras
plt.figure(figsize=(10, 6))
plt.plot(samples)
plt.title('Proceso Estacionario en Sentido Amplio (Ruido Blanco Gaussiano)')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.grid(True)
plt.show()
