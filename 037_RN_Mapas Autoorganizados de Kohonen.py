#Salazar Chavez Cristian Uriel
#21310215  

from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

# Crear un conjunto de datos de ejemplo
data = np.random.rand(100, 2)  # 100 muestras con 2 características cada una

# Definir las dimensiones del mapa autoorganizado (SOM)
som_width = 10
som_height = 10

# Inicializar y entrenar el SOM
som = MiniSom(som_width, som_height, 2, sigma=1.0, learning_rate=0.5)  # 2 características en los datos de entrada
som.random_weights_init(data)
som.train_random(data, 100)  # 100 iteraciones de entrenamiento

# Visualizar el SOM
plt.figure(figsize=(8, 8))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # distance map as background
plt.colorbar()

# Visualizar los datos de entrada y las ubicaciones de los nodos ganadores
for i, x in enumerate(data):
    winner = som.winner(x)
    plt.plot(winner[0] + 0.5, winner[1] + 0.5, 'o', markerfacecolor='None', markeredgecolor='r', markersize=10, markeredgewidth=2)
plt.axis([0, som_width, 0, som_height])
plt.title('Mapa Autoorganizado de Kohonen')
plt.show()
