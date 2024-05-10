#Salazar Chavez Cristian Uriel
#21310215         

import numpy as np
import matplotlib.pyplot as plt

# Función para verificar si una configuración es válida en el entorno
def is_valid_configuration(configuration):
    x, y = configuration
    # En este ejemplo, el robot no puede ir más allá de ciertos límites en el entorno
    return 0 <= x <= 10 and 0 <= y <= 10

# Función para visualizar el espacio de configuración
def visualize_configuration_space():
    # Definir límites del espacio de configuración
    x_range = np.arange(0, 11, 0.1)
    y_range = np.arange(0, 11, 0.1)
    
    # Crear una cuadrícula de configuraciones y verificar su validez
    configuration_space = []
    for x in x_range:
        for y in y_range:
            configuration = [x, y]
            if is_valid_configuration(configuration):
                configuration_space.append(configuration)
    
    # Convertir a matriz para trazado
    configuration_space = np.array(configuration_space)
    
    # Visualizar el espacio de configuración
    plt.figure(figsize=(8, 6))
    plt.scatter(configuration_space[:, 0], configuration_space[:, 1], s=1, color='blue')
    plt.xlabel('Posición en X')
    plt.ylabel('Posición en Y')
    plt.title('Espacio de Configuración')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Visualizar el espacio de configuración
visualize_configuration_space()
