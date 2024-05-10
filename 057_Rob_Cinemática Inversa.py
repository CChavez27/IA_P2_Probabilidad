#Salazar Chavez Cristian Uriel
#21310215         

import numpy as np

# Función para la cinemática inversa de un brazo robótico de dos grados de libertad
def inverse_kinematics(x, y, l1, l2):
    # Distancia del efector final al origen
    d = np.sqrt(x**2 + y**2)

    # Ángulo entre el primer eslabón y la línea que conecta el origen y el efector final
    alpha = np.arccos((l1**2 + d**2 - l2**2) / (2 * l1 * d))

    # Ángulo entre el primer eslabón y la línea x
    beta = np.arctan2(y, x) - np.arctan2(l2 * np.sin(np.pi - alpha), l1 + l2 * np.cos(np.pi - alpha))

    return beta, np.pi - alpha

# Posición deseada del efector final
x_desired = 5
y_desired = 5

# Longitudes de los eslabones
l1 = 3
l2 = 3

# Calcular las posiciones de las articulaciones para alcanzar la posición deseada
theta1, theta2 = inverse_kinematics(x_desired, y_desired, l1, l2)

# Mostrar los resultados
print("Ángulo de la articulación 1:", np.degrees(theta1))
print("Ángulo de la articulación 2:", np.degrees(theta2))
