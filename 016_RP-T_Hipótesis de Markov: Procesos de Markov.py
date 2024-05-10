#Salazar Chavez Cristian Uriel
#21310215           

import numpy as np
import matplotlib.pyplot as plt

# Definir la matriz de transición para el proceso de Markov
transition_matrix = np.array([[0.8, 0.2],  # Probabilidad de transición de A a A y de A a B
                              [0.3, 0.7]]) # Probabilidad de transición de B a A y de B a B

# Definir los estados del proceso de Markov
states = ['A', 'B']

# Generar una secuencia de estados del proceso de Markov
num_steps = 1000
current_state = np.random.choice(states)  # Estado inicial aleatorio
sequence = [current_state]
for _ in range(num_steps):
    current_state = np.random.choice(states, p=transition_matrix[states.index(current_state)])
    sequence.append(current_state)

# Contar la frecuencia de cada estado en la secuencia
state_counts = {state: sequence.count(state) for state in states}

# Graficar la secuencia de estados
plt.figure(figsize=(10, 6))
plt.plot(sequence, marker='o', linestyle='-')
plt.title('Proceso de Markov: Secuencia de Estados')
plt.xlabel('Paso de Tiempo')
plt.ylabel('Estado')
plt.yticks(range(len(states)), states)
plt.grid(True)
plt.show()

# Imprimir la frecuencia de cada estado en la secuencia
print("Frecuencia de cada estado en la secuencia:")
for state, count in state_counts.items():
    print(f"Estado {state}: {count} veces")
