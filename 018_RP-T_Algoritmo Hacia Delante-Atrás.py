#Salazar Chavez Cristian Uriel
#21310215       

import numpy as np

# Definir el modelo oculto de Markov (HMM)
# Matriz de transición de estado
transition_matrix = np.array([[0.7, 0.3],
                              [0.4, 0.6]])

# Matriz de emisión
emission_matrix = np.array([[0.9, 0.1],
                             [0.2, 0.8]])

# Probabilidades iniciales de estado
initial_state_probabilities = np.array([0.5, 0.5])

# Observaciones
observations = [0, 1, 0, 1]  # Por ejemplo, 0 = cabeza, 1 = cola

# Implementar el algoritmo hacia adelante
def forward_algorithm(obs, initial_probs, trans_probs, emit_probs):
    num_states = len(initial_probs)
    num_obs = len(obs)
    
    # Inicializar matriz hacia adelante
    forward = np.zeros((num_states, num_obs))
    
    # Paso hacia adelante
    forward[:, 0] = initial_probs * emit_probs[:, obs[0]]
    for t in range(1, num_obs):
        for j in range(num_states):
            forward[j, t] = np.sum(forward[:, t-1] * trans_probs[:, j]) * emit_probs[j, obs[t]]
    
    return forward

# Implementar el algoritmo hacia atrás
def backward_algorithm(obs, trans_probs, emit_probs):
    num_states = trans_probs.shape[0]
    num_obs = len(obs)
    
    # Inicializar matriz hacia atrás
    backward = np.zeros((num_states, num_obs))
    backward[:, -1] = 1
    
    # Paso hacia atrás
    for t in range(num_obs - 2, -1, -1):
        for i in range(num_states):
            backward[i, t] = np.sum(trans_probs[i, :] * emit_probs[:, obs[t+1]] * backward[:, t+1])
    
    return backward

# Calcular la probabilidad de observar la secuencia de observaciones
forward_probs = forward_algorithm(observations, initial_state_probabilities, transition_matrix, emission_matrix)
backward_probs = backward_algorithm(observations, transition_matrix, emission_matrix)

total_probability = np.sum(forward_probs[:, -1] * initial_state_probabilities * emission_matrix[:, observations[0]] * backward_probs[:, 0])
print("La probabilidad total de la secuencia de observaciones es:", total_probability)
