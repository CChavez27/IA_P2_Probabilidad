#Salazar Chavez Cristian Uriel
#21310215

import numpy as np

class MarkovChain:
    def __init__(self, transition_matrix, states):
        """
        Inicializa la cadena de Markov con la matriz de transición y los estados.
        
        Args:
        - transition_matrix (numpy.ndarray): Matriz de transición de la cadena de Markov.
        - states (list): Lista de estados de la cadena de Markov.
        """
        self.transition_matrix = transition_matrix
        self.states = states
        self.current_state = np.random.choice(states)

    def next_state(self):
        """
        Calcula el siguiente estado de la cadena de Markov.
        
        Returns:
        - int: Siguiente estado.
        """
        next_state_index = np.random.choice(range(len(self.states)), p=self.transition_matrix[self.current_state])
        self.current_state = self.states[next_state_index]
        return self.current_state

# Definir la matriz de transición y los estados
transition_matrix = np.array([[0.7, 0.3],
                              [0.2, 0.8]])
states = [0, 1]  # Por ejemplo, 0 = soleado, 1 = lluvioso

# Crear la cadena de Markov
markov_chain = MarkovChain(transition_matrix, states)

# Realizar algunas iteraciones y mostrar los estados
print("Estados generados por la cadena de Markov:")
for _ in range(10):
    print(markov_chain.next_state())
