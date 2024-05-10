#Salazar Chavez Cristian Uriel
#21310215

import numpy as np

# Función para verificar la independencia condicional
def check_conditional_independence(data, var1, var2, condition):
    # Filtrar los datos que cumplen con la condición
    filtered_data = data[data[:, 2] == condition]
    
    # Calcular las probabilidades condicionales
    p_var1_given_condition = np.mean(filtered_data[:, 0] == var1)
    p_var2_given_condition = np.mean(filtered_data[:, 1] == var2)
    
    # Calcular la probabilidad conjunta
    p_joint = np.mean((filtered_data[:, 0] == var1) & (filtered_data[:, 1] == var2))
    
    # Verificar la independencia condicional
    if np.isclose(p_joint, p_var1_given_condition * p_var2_given_condition):
        return True
    else:
        return False

# Generar datos aleatorios: dos variables (var1 y var2) y una variable de condición
np.random.seed(42)
data_size = 1000
var1_values = np.random.choice([0, 1], size=data_size)
var2_values = np.random.choice([0, 1], size=data_size)
condition_values = np.random.choice([0, 1], size=data_size)

# Combinar los valores en un solo conjunto de datos
data = np.column_stack((var1_values, var2_values, condition_values))

# Verificar la independencia condicional dada la condición
var1 = 0
var2 = 0
condition = 0
is_independent = check_conditional_independence(data, var1, var2, condition)

# Imprimir el resultado
print(f"¿Las variables {var1} y {var2} son independientes condicionales a la condición {condition}? {is_independent}")
