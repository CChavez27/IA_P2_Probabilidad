#Salazar Chavez Cristian Uriel
#21310215          
# Probabilidades marginales de los eventos A, B y C
P_A = 0.6
P_B_given_A = 0.7
P_C_given_A_and_B = 0.8

# Calcular la probabilidad conjunta utilizando la Regla de la Cadena
P_A_and_B_and_C = P_A * P_B_given_A * P_C_given_A_and_B

# Imprimir el resultado
print("La probabilidad de que ocurran los eventos A, B y C en sucesión es:", P_A_and_B_and_C)
