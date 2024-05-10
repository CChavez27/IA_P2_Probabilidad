#Salazar Chavez Cristian Uriel
#21310215    

# Importar la biblioteca pgmpy para trabajar con redes bayesianas
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Definir la estructura de la red bayesiana
model = BayesianNetwork([('A', 'C'), ('B', 'C')])

# Definir las tablas de probabilidad condicional (CPDs)
cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.3], [0.7]])
cpd_b = TabularCPD(variable='B', variable_card=2, values=[[0.6], [0.4]])
cpd_c = TabularCPD(variable='C', variable_card=2, 
                   values=[[0.8, 0.4, 0.7, 0.1],
                           [0.2, 0.6, 0.3, 0.9]],
                   evidence=['A', 'B'], evidence_card=[2, 2])

# Añadir las CPDs al modelo
model.add_cpds(cpd_a, cpd_b, cpd_c)

# Verificar si el modelo es válido
assert model.check_model()

# Realizar inferencia por enumeración para calcular la probabilidad de C dado A=0 y B=1
inference = VariableElimination(model)
posterior = inference.query(variables=['C'], evidence={'A': 0, 'B': 1})
print("La probabilidad de C dado A=0 y B=1 es:", posterior.values[1])
