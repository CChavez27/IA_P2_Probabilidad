#Salazar Chavez Cristian Uriel
#21310215

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Definir la estructura de la red bayesiana
model = BayesianNetwork([('D', 'S1'), ('D', 'S2')])

# Definir las tablas de probabilidad condicional (CPDs)
cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.6], [0.4]])
cpd_s1 = TabularCPD(variable='S1', variable_card=2, 
                    values=[[0.8, 0.2],
                            [0.2, 0.8]],
                    evidence=['D'], evidence_card=[2])
cpd_s2 = TabularCPD(variable='S2', variable_card=2, 
                    values=[[0.7, 0.3],
                            [0.3, 0.7]],
                    evidence=['D'], evidence_card=[2])

# Añadir las CPDs al modelo
model.add_cpds(cpd_d, cpd_s1, cpd_s2)

# Verificar si el modelo es válido
assert model.check_model()

# Realizar inferencia probabilística
inference = VariableElimination(model)
posterior = inference.query(variables=['D'], evidence={'S1': 1, 'S2': 0})
print(posterior['D'])
