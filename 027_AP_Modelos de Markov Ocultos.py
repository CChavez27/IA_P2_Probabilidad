#Salazar Chavez Cristian Uriel
#21310215

from hmmlearn import hmm
import numpy as np

# Definir el modelo HMM
model = hmm.GaussianHMM(n_components=2, covariance_type="full")

# Parámetros del modelo
model.startprob_ = np.array([0.5, 0.5])  # Probabilidades iniciales de estado
model.transmat_ = np.array([[0.7, 0.3],  # Matriz de transición de estado
                            [0.4, 0.6]])
model.means_ = np.array([[0.0, 0.0],    # Medias de las distribuciones Gaussianas
                         [1.0, 1.0]])
model.covars_ = np.tile(np.identity(2), (2, 1, 1))  # Covarianza de las distribuciones Gaussianas

# Generar secuencia de observaciones
X, Z = model.sample(100)

# Ajustar el modelo HMM a los datos de ejemplo
model.fit(X)

# Predicción del estado más probable
new_X = np.array([[0.1, 0.2], [0.8, 0.9]])
predicted_states = model.predict(new_X)

print("Estado más probable para nuevas observaciones:", predicted_states)
