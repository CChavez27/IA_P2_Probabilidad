#Salazar Chavez Cristian Uriel
#21310215      

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

# Generar datos de ejemplo
X, _ = make_blobs(n_samples=1000, centers=3, cluster_std=1.5, random_state=42)

# Inicializar el modelo de mezcla gaussiana
model = GaussianMixture(n_components=3, random_state=42)

# Ajustar el modelo a los datos de ejemplo
model.fit(X)

# Mostrar los parámetros estimados del modelo
print("Parámetros estimados del modelo:")
print("Peso de cada componente:", model.weights_)
print("Media de cada componente:", model.means_)
print("Covarianza de cada componente:", model.covariances_)
