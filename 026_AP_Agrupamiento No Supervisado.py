#Salazar Chavez Cristian Uriel
#21310215         

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generar datos de ejemplo
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Inicializar y ajustar el modelo K-Means
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Obtener las etiquetas de los clusters y los centroides
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Graficar los clusters y los centroides
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], marker='o', c='red', s=200, edgecolors='k')
plt.title('Agrupamiento con K-Means')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
