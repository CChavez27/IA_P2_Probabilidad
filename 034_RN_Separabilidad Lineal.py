#Salazar Chavez Cristian Uriel
#21310215   

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# Generar datos de ejemplo con dos clases y características
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, random_state=42)

# Visualizar los datos de ejemplo
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
plt.title('Datos de ejemplo')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Verificar la separabilidad lineal utilizando un clasificador SVM lineal
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X, y)

# Visualizar el hiperplano resultante
w = svm_classifier.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-2, 2)
yy = a * xx - (svm_classifier.intercept_[0]) / w[1]

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
plt.plot(xx, yy, 'k-')
plt.title('Separabilidad Lineal')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
