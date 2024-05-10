#Salazar Chavez Cristian Uriel
#21310215       

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

# Generar datos de ejemplo (círculos)
X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)

# Crear un clasificador SVM con un núcleo RBF (radial basis function)
svm_classifier = SVC(kernel='rbf', C=10, gamma='auto')

# Ajustar el clasificador a los datos de ejemplo
svm_classifier.fit(X, y)

# Función para visualizar la frontera de decisión
def plot_decision_boundary(clf, X, y):
    plt.figure(figsize=(8, 6))
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Support Vector Machine with RBF Kernel')
    plt.show()

# Visualizar la frontera de decisión
plot_decision_boundary(svm_classifier, X, y)
