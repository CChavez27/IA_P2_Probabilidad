#Salazar Chavez Cristian Uriel
#21310215

import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.class_priors = None
        self.means = None
        self.stds = None

    def fit(self, X, y):
        # Calcular la probabilidad a priori de cada clase
        self.class_priors = {}
        for label in np.unique(y):
            self.class_priors[label] = np.mean(y == label)

        # Calcular las medias y desviaciones estándar de cada característica para cada clase
        self.means = {}
        self.stds = {}
        for label in np.unique(y):
            self.means[label] = np.mean(X[y == label], axis=0)
            self.stds[label] = np.std(X[y == label], axis=0)

    def _calculate_likelihood(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def _calculate_posterior(self, X):
        posteriors = []
        for label in self.class_priors:
            prior = self.class_priors[label]
            likelihood = np.prod(self._calculate_likelihood(X, self.means[label], self.stds[label]), axis=1)
            posterior = prior * likelihood
            posteriors.append(posterior)
        return np.array(posteriors).T

    def predict(self, X):
        posteriors = self._calculate_posterior(X)
        return np.argmax(posteriors, axis=1)

# Datos de ejemplo (longitud y ancho del pétalo de flores)
X = np.array([[1.5, 0.3],
              [4.5, 1.3],
              [5.7, 2.4],
              [1.3, 0.2],
              [5.2, 2.3]])
y = np.array([0, 1, 2, 0, 2])  # Etiquetas de clase (0, 1, 2)

# Crear y ajustar el clasificador Naive Bayes
classifier = NaiveBayesClassifier()
classifier.fit(X, y)

# Datos de prueba
X_test = np.array([[1.4, 0.2],
                   [4.9, 2.0]])

# Predicciones
predictions = classifier.predict(X_test)
print("Predicciones:", predictions)
