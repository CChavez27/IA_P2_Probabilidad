#Salazar Chavez Cristian Uriel
#21310215

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir el conjunto de datos en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un clasificador Naive Bayes gaussiano
classifier = GaussianNB()

# Entrenar el clasificador con los datos de entrenamiento
classifier.fit(X_train, y_train)

# Realizar predicciones sobre los datos de prueba
predictions = classifier.predict(X_test)

# Calcular la precisión del clasificador
accuracy = accuracy_score(y_test, predictions)
print("Precisión del clasificador Naive Bayes:", accuracy)
