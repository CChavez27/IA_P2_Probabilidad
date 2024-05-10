#Salazar Chavez Cristian Uriel
#21310215      

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Función para extraer características de una imagen (en este caso, solo se usa la intensidad de los píxeles)
def extract_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image.flatten()

# Cargar imágenes de líneas etiquetadas y no etiquetadas
labeled_lines = []
unlabeled_lines = []

# Cargar imágenes de líneas etiquetadas
for i in range(1, 101):
    image = cv2.imread(f'labeled_lines/line_{i}.png')
    labeled_lines.append((image, 1))  # Etiqueta 1 para líneas etiquetadas

# Cargar imágenes de líneas no etiquetadas
for i in range(1, 101):
    image = cv2.imread(f'unlabeled_lines/line_{i}.png')
    unlabeled_lines.append((image, 0))  # Etiqueta 0 para líneas no etiquetadas

# Combinar líneas etiquetadas y no etiquetadas y dividirlas en conjuntos de entrenamiento y prueba
data = labeled_lines + unlabeled_lines
X = [extract_features(image) for image, _ in data]
y = [label for _, label in data]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de clasificación (SVM en este caso)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predecir etiquetas para el conjunto de prueba
y_pred = svm_model.predict(X_test)

# Calcular precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)
