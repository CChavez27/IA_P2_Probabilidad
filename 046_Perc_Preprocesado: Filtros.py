#Salazar Chavez Cristian Uriel
#21310215

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar una imagen
image = cv2.imread('imagen.jpg')

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar un filtro de suavizado Gaussiano
gaussian_blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Aplicar un filtro de mediana
median_blurred = cv2.medianBlur(gray_image, 5)

# Aplicar un filtro bilateral
bilateral_filtered = cv2.bilateralFilter(gray_image, 9, 75, 75)

# Mostrar las imágenes resultantes
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Imagen en escala de grises')

plt.subplot(2, 2, 2)
plt.imshow(gaussian_blurred, cmap='gray')
plt.title('Suavizado Gaussiano')

plt.subplot(2, 2, 3)
plt.imshow(median_blurred, cmap='gray')
plt.title('Suavizado Mediano')

plt.subplot(2, 2, 4)
plt.imshow(bilateral_filtered, cmap='gray')
plt.title('Filtro Bilateral')

plt.tight_layout()
plt.show()
