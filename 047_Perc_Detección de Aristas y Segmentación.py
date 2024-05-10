#Salazar Chavez Cristian Uriel
#21310215        

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar una imagen en escala de grises
image = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar el operador de Sobel para detección de bordes
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
edges = np.sqrt(sobel_x**2 + sobel_y**2)

# Aplicar umbralización para obtener una imagen binaria de los bordes
edges_binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)[1]

# Realizar la segmentación utilizando la umbralización de Otsu
_, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Mostrar las imágenes resultantes
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagen Original')

plt.subplot(2, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Detección de Bordes (Sobel)')

plt.subplot(2, 2, 3)
plt.imshow(edges_binary, cmap='gray')
plt.title('Bordes Binarios')

plt.subplot(2, 2, 4)
plt.imshow(segmented_image, cmap='gray')
plt.title('Segmentación (Umbralización de Otsu)')

plt.tight_layout()
plt.show()
