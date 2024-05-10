#Salazar Chavez Cristian Uriel
#21310215        

import numpy as np
import cv2

# Crear una imagen de fondo
background = np.zeros((400, 400, 3), dtype=np.uint8)
background[:] = (255, 255, 255)  # Color de fondo blanco

# Dibujar una textura (patrón de líneas diagonales)
for i in range(0, 400, 20):
    cv2.line(background, (0, i), (400, i), (0, 0, 0), thickness=2)
    cv2.line(background, (i, 0), (i, 400), (0, 0, 0), thickness=2)

# Crear una máscara para la sombra
mask = np.zeros((400, 400), dtype=np.uint8)
cv2.circle(mask, (200, 200), 100, (255, 255, 255), thickness=-1)

# Crear una imagen con sombra
shadow = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
shadow = cv2.bitwise_and(shadow, mask)

# Mostrar las imágenes resultantes
cv2.imshow('Textura', background)
cv2.imshow('Sombra', shadow)
cv2.waitKey(0)
cv2.destroyAllWindows()
