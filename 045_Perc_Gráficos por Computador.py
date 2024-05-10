#Salazar Chavez Cristian Uriel
#21310215        

import numpy as np
import matplotlib.pyplot as plt

# Crear una matriz de píxeles (imagen) con valores aleatorios entre 0 y 255
image = np.random.randint(0, 256, size=(100, 100))

# Mostrar la imagen utilizando matplotlib
plt.imshow(image, cmap='gray')
plt.axis('off')  # Ocultar ejes
plt.title('Imagen generada')
plt.show()
