#Salazar Chavez Cristian Uriel
#21310215        

import cv2
import pytesseract

# Configuración de Tesseract (ubicación del ejecutable)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Cargar la imagen
image = cv2.imread('handwritten_text.png')

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar umbral (binarización) para resaltar los caracteres
_, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Realizar OCR en la imagen
text = pytesseract.image_to_string(threshold_image, lang='eng')

# Mostrar el texto reconocido
print("Texto reconocido:")
print(text)
