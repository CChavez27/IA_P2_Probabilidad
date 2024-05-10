#Salazar Chavez Cristian Uriel
#21310215

import cv2

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

# Inicializar el primer frame
_, frame1 = cap.read()
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Configurar el detector de movimiento
motion_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    # Capturar el siguiente frame
    _, frame2 = cap.read()
    current_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Obtener la máscara de movimiento
    mask = motion_detector.apply(frame2)

    # Filtrar el ruido y encontrar contornos
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar rectángulos alrededor de los objetos en movimiento
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar el frame con los objetos en movimiento
    cv2.imshow('Motion Detection', frame2)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
