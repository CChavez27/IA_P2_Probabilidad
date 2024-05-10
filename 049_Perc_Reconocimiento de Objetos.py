#Salazar Chavez Cristian Uriel
#21310215     

import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util

# Cargar el modelo preentrenado de detección de objetos de TensorFlow
model_path = 'ssd_mobilenet_v2_coco'
model = tf.saved_model.load(model_path)

# Cargar el archivo de configuración del mapa de etiquetas
label_map_path = 'mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

# Función para realizar la detección de objetos en una imagen
def detect_objects(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    return detections

# Cargar una imagen de prueba
image_path = 'imagen.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Realizar la detección de objetos en la imagen
detections = detect_objects(image_rgb)

# Dibujar los cuadros delimitadores de los objetos detectados en la imagen
for i in range(len(detections['detection_scores'])):
    class_id = int(detections['detection_classes'][i])
    score = detections['detection_scores'][i]
    if score > 0.5:
        label = category_index[class_id]['name']
        bbox = detections['detection_boxes'][i]
        h, w, _ = image.shape
        ymin, xmin, ymax, xmax = bbox
        xmin = int(xmin * w)
        xmax = int(xmax * w)
        ymin = int(ymin * h)
        ymax = int(ymax * h)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, f'{label}: {score:.2f}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostrar la imagen con los objetos detectados
cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
