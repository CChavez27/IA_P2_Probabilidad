#Salazar Chavez Cristian Uriel
#21310215     

import speech_recognition as sr

# Crear un objeto reconocedor
recognizer = sr.Recognizer()

# Configurar el micrófono como fuente de entrada
microphone = sr.Microphone()

# Escuchar del micrófono
with microphone as source:
    print("Di algo...")
    audio = recognizer.listen(source)

# Realizar el reconocimiento de voz
try:
    print("Reconociendo...")
    text = recognizer.recognize_google(audio, language="es-ES")
    print("Texto reconocido:", text)
except sr.UnknownValueError:
    print("No se pudo entender el habla")
except sr.RequestError as e:
    print("Error al solicitar resultados; {0}".format(e))
