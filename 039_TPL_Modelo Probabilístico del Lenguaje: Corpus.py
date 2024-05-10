#Salazar Chavez Cristian Uriel
#21310215          

import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import random

# Descargar el tokenizer y el corpus (necesario solo la primera vez)
nltk.download('punkt')
nltk.download('brown')

# Cargar el corpus Brown de NLTK
corpus = nltk.corpus.brown

# Obtener las palabras del corpus
words = corpus.words()

# Tokenizar las palabras y calcular la distribución de frecuencia
word_freq = FreqDist(words)

# Ejemplo de las 10 palabras más frecuentes en el corpus
print("10 palabras más frecuentes en el corpus:")
print(word_freq.most_common(10))

# Función para generar texto aleatorio basado en el modelo probabilístico
def generate_text(model, num_words=50):
    generated_text = []
    for _ in range(num_words):
        word = model.generate()
        generated_text.append(word)
    return ' '.join(generated_text)

# Clase para el modelo probabilístico del lenguaje
class LanguageModel:
    def __init__(self, word_freq):
        self.words = list(word_freq.keys())
        self.probs = np.array(list(word_freq.values())) / sum(word_freq.values())

    def generate(self):
        return random.choices(self.words, weights=self.probs)[0]

# Crear el modelo probabilístico del lenguaje
model = LanguageModel(word_freq)

# Generar texto aleatorio basado en el modelo
generated_text = generate_text(model, num_words=50)
print("\nTexto generado:")
print(generated_text)
