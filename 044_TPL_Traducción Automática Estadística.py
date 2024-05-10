#Salazar Chavez Cristian Uriel
#21310215       

import numpy as np

class SMT1Translator:
    def __init__(self, source_vocab, target_vocab):
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.translation_probs = np.random.rand(len(source_vocab), len(target_vocab))

    def translate(self, source_sentence):
        source_indices = [self.source_vocab.index(word) for word in source_sentence.split()]
        target_indices = np.argmax(self.translation_probs[source_indices], axis=1)
        target_sentence = ' '.join([self.target_vocab[index] for index in target_indices])
        return target_sentence

# Ejemplo de uso
source_vocab = ['cat', 'dog', 'house']
target_vocab = ['gato', 'perro', 'casa']

translator = SMT1Translator(source_vocab, target_vocab)

source_sentence = 'cat house'
target_sentence = translator.translate(source_sentence)
print("Traducción de '{}': {}".format(source_sentence, target_sentence))
