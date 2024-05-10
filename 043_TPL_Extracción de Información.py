#Salazar Chavez Cristian Uriel
#21310215         

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ne_chunk

# Ejemplo de texto
text = """
Barack Obama was born in Hawaii. He was the 44th President of the United States.
Joe Biden is the current President. He was Vice President under Obama.
"""

# Tokenizar el texto en oraciones y palabras
sentences = sent_tokenize(text)
words = [word_tokenize(sentence) for sentence in sentences]

# Etiquetar partes del habla (POS tagging) para cada palabra en cada oración
tagged_words = [pos_tag(sentence) for sentence in words]

# Identificar entidades nombradas (NER) en el texto
named_entities = [ne_chunk(tagged_word) for tagged_word in tagged_words]

# Extraer nombres de personas y sus roles
people_roles = []
for named_entity in named_entities:
    for chunk in named_entity:
        if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
            person = ' '.join([c[0] for c in chunk])
            role = ' '.join([c[0] for c in chunk.subtrees() if c.label() != 'PERSON'])
            people_roles.append((person, role))

# Imprimir los nombres de personas y sus roles
for person, role in people_roles:
    print("Nombre:", person)
    print("Rol:", role)
    print()
