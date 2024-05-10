#Salazar Chavez Cristian Uriel
#21310215         

import nltk
from nltk.corpus import treebank
from nltk.grammar import PCFG
from nltk.probability import MLEProbDist

# Descargar el corpus Treebank de NLTK (necesario solo la primera vez)
nltk.download('treebank')

# Obtener las producciones de las frases del corpus Treebank
productions = []
for tree in treebank.parsed_sents():
    productions += tree.productions()

# Crear una gramática PCFG utilizando las producciones
pcfg = PCFG.from_productions(productions)

# Entrenar un modelo de distribución de probabilidad MLE para la gramática PCFG
mle_prob_dist = MLEProbDist(pcfg)

# Ejemplo: calcular la probabilidad de una producción específica
production = nltk.grammar.Production(nltk.Nonterminal('NP'), ['DT', 'NN'])
print("Probabilidad de la producción", production, ":", mle_prob_dist.prob(production))
