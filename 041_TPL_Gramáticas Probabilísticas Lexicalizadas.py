#Salazar Chavez Cristian Uriel
#21310215         

import nltk
from nltk.corpus import treebank
from nltk.grammar import PCFG, induce_pcfg
from nltk.probability import MLEProbDist

# Descargar el corpus Treebank de NLTK (necesario solo la primera vez)
nltk.download('treebank')

# Obtener el corpus Treebank
corpus = treebank.parsed_sents()

# Extraer las producciones binarias del corpus Treebank
binary_productions = []
for tree in corpus:
    binary_productions += tree.productions()

# Inducir una gramática PCFG lexicalizada a partir de las producciones binarias
lpcfg = induce_pcfg(nltk.Nonterminal('S'), binary_productions)

# Entrenar un modelo de distribución de probabilidad MLE para la gramática LPCFG
mle_prob_dist = MLEProbDist(lpcfg)

# Ejemplo: calcular la probabilidad de una producción específica
production = nltk.grammar.Production(nltk.Nonterminal('NP'), ['DT', 'NN'])
print("Probabilidad de la producción", production, ":", mle_prob_dist.prob(production))
