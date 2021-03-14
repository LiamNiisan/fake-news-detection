#   Conception de processus d’analyse textuelle permettant de déterminer le niveau de validité d’articles
#   Auteurs : Nicolas Clermont, Badr Jaidi et Jonathan Boudreau
#   Date : 18/02/21
#   Description : Le script traite un fichier xlsx pour supprimer la ponctuation,
#   segmenter les fichiers, supprimer les mots vides et effectuer la lemmatisation.

# librairies utilisees
import os as os
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

os.chdir("C:/Users/Nick/IA")  # remplacer \ par /
print("Repertoire de travail actuel : ", os.getcwd())

# importer xlsx
xlsx = pd.read_excel('test.xlsx')
print(xlsx.head(3))  # en-tête du tableau
colnames = list(xlsx.columns)  # Noms des colonnes
ncol = len(xlsx.columns)  # Nombre de colonnes
nrow = len(xlsx.index)  # Nombre de lignes

# supprimer la ponctuation
corpus = xlsx.iloc[:, 1]
source = xlsx.iloc[:, 1]
for i in range(len(corpus)):
    tmp = str(corpus[i])
    tmp = tmp.lower()  # Supprimer les majuscules
    tmp = tmp.replace("'", " ")  # Remplacer apostrophe par une espace
    tmp = tmp.replace("\n", " ") # Supprimer les retours chariot
    tmp = tmp.translate(str.maketrans("", "", string.punctuation)) # Supprimer la ponctuation
    tmp = tmp.encode('utf-8').decode('utf-8') # Éliminations des caractères spéciaux et accentués
    corpus[i] = tmp
print(corpus)

# tokenize
for i in range(len(corpus)):
    corpus_tokens = [nltk.word_tokenize(sent) for sent in corpus]
    print(corpus_tokens)

# stopword
mots_vides = set(stopwords.words('english'))
corpus_filtre = []
for i in range(len(corpus_tokens)):
    corpus_filtre.append([mots for mots in corpus_tokens[i] if not mots in mots_vides])
    print(corpus_filtre)

#lemmatize
corpus_lemma = []
wordnet_lemmatizer = WordNetLemmatizer()
for txt in corpus_filtre:    
    for m in txt:
        mot1 = wordnet_lemmatizer.lemmatize(m, pos = "n")
        mot2 = wordnet_lemmatizer.lemmatize(mot1, pos = "v")
        mot3 = wordnet_lemmatizer.lemmatize(mot2, pos = ("a"))
        corpus_lemma.append(mot3)
print(corpus_lemma)