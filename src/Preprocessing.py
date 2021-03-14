#   Conception de processus d’analyse textuelle permettant de déterminer le niveau de validité d’articles
#   Auteurs : Nicolas Clermont, Badr Jaidi et Jonathan Boudreau
#   Date : 18/02/21
#   Description : Le script traite un fichier xlsx pour supprimer la ponctuation,
#   segmenter les fichiers, supprimer les mots vides et effectuer la lemmatisation.

# librairies utilisees
import os as os
import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class Preprocessing:

    def get_data(self, path):

        data = pd.read_excel(path)

        return data


    def remove_punct(self, data):

        corpus_df = data['title'] + data['text']
        corpus = corpus_df.to_numpy()

        for i in range(len(corpus)):
            text = corpus[i]
            text = text.lower()  # Supprimer les majuscules
            text = text.replace("'", " ")  # Remplacer apostrophe par une espace
            text = text.replace("\n", " ") # Supprimer les retours chariot
            text = text.translate(str.maketrans("", "", string.punctuation)) # Supprimer la ponctuation
            text = text.encode('utf-8').decode('utf-8') # Éliminations des caractères spéciaux et accentués
            corpus[i] = text

        return corpus
        

    def tokenize(self, corpus):

        return np.array([nltk.word_tokenize(corpus[i]) for i in range(len(corpus))])


    def remove_stop_words(self, corpus):

        stop_words = set(stopwords.words('english'))
        filtered_corpus = np.array([None] * len(corpus))
        for i in range(len(corpus)):
            filtered_corpus[i] = np.array([word for word in corpus[i] if not word in stop_words])
            
        return filtered_corpus

    
    def lemmatize(self, corpus):

        lemma_corpus = np.array([None] * len(corpus))
        wordnet_lemmatizer = WordNetLemmatizer()

        for i in range(len(corpus)):
            text = corpus[i]
            lemma_text = np.array([None] * len(text)) 
            for j in range(len(text)):
                word = text[j]
                word = wordnet_lemmatizer.lemmatize(word, pos = "n")
                word = wordnet_lemmatizer.lemmatize(word, pos = "v")
                word = wordnet_lemmatizer.lemmatize(word, pos = ("a"))
                lemma_text[j] = word  
            lemma_corpus[i] = lemma_text

        return lemma_corpus


    def run(self, path):

        data = self.get_data(path)
        corpus = self.remove_punct(data)
        corpus = self.tokenize(corpus)
        corpus = self.remove_stop_words(corpus)
        corpus = self.lemmatize(corpus)

        return corpus


