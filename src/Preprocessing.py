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

        for i in range(corpus_df.shape[0]):
            text = str(corpus_df.iloc[i])
            text = text.lower()  # Supprimer les majuscules
            text = text.replace("'", " ")  # Remplacer apostrophe par une espace
            text = text.replace("\n", " ") # Supprimer les retours chariot
            text = text.translate(str.maketrans("", "", string.punctuation)) # Supprimer la ponctuation
            text = text.encode('utf-8').decode('utf-8') # Éliminations des caractères spéciaux et accentués
            corpus_df.iloc[i] = text

        return corpus_df
        

    def tokenize(self, corpus_df):

        return [nltk.word_tokenize(corpus_df.iloc[i]) for i in range(corpus_df.shape[0])]


    def remove_stop_words(self, corpus):

        stop_words = set(stopwords.words('english'))
        filtered_corpus = []
        for text in corpus:
            filtered_corpus.append([word for word in text if not word in stop_words])
            
        return filtered_corpus

    
    def lemmatize(self, corpus):

        lemma_corpus = []
        wordnet_lemmatizer = WordNetLemmatizer()

        for text in corpus:
            lemma_text = []    
            for word in text:
                word = wordnet_lemmatizer.lemmatize(word, pos = "n")
                word = wordnet_lemmatizer.lemmatize(word, pos = "v")
                word = wordnet_lemmatizer.lemmatize(word, pos = ("a"))
                lemma_text.append(word)       
            lemma_corpus.append(lemma_text)

        return lemma_corpus


    def run(self, path):

        data = self.get_data(path)
        corpus_df = self.remove_punct(data)
        corpus = np.array(self.tokenize(corpus_df))
        corpus = np.array(self.remove_stop_words(corpus))
        corpus = np.array(self.lemmatize(corpus))

        return corpus


