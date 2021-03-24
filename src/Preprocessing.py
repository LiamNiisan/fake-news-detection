#----------------- Libraries -------------------#
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
        """
        This function gets data from and excel file and returns a list of text.

        Args:
            path (string): file location

        Returns:
            numpy : list of text
        """

        data = pd.read_excel(path)
        corpus_df = data['title'] + data['text']
        corpus = corpus_df.to_numpy()

        return corpus

    def remove_punct(self, corpus):
        """
        This function removes punctuation from text.

        Args:
            corpus (numpy): list of text

        Returns:
            numpy: list of text
        """

        for i in range(len(corpus)):
            text = corpus[i]
            text = text.lower()  # Supprimer les majuscules
            # Remplacer apostrophe par une espace
            text = text.replace("'", " ")
            text = text.replace("\n", " ")  # Supprimer les retours chariot
            # Supprimer la ponctuation
            text = text.translate(str.maketrans("", "", string.punctuation))
            # Éliminations des caractères spéciaux et accentués
            text = text.encode('utf-8').decode('utf-8')
            #text = text.str.replace(r'\W',"")
            corpus[i] = text

        return corpus

    def tokenize(self, corpus):
        """
        This function tokenizes the text.

        Args:
            corpus (numpy): list of text

        Returns:
            numpy: list of tokenized words
        """

        return np.array([nltk.word_tokenize(corpus[i]) for i in range(len(corpus))])

    def remove_stop_words(self, corpus):
        """
        This function removes stop words from text.

        Args:
            corpus (numpy): list of tokenized words

        Returns:
            numpy: list of filtered words
        """

        stop_words = set(stopwords.words('english'))
        filtered_corpus = np.array([None] * len(corpus))
        for i in range(len(corpus)):
            filtered_corpus[i] = np.array(
                [word for word in corpus[i] if not word in stop_words])

        return filtered_corpus

    def lemmatize(self, corpus):
        """
        This function lemmatizes the words in the text.

        Args:
            corpus (numpy): list of tokenized words

        Returns:
            numpy: list of text
        """

        lemma_corpus = np.array([None] * len(corpus))
        wordnet_lemmatizer = WordNetLemmatizer()

        for i in range(len(corpus)):
            text = corpus[i]
            lemma_text = np.array([None] * len(text))
            for j in range(len(text)):
                word = text[j]
                word = wordnet_lemmatizer.lemmatize(word, pos="n")
                word = wordnet_lemmatizer.lemmatize(word, pos="v")
                word = wordnet_lemmatizer.lemmatize(word, pos=("a"))
                lemma_text[j] = word
            lemma_corpus[i] = lemma_text

        return lemma_corpus

    def textjoin(self, corpus):
        """
        This function put a list of words in a list of text.

        Args:
            corpus (numpy): list of tokenized words

        Returns:
            numpy: list of text
        """
        textjoin_corpus = np.array([None] * len(corpus))

        for i in range(len(corpus)):
            textjoin_corpus[i] = ' '.join(corpus[i])

        return textjoin_corpus


    def preprocess(self, data):
        """
        This function preprocesses the data by applying the following operations:

        1- Removing punctuation
        2- Tokenizing
        3- Removing stop words
        4- Lemmatizing the words and merging them into text again
        5- Text join

        Args:
            data (numpy): list of text

        Returns:
            numpy: list of preprocessed text
        """

        corpus = self.remove_punct(data)
        corpus = self.tokenize(corpus)
        corpus = self.remove_stop_words(corpus)
        corpus = self.lemmatize(corpus)
        corpus = self.textjoin(corpus)
        return corpus
