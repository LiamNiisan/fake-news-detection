#   Conception de processus d’analyse textuelle permettant de déterminer le niveau de validité d’articles
#   Auteurs : Nicolas Clermont, Badr Jaidi et Jonathan Boudreau
#   Date : 18/02/21
#   Description : Le script traite un fichier xlsx pour supprimer la ponctuation,
#   segmenter les fichiers, supprimer les mots vides et effectuer la lemmatisation.

# librairies utilisees
import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from Preprocessing import Preprocessing


def kfold_decompose(data, kfold_n):

    X = np.array(data)
    kf = KFold(n_splits=kfold_n, random_state=2, shuffle=True)
    data_output = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        data_output.append([X_train, X_test])

    return data_output


def create_folders(data_label_true, data_label_false, kfold_n, cwd='../data/'):

    training = cwd + 'training/'
    os.mkdir(training)
    prp = Preprocessing()

    for i in tqdm(range(kfold_n)):

        data_set_path = cwd + training + 'data_set_' + str(i+1)

        os.mkdir(data_set_path)
        os.mkdir(data_set_path+'/test')
        os.mkdir(data_set_path+'/train')
        os.mkdir(data_set_path+'/train/vc')

        pd.DataFrame(data_label_true[i][0]).to_excel(
            data_set_path+'/test/true.xlsx')
        pd.DataFrame(data_label_false[i][0]).to_excel(
            data_set_path+'/test/false.xlsx')

        X_train, X_val = train_test_split(
            data_label_true[i][1], test_size=0.20, random_state=1)
        pd.DataFrame(prp.preprocess(X_train)).to_excel(data_set_path+'/train/true.xlsx')
        pd.DataFrame(X_val).to_excel(data_set_path+'/train/vc/true.xlsx')

        X_train, X_val = train_test_split(
            data_label_false[i][1], test_size=0.20, random_state=1)
        pd.DataFrame(prp.preprocess(X_train)).to_excel(data_set_path+'/train/false.xlsx')
        pd.DataFrame(X_val).to_excel(data_set_path+'/train/vc/false.xlsx')


def main(true_set, false_set, kfold_n=5):

    data_label_true = kfold_decompose(true_set, kfold_n)
    data_label_false = kfold_decompose(false_set, kfold_n)

    create_folders(data_label_true, data_label_false, kfold_n)
