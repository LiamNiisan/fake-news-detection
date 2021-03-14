#   Conception de processus d’analyse textuelle permettant de déterminer le niveau de validité d’articles
#   Auteurs : Nicolas Clermont, Badr Jaidi et Jonathan Boudreau
#   Date : 18/02/21
#   Description : Le script traite un fichier xlsx pour supprimer la ponctuation,
#   segmenter les fichiers, supprimer les mots vides et effectuer la lemmatisation.

# librairies utilisees
import numpy as np
from sklearn.model_selection import KFold
import csv
import os, sys
from sklearn.model_selection import train_test_split

# kfold decomposition
X = np.array(source)
kf = KFold(n_splits=5,random_state=2, shuffle=True)
kf.get_n_splits(X)
donne_true  = []
for train_index, test_index in kf.split(source):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    donne_true.append([X_train, X_test])

Y = np.array(source)
kf = KFold(n_splits=5,random_state=2, shuffle=True)
kf.get_n_splits(Y)
donne_false  = []
for train_index, test_index in kf.split(source):
    print("TRAIN:", train_index, "TEST:", test_index)
    Y_train, Y_test = Y[train_index], Y[test_index]
    donne_false.append([Y_train, Y_test])    

# create training folder
cwd = os.getcwd()
training = cwd + '\data\\training' 
print(training)
os.mkdir(training)

# create kfold folder in training folder
kfold_n = 5
for i in range(kfold_n):
    data_set_path = cwd + '\data\\training\data_set_' + str(i+1)
    os.mkdir(data_set_path)
    os.mkdir(data_set_path+'\\test')
    pd.DataFrame(donne_true[i][0]).to_excel(data_set_path+'\\test\\true.xlsx')  
    pd.DataFrame(donne_false[i][0]).to_excel(data_set_path+'\\test\\false.xlsx')   
    os.mkdir(data_set_path+'\\train')
    os.mkdir(data_set_path+'\\train\\vc')
    X_train, X_val = train_test_split(donne_true[i][1], test_size=0.20, random_state = 1) 
    pd.DataFrame(X_train).to_excel(data_set_path+'\\train\\true.xlsx')  
    pd.DataFrame(X_val).to_excel(data_set_path+'\\train\\vc\\true.xlsx')
    Y_train, Y_val = train_test_split(donne_false[i][1], test_size=0.20, random_state = 1) 
    pd.DataFrame(Y_train).to_excel(data_set_path+'\\train\\false.xlsx')  
    pd.DataFrame(Y_val).to_excel(data_set_path+'\\train\\vc\\false.xlsx')   
    




